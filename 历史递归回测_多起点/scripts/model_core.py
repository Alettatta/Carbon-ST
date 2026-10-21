"""180-step recursive hindcast with historical labels isolated from inference."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'cache/numba')
sys.path.insert(0, str(ROOT / 'inputs/model_source'))
import hashlib
import json
import logging
import time
from datetime import date, timedelta
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from libcity.model.traffic_flow_prediction.PDFormer import PDFormer

HORIZON = 180
WINDOW = 12
MARKS = [1, 7, 30, 90, 180]
DISPLAY_START = date(2019, 1, 3)
MODEL_START = date(2019, 1, 1)

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def calendar_features(indices, nodes):
    features = np.zeros((len(indices), nodes, 8), dtype=np.float32)
    for i, row in enumerate(indices):
        weekday = (MODEL_START + timedelta(days=int(row))).weekday()
        features[i, :, 1 + weekday] = 1
    return features

def load_model():
    cfg = json.loads((ROOT / 'inputs/carbonst_config.json').read_text(encoding='utf-8'))
    features_dir = ROOT / 'inputs/training_features'
    with np.load(next(features_dir.glob('*.npz'))) as windows:
        train = windows['x_train']
        mean, std = float(train[..., :1].mean()), float(train[..., :1].std())
        counts = {k: int(windows['x_' + k].shape[0]) for k in ['train', 'val', 'test']}
        # Last validation target index: (train+val-1) + 12 + 11.
        holdout_start = counts['train'] + counts['val'] + 23
        assert counts == {'train': 1662, 'val': 207, 'test': 208}
        assert abs(mean - 0.8571480798035013) < 1e-12
        assert abs(std - 0.6533952796190988) < 1e-12
        reference_first_x = train[0].copy()
    geo = pd.read_csv(ROOT / 'inputs/spatial/CarbonProvince.geo')
    provinces = geo['province_en'].tolist()
    rel = pd.read_csv(ROOT / 'inputs/spatial/CarbonProvince.rel')
    adj = np.zeros((31, 31), dtype=np.float32)
    for row in rel.itertuples():
        adj[int(row.origin_id), int(row.destination_id)] = 1
    # Same dense eigendecomposition and eigenvector slice as the original executor.
    sparse_adj = sp.coo_matrix(adj)
    degrees = np.array(sparse_adj.sum(1))
    isolated = int(np.sum(np.where(degrees, 0, 1)))
    with np.errstate(divide='ignore'):
        inverse = np.power(degrees, -0.5).flatten()
    inverse[np.isinf(inverse)] = 0.
    diagonal = sp.diags(inverse)
    laplacian = sp.eye(31) - sparse_adj.dot(diagonal).transpose().dot(diagonal).tocoo()
    eigenvalues, eigenvectors = np.linalg.eig(laplacian.toarray())
    eigenvectors = np.real(eigenvectors[:, eigenvalues.argsort()])
    lap = torch.from_numpy(eigenvectors[:, isolated + 1:isolated + 9]).float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg.update({'device': device, 'dataset': 'CarbonProvince', 'lape_dim': 8})
    data_feature = {'scaler': None, 'num_nodes': 31, 'feature_dim': 9, 'ext_dim': 8,
                    'output_dim': 1, 'num_batches': 104, 'adj_mx': adj, 'sd_mx': adj.copy(),
                    'sh_mx': np.load(features_dir / 'CarbonProvince_shortest_hops.npy'),
                    'dtw_matrix': np.load(features_dir / 'dtw_CarbonProvince_daily.npy'),
                    'pattern_keys': np.load(features_dir / 'pattern_keys_kshape_CarbonProvince_14_3_16_5.npy')}
    model = PDFormer(cfg, data_feature).to(device).eval()
    checkpoint = torch.load(ROOT / 'inputs/epoch165.tar', map_location='cpu', weights_only=False)
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint[0] if isinstance(checkpoint, (tuple, list)) else checkpoint
    assert checkpoint['epoch'] == 165
    model.load_state_dict(state, strict=True)
    np.savez(ROOT / 'cache/restored_features.npz', adjacency=adj, laplacian_pe=lap.numpy(), mean=mean, std=std)
    return model, lap.to(device), mean, std, provinces, holdout_start, reference_first_x, cfg

def initial_windows(values, origins, mean, std):
    result = np.zeros((len(origins), WINDOW, 31, 9), dtype=np.float32)
    for i, origin in enumerate(origins):
        result[i, ..., :1] = ((values[origin-WINDOW:origin] - mean) / std)[..., None]
        result[i, ..., 1:] = calendar_features(range(origin-WINDOW, origin), 31)
    return result

def recursive_forecast(model, lap, initial, origins, mean, std, horizon=HORIZON, progress=False):
    """Receives only initial windows and row indices, never scoring labels."""
    x = torch.as_tensor(initial.copy(), device=lap.device)
    forecasts = []
    with torch.no_grad():
        for step in range(horizon):
            output = model({'X': x}, lap)
            prediction = output[:, 0, :, :1]
            if not torch.isfinite(prediction).all():
                bad = (~torch.isfinite(prediction)).nonzero()[0].tolist()
                raise FloatingPointError('Nonfinite forecast: origin={}, horizon={}, node={}'.format(
                    int(origins[bad[0]]) + 1, step + 1, bad[1]))
            forecasts.append((prediction[..., 0] * std + mean).cpu().numpy().astype(np.float64))
            appended = torch.zeros((len(origins), 1, 31, 9), device=lap.device)
            appended[:, 0, :, :1] = prediction
            appended[:, 0, :, 1:] = torch.as_tensor(calendar_features(origins + step, 31), device=lap.device)
            next_x = torch.cat((x[:, 1:], appended), dim=1)
            assert torch.equal(next_x[:, :-1], x[:, 1:])
            assert torch.equal(next_x[:, -1, :, :1], prediction)
            x = next_x
            if progress and ((step + 1) % 30 == 0 or step == 0):
                logging.info('Origins=%d, recursive horizon=%d/%d', len(origins), step+1, horizon)
    return np.stack(forecasts, axis=1)

def metrics(actual, predicted):
    error = predicted - actual
    nonzero = actual != 0
    return {'MAE': float(np.mean(np.abs(error))), 'RMSE': float(np.sqrt(np.mean(error**2))),
            'MAPE_percent': float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
            'scored_values': int(actual.size), 'mape_values': int(nonzero.sum())}

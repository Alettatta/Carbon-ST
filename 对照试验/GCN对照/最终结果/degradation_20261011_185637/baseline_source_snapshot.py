"""GCN baseline for province-level daily carbon-emission forecasting.

Uses the same historical data, raw-time split, 12-day horizon, and rolling test
origins as the ARIMA/SARIMA comparison folders. All outputs stay beside this file.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "复现" / "raw_inputs" / "历史与预测.csv"
GEO_SOURCE = OUT / "CarbonProvince.geo"
REL_SOURCE = OUT / "CarbonProvince.rel"
CONFIG_PATH = OUT / "gcn_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()



def load_normalized_adjacency(provinces):
    geo = pd.read_csv(GEO_SOURCE)
    if geo["province_en"].tolist() != provinces:
        raise ValueError("CarbonProvince.geo order does not match the carbon time-series columns")
    rel = pd.read_csv(REL_SOURCE)
    if set(rel["type"].astype(str)) != {"geo"}:
        raise ValueError("Unexpected relation types in CarbonProvince.rel")
    n = len(provinces)
    adjacency = np.zeros((n, n), dtype=np.float32)
    for row in rel.itertuples(index=False):
        i, j = int(row.origin_id), int(row.destination_id)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("Relation graph contains an out-of-range province id")
        adjacency[i, j] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    degree_without_self = adjacency.sum(axis=1)
    isolates = [provinces[i] for i in np.flatnonzero(degree_without_self == 0)]
    adjacency += np.eye(n, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.power(degree, -0.5)
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    info = {"source_geo": str(GEO_SOURCE), "source_rel": str(REL_SOURCE),
            "geo_sha256": sha256(GEO_SOURCE), "rel_sha256": sha256(REL_SOURCE),
            "province_order": provinces, "raw_relation_rows": int(len(rel)),
            "undirected_edges": int(np.triu(np.maximum(adjacency - np.eye(n, dtype=np.float32), 0), 1).sum()),
            "isolated_nodes_before_self_loops": isolates,
            "self_loops_added": True,
            "normalization": "D^-1/2 (A + I) D^-1/2; symmetrize source relations first"}
    return normalized.astype(np.float32), info

def setup_logging():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(OUT / "training.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphConvLayer(nn.Module):
    """Ahat X W graph convolution layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        return self.linear(torch.matmul(adjacency, x))


class GCNForecaster(nn.Module):
    """Two-layer GCN over provinces; 12 lag values per node -> 12-day output."""
    def __init__(self, num_nodes, input_window, output_window, hidden_size, num_layers, dropout, adjacency):
        super().__init__()
        if num_layers != 2:
            raise ValueError("This baseline defines exactly two graph-convolution layers")
        self.num_nodes = num_nodes
        self.output_window = output_window
        self.register_buffer("adjacency", adjacency.float())
        self.gc1 = GraphConvLayer(input_window, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_window)

    def forward(self, x):
        # [batch, time, province] -> [batch, province, time features]
        z = x.transpose(1, 2)
        z = torch.relu(self.gc1(z, self.adjacency))
        z = self.dropout(z)
        z = torch.relu(self.gc2(z, self.adjacency))
        return self.head(z).transpose(1, 2)


def metric_pack(actual, predicted):
    e = predicted - actual
    nz = actual != 0
    return {"MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE_percent": float(np.mean(np.abs(e[nz] / actual[nz])) * 100.0),
            "scored_values": int(actual.size), "mape_values": int(nz.sum())}


def set_epoch_lr(optimizer, epoch):
    """PDFormer CosineLRScheduler schedule: 300 epochs, 5-epoch warmup."""
    base = float(CONFIG["learning_rate"])
    warm = int(CONFIG["lr_warmup_epoch"])
    warm_init = float(CONFIG["lr_warmup_init"])
    min_lr = float(CONFIG["lr_eta_min"])
    total = int(CONFIG["max_epoch"])
    t = epoch
    if t < warm:
        lr = warm_init + t * (base - warm_init) / warm
    else:
        lr = min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * t / total))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def build_windows(values, origins, input_window, output_window):
    x = np.stack([values[o-input_window:o] for o in origins]).astype(np.float32)
    y = np.stack([values[o:o+output_window] for o in origins]).astype(np.float32)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run data/model forward-backward checks only")
    args = parser.parse_args()
    setup_logging()
    started = time.time()
    set_seed(int(CONFIG["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and CONFIG["gpu"] else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Configured GPU training, but CUDA is unavailable in the selected environment")
    logging.info("Output directory: %s", OUT)
    logging.info("Python=%s torch=%s CUDA=%s GPU=%s", sys.version.split()[0], torch.__version__,
                 torch.version.cuda, torch.cuda.get_device_name(0))
    logging.info("Loading source: %s", SOURCE)

    frame = pd.read_csv(SOURCE, encoding="utf-8-sig")
    if len(frame) != int(CONFIG["source_rows"]) or "Day" not in frame or "total" not in frame:
        raise ValueError("Input source row count/schema mismatch")
    provinces = [c for c in frame.columns if c not in ("Day", "total")]
    n_hist = int(CONFIG["historical_rows"])
    if len(provinces) != 31 or n_hist + CONFIG["forecast_tail_rows_excluded"] != len(frame):
        raise ValueError("Expected 31 provinces and 182 excluded tail rows")
    values = frame.loc[:n_hist-1, provinces].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Historical input contains NaN/Inf")
    n_days, n_nodes = values.shape
    input_window, horizon = int(CONFIG["input_window"]), int(CONFIG["output_window"])
    train_end = int(n_days * CONFIG["train_ratio"])
    val_end = int(n_days * (CONFIG["train_ratio"] + CONFIG["validation_ratio"]))
    dates = [(date.fromisoformat(CONFIG["history_start_date"]) + timedelta(days=i)).isoformat() for i in range(n_days)]
    if dates[-1] != CONFIG["history_end_date"]:
        raise ValueError("Historical date mapping does not match confirmed dates")

    # Global standardization fitted only on the training segment, matching PDFormer's standard scaler style.
    mean = float(values[:train_end].mean())
    std = float(values[:train_end].std()) or 1.0
    scaled = (values - mean) / std
    train_origins = list(range(input_window, train_end - horizon + 1))
    val_origins = list(range(train_end, val_end - horizon + 1, int(CONFIG["validation_window_stride"])))
    test_origins = list(range(val_end, n_days - horizon + 1, int(CONFIG["test_window_stride"])))
    if not train_origins or not val_origins or len(test_origins) != 199:
        raise ValueError("Unexpected train/validation/test origin counts")
    xtr, ytr = build_windows(scaled, train_origins, input_window, horizon)
    xval, yval = build_windows(scaled, val_origins, input_window, horizon)
    xte, yte = build_windows(scaled, test_origins, input_window, horizon)
    np.savez_compressed(OUT / "prepared_windows.npz", train_origins=np.asarray(train_origins),
                        validation_origins=np.asarray(val_origins), test_origins=np.asarray(test_origins),
                        x_train_shape=np.asarray(xtr.shape), y_train_shape=np.asarray(ytr.shape))
    history = frame.iloc[:n_hist].copy()
    history.insert(1, "date_inferred", dates)
    history.to_csv(OUT / "history_data_2098.csv", index=False, encoding="utf-8-sig")
    split = {"historical_rows": n_days, "province_count": n_nodes,
             "train_rows": train_end, "validation_rows": val_end-train_end, "test_rows": n_days-val_end,
             "train_date_start": dates[0], "train_date_end": dates[train_end-1],
             "validation_date_start": dates[train_end], "validation_date_end": dates[val_end-1],
             "test_date_start": dates[val_end], "test_date_end": dates[-1],
             "validation_origins": len(val_origins), "test_forecast_origin_count": len(test_origins),
             "horizon_days": horizon, "test_first_target_date": dates[test_origins[0]],
             "test_last_target_date": dates[test_origins[-1]+horizon-1],
             "source_csv_sha256": sha256(SOURCE), "source_rows_excluded_as_unverified_forecasts": 182,
             "province_order": provinces, "date_mapping": "row 1 = 2019-01-03; daily dates assigned by row order"}
    (OUT / "split_manifest.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "effective_config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    env = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
           "cuda_available": torch.cuda.is_available()}
    (OUT / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Data=%d days x %d provinces; train windows=%d, validation windows=%d, test origins=%d; scaler mean=%.8g std=%.8g",
                 n_days, n_nodes, len(train_origins), len(val_origins), len(test_origins), mean, std)

    adjacency, graph_info = load_normalized_adjacency(provinces)
    np.save(OUT / "normalized_adjacency.npy", adjacency)
    pd.DataFrame(adjacency, index=provinces, columns=provinces).to_csv(OUT / "normalized_adjacency.csv", encoding="utf-8-sig")
    graph_info["normalized_adjacency_artifact"] = str(OUT / "normalized_adjacency.npy")
    (OUT / "graph_manifest.json").write_text(json.dumps(graph_info, ensure_ascii=False, indent=2), encoding="utf-8")
    model = GCNForecaster(n_nodes, input_window, horizon, int(CONFIG["hidden_size"]),
                           int(CONFIG["num_layers"]), float(CONFIG["dropout"]), torch.from_numpy(adjacency)).to(device)
    xt = torch.from_numpy(xtr[:min(4, len(xtr))]).to(device)
    yt = model(xt)
    if yt.shape != (len(xt), horizon, n_nodes) or not torch.isfinite(yt).all():
        raise RuntimeError("GCN forward output shape/finite check failed")
    loss = nn.L1Loss()(yt, torch.from_numpy(ytr[:len(xt)]).to(device))
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("GCN backward gradient check failed")
    model.zero_grad(set_to_none=True)
    logging.info("GPU smoke check passed: input=%s forecast=%s", tuple(xt.shape), tuple(yt.shape))
    if args.smoke:
        return

    train_set = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    generator = torch.Generator().manual_seed(int(CONFIG["seed"]))
    train_loader = DataLoader(train_set, batch_size=int(CONFIG["batch_size"]), shuffle=True,
                              num_workers=0, generator=generator, pin_memory=True)
    xval_t = torch.from_numpy(xval).to(device)
    yval_original = yval.astype(np.float64) * std + mean
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["learning_rate"]),
                                  weight_decay=float(CONFIG["weight_decay"]))
    criterion = nn.L1Loss()
    best_val, best_epoch, wait = float("inf"), -1, 0
    history_rows = []
    max_epoch, patience = int(CONFIG["max_epoch"]), int(CONFIG["patience"])
    for epoch in range(max_epoch):
        lr = set_epoch_lr(optimizer, epoch)
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            train_loss = criterion(pred, yb)
            if not torch.isfinite(train_loss):
                raise RuntimeError("Non-finite train loss")
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["max_grad_norm"]))
            optimizer.step()
            losses.append(float(train_loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_scaled = model(xval_t).detach().cpu().numpy().astype(np.float64)
        val_original = val_scaled * std + mean
        val_mae = float(np.mean(np.abs(val_original - yval_original)))
        train_mae_scaled = float(np.mean(losses))
        row = {"epoch": epoch+1, "train_mae_scaled": train_mae_scaled, "validation_MAE": val_mae, "learning_rate": lr}
        history_rows.append(row)
        if val_mae < best_val:
            best_val, best_epoch, wait = val_mae, epoch+1, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch,
                        "validation_MAE": best_val, "scaler_mean": mean, "scaler_std": std,
                        "provinces": provinces, "config": CONFIG}, OUT / "best_model.pt")
        else:
            wait += 1
        if (epoch+1) % 5 == 0 or epoch == 0:
            logging.info("Epoch %d/%d train_MAE_scaled=%.6f validation_MAE=%.8f lr=%.7g wait=%d/%d",
                         epoch+1, max_epoch, train_mae_scaled, val_mae, lr, wait, patience)
        if wait >= patience:
            logging.info("Early stopping at epoch %d; best_epoch=%d best_validation_MAE=%.8f", epoch+1, best_epoch, best_val)
            break
    pd.DataFrame(history_rows).to_csv(OUT / "epoch_history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(OUT / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(xte).to(device)).cpu().numpy().astype(np.float64)
    actual = np.stack([values[o:o+horizon] for o in test_origins]).astype(np.float64)
    predicted = pred_scaled * std + mean
    if not np.isfinite(predicted).all():
        raise RuntimeError("Test prediction contains NaN/Inf")
    overall = metric_pack(actual, predicted)
    by_horizon = [{"horizon_day": h+1, **metric_pack(actual[:,h,:], predicted[:,h,:])} for h in range(horizon)]
    by_province = [{"province": province, **metric_pack(actual[:,:,j], predicted[:,:,j])} for j,province in enumerate(provinces)]
    pd.DataFrame(by_horizon).to_csv(OUT / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(by_province).to_csv(OUT / "metrics_by_province.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(OUT / "test_predictions.npz", actual=actual, predicted=predicted,
                        origins=np.asarray(test_origins), provinces=np.asarray(provinces),
                        forecast_dates=np.asarray([[dates[o+h] for h in range(horizon)] for o in test_origins]))
    with (OUT / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["origin_date","target_date","horizon_day","province","actual","predicted"])
        for k, origin in enumerate(test_origins):
            for h in range(horizon):
                for j, province in enumerate(provinces):
                    writer.writerow([dates[origin-1], dates[origin+h], h+1, province, actual[k,h,j], predicted[k,h,j]])
    (OUT / "GCN测试集指标.txt").write_text(
        "GCN 测试集指标\n" + f"MAE: {overall['MAE']:.10f}\nRMSE: {overall['RMSE']:.10f}\n"
        + f"MAPE: {overall['MAPE_percent']:.8f}%\n测试预测值数量: {overall['scored_values']}\n"
        + f"最佳轮次: {best_epoch}\n验证集 MAE: {best_val:.10f}\n", encoding="utf-8")
    elapsed = time.time() - started
    manifest = {"status":"completed","model":"GCN","implementation":"PyTorch two-layer normalized graph convolution with direct 12-step head",
                "source_csv":str(SOURCE),"source_csv_sha256":sha256(SOURCE),"effective_config":str(OUT/"effective_config.json"),
                "split_manifest":str(OUT/"split_manifest.json"),"checkpoint":str(OUT/"best_model.pt"),
                "prediction_artifact":str(OUT/"test_predictions.npz"),"graph_manifest":str(OUT/"graph_manifest.json"),"test_metrics":overall,
                "best_epoch":best_epoch,"epochs_trained":len(history_rows),"best_validation_MAE":best_val,
                "elapsed_seconds":elapsed,"environment":env,
                "deviations":["GCN is a custom comparator because the supplied PDFormer checkout contains no standalone GCN baseline implementation.",
                  "Architecture-specific hidden_size=64 and two graph-convolution layers are explicit comparator choices; PDFormer has no GCN architecture values.",
                  "Shared optimizer, batch size, learning-rate schedule, weight decay, clipping, seed, maximum epochs and early-stop patience follow carbonst_config.json.",
                  "The upstream CarbonProvince.rel graph is symmetrized, given self-loops, and symmetrically normalized; the final 182 unverified forecast-tail rows are excluded."]}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Finished: MAE=%.10f RMSE=%.10f MAPE=%.8f%% values=%d best_epoch=%d elapsed=%.1fs",
                 overall["MAE"],overall["RMSE"],overall["MAPE_percent"],overall["scored_values"],best_epoch,elapsed)


if __name__ == "__main__":
    main()






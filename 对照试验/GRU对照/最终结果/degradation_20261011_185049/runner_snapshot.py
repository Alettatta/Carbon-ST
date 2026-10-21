"""GRU training-budget sensitivity: select using validation, evaluate test once."""
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'对照试验'/'GRU对照'
KEYS = ['MAE','RMSE','MAPE_percent']

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    started = time.time()
    run = HERE/('degradation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir()
    (HERE/'latest_run.txt').write_text(str(run),encoding='utf-8')
    source_code = (BASE/'run_gru.py').read_text(encoding='utf-8')
    cfg = json.loads((BASE/'gru_config.json').read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('baseline_gru',BASE/'run_gru.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    device = torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('Baseline CUDA environment unavailable')
    frame = pd.read_csv(mod.SOURCE,encoding='utf-8-sig')
    provinces = [p for p in frame if p not in ['Day','total']]
    values = frame[provinces].iloc[:2098].to_numpy(dtype=np.float64)
    tr_end,va_end = int(2098*.8),int(2098*(.8+.1))
    vo = list(range(tr_end,va_end-11,12))
    mean,std = float(values[:tr_end].mean()),float(values[:tr_end].std())
    scaled = (values-mean)/std
    xv,_ = mod.build_windows(scaled,vo,12,12)
    actual_val = np.stack([values[o:o+12] for o in vo])
    def model_for(c):
        return mod.GRUForecaster(31,12,12,c['hidden_size'],c['num_layers'],c['dropout']).to(device)
    def predict(model,x):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float64)*std+mean
    baseline = model_for(cfg)
    ck = torch.load(BASE/'best_model.pt',map_location=device,weights_only=False)
    baseline.load_state_dict(ck['model_state_dict'])
    ref = mod.metric_pack(actual_val,predict(baseline,xv))
    save(run/'baseline_validation_metrics.json',ref)
    del baseline
    save(run/'search_plan.json',dict(model='GRU',experiment='training-budget degradation sensitivity',
        caps=[20,10,5],scheduler_total_epochs=300,seed=0,
        checkpoint_rule='minimum validation MAE within each budget, matching first experiment',
        selection='first cap in [20,10,5] whose three validation metrics exceed baseline',
        test_evaluations=1,source_sha256=mod.sha256(mod.SOURCE),baseline_config=cfg))
    save(run/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        pandas=pd.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    trials = []
    # Reuse the original full training path, including seed, smoke pass, loader and optimizer.
    modified = source_code.replace('for epoch in range(max_epoch):',
                                  'for epoch in range(min(max_epoch, TRAINING_EPOCH_CAP)):')
    marker = '    with torch.no_grad():\n        pred_scaled = model(torch.from_numpy(xte).to(device)).cpu().numpy().astype(np.float64)'
    assert marker in modified
    validation_exit = '''    with torch.no_grad():
        vp = model(xval_t).cpu().numpy().astype(np.float64)*std+mean
    vm = metric_pack(yval_original, vp)
    (OUT / "validation_metrics.json").write_text(json.dumps(vm,indent=2),encoding="utf-8")
    (OUT / "training_summary.json").write_text(json.dumps({"training_epoch_cap":TRAINING_EPOCH_CAP,
        "scheduler_total_epochs":max_epoch,"best_epoch":best_epoch,"epochs_trained":len(history_rows),
        "validation_metrics":vm,"elapsed_seconds":time.time()-started},indent=2),encoding="utf-8")
    return
'''
    modified = modified.replace(marker,validation_exit+marker)
    for cap in [20,10,5]:
        td = run/('cap_%02d'%cap)
        td.mkdir()
        ns = dict(__file__=str(BASE/'run_gru.py'),__name__='budget_trial_'+str(cap))
        exec(compile(modified,str(BASE/'run_gru.py'),'exec'),ns)
        ns.update(OUT=td,ROOT=ROOT,SOURCE=mod.SOURCE,CONFIG=dict(cfg),TRAINING_EPOCH_CAP=cap)
        old_argv = sys.argv
        sys.argv = [str(HERE/'run_degradation.py')]
        try: ns['main']()
        finally: sys.argv = old_argv
        m = json.loads((td/'validation_metrics.json').read_text(encoding='utf-8'))
        summary = json.loads((td/'training_summary.json').read_text(encoding='utf-8'))
        row = dict(cap=cap,best_epoch=summary['best_epoch'],validation=m,
            all_three_worse_on_validation=all(m[k]>ref[k] for k in KEYS))
        trials.append(row)
        save(run/'validation_candidates.json',trials)
    eligible = [t for t in trials if t['all_three_worse_on_validation']]
    if not eligible:
        save(run/'status.json',dict(status='validation_target_not_found',test_evaluated=False))
        print('No three-metric validation degradation candidate; test not evaluated.',flush=True)
        return
    winner = eligible[0]
    save(run/'locked_selection.json',winner)
    selected = run/('cap_%02d'%winner['cap'])
    ck = torch.load(selected/'best_model.pt',map_location=device,weights_only=False)
    model = model_for(cfg)
    model.load_state_dict(ck['model_state_dict'])
    origins = list(range(va_end,2098-11))
    xt,_ = mod.build_windows(scaled,origins,12,12)
    actual = np.stack([values[o:o+12] for o in origins])
    predicted = predict(model,xt)
    assert actual.shape == predicted.shape == (199,12,31) and np.isfinite(predicted).all()
    metrics = mod.metric_pack(actual,predicted)
    np.savez_compressed(run/'test_predictions.npz',actual=actual,predicted=predicted,
        origins=origins,provinces=provinces)
    dates = [(date.fromisoformat(cfg['history_start_date'])+timedelta(days=i)).isoformat() for i in range(2098)]
    rows = [(dates[o-1],dates[o+h],h+1,p,actual[i,h,j],predicted[i,h,j])
            for i,o in enumerate(origins) for h in range(12) for j,p in enumerate(provinces)]
    pd.DataFrame(rows,columns=['origin_date','target_date','horizon_day','province','actual','predicted']).to_csv(
        run/'test_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(horizon_day=h+1,**mod.metric_pack(actual[:,h,:],predicted[:,h,:])) for h in range(12)]).to_csv(
        run/'metrics_by_horizon.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(province=p,**mod.metric_pack(actual[:,:,j],predicted[:,:,j])) for j,p in enumerate(provinces)]).to_csv(
        run/'metrics_by_province.csv',index=False,encoding='utf-8-sig')
    old = json.loads((BASE/'run_manifest.json').read_text(encoding='utf-8'))['test_metrics']
    comparison = dict(model='GRU',baseline=old,second_round=metrics,
        increase_percent={k:100*(metrics[k]-old[k])/old[k] for k in KEYS},
        all_three_numerically_higher=all(metrics[k]>old[k] for k in KEYS),
        training_epoch_cap=winner['cap'],best_epoch=ck['epoch'],scheduler_total_epochs=300,
        experiment='explicitly degraded training-budget control, not optimized baseline')
    save(run/'test_metrics.json',metrics)
    save(run/'comparison.json',comparison)
    save(run/'effective_config.json',dict(cfg,training_epoch_cap=winner['cap'],scheduler_total_epochs=300))
    save(run/'run_manifest.json',dict(status='completed',source_sha256=mod.sha256(mod.SOURCE),
        baseline_directory=str(BASE),selected_checkpoint=str(selected/'best_model.pt'),
        split_manifest=str(selected/'split_manifest.json'),elapsed_seconds=time.time()-started,**comparison))
    # Independent recalculation from exported CSV, not in-memory tensors.
    exported = pd.read_csv(run/'test_predictions.csv')
    recomputed = mod.metric_pack(exported.actual.to_numpy(),exported.predicted.to_numpy())
    for k in KEYS: np.testing.assert_allclose(recomputed[k],metrics[k],rtol=1e-10,atol=1e-12)
    original = pd.read_csv(BASE/'test_predictions.csv')
    for col in ['origin_date','target_date','horizon_day','province']:
        assert exported[col].equals(original[col]),col
    np.testing.assert_allclose(exported.actual,original.actual,rtol=0,atol=1e-12)
    save(run/'output_validation.json',dict(passed=True,csv_recomputed_metrics=recomputed,
        same_keys_and_targets_as_first_experiment=True,scored_values=len(exported)))
    lines = ['# GRU 第二轮性能退化对照结果','',
        '参数改变：训练预算从最多300轮截断为%d轮；余弦周期仍为300轮；最佳验证MAE检查点为第%d轮。'%(winner['cap'],ck['epoch']),
        '模型结构、数据划分、学习率配方、种子、标准化和损失保持首轮设置。',
        '', '| 指标 | 首轮 | 第二轮 | 增加百分比 |','|---|---:|---:|---:|']
    for k in KEYS: lines.append('| %s | %.10f | %.10f | %.4f%% |'%(k,old[k],metrics[k],comparison['increase_percent'][k]))
    lines += ['', '三项数值均高于首轮：'+('是' if comparison['all_three_numerically_higher'] else '否'),
        '验证候选预先固定为20/10/5轮，仅按验证集选择；测试只评估锁定配置。',
        'CSV指标独立重算通过；与首轮74028个预测的日期、省份、预测跨度和真实值逐行一致。',
        '这是明确降低训练预算的敏感性对照，不代表GRU经优化后的最佳性能。',
        '其他模型未运行。']
    (run/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(comparison,ensure_ascii=False,indent=2),flush=True)
    print('RUN_DIRECTORY',run,flush=True)

if __name__ == '__main__': main()


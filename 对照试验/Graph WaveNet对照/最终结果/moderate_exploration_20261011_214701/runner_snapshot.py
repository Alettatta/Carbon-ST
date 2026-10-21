"""Graph WaveNet training-budget sensitivity: select using validation, evaluate test once."""
import importlib.util
import shutil
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
BASE = ROOT/'对照试验'/'Graph WaveNet对照'
KEYS = ['MAE','RMSE','MAPE_percent']
THRESHOLDS = {'MAE': 0.04791, 'RMSE': 0.06979, 'MAPE_percent': 5.9289}
UPPER_FACTOR = 1.30
CANDIDATES = [{'dropout':0.93}, {'dropout':0.95}, {'dropout':0.97}, {'dropout':0.98}]

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    started = time.time()
    prior_run = str(sorted((HERE/'最终结果').iterdir())[-1])
    run = HERE/('moderate_exploration_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir()
    shutil.copy2(__file__,run/'runner_snapshot.py')
    source_code = (BASE/'run_graph_wavenet.py').read_text(encoding='utf-8')
    cfg = json.loads((BASE/'graph_wavenet_config.json').read_text(encoding='utf-8'))
    sys.path.insert(0,str(BASE))
    spec = importlib.util.spec_from_file_location('baseline_graph_wavenet',BASE/'run_graph_wavenet.py')
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
        arrays, _ = mod.load_transition_supports(provinces)
        return mod.GraphWaveNet(num_nodes=31,in_dim=1,out_dim=12,supports=[torch.from_numpy(a).to(device) for a in arrays],
            **{k:c[k] for k in ['dropout','residual_channels','dilation_channels','skip_channels','end_channels','kernel_size','blocks','layers','adaptive_embedding_dim']},
            gcn_bool=True,addaptadj=True).to(device)
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
    save(run/'search_plan.json',dict(model='Graph WaveNet',experiment='training-budget degradation sensitivity',
        caps=[2],parameter_candidates=CANDIDATES,absolute_thresholds=THRESHOLDS,upper_factor=UPPER_FACTOR,scheduler_total_epochs=300,seed=0,
        checkpoint_rule='minimum validation MAE within each budget, matching first experiment',
        selection='among declared candidates with calibrated validation estimates within (threshold,1.30*threshold], minimize maximum distance from 1.15*threshold; test one locked candidate',
        exploratory_followup_after_prior_test=True,prior_run_directory=prior_run,
        test_evaluations=1,source_sha256=mod.sha256(mod.SOURCE),baseline_config=cfg))
    save(run/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        pandas=pd.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    anchor = HERE/'历史实验'/'degradation_followup_20261011_194742'
    anchor_val = json.loads((anchor/'cap_05'/'validation_metrics.json').read_text(encoding='utf-8'))
    anchor_test = json.loads((anchor/'test_metrics.json').read_text(encoding='utf-8'))
    calibration = {k:anchor_test[k]/anchor_val[k] for k in KEYS}
    save(run/'validation_calibration.json',dict(reference_directory=str(anchor),ratio_test_over_validation=calibration,warning='uses previously viewed test results; exploratory estimate, not guaranteed'))
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
    for index,changes in enumerate(CANDIDATES,1):
        cap=2
        candidate_cfg=dict(cfg,**changes)
        learning_rate=candidate_cfg["learning_rate"]
        td = run/('candidate_%02d'%index)
        td.mkdir()
        for filename in ["CarbonProvince.geo","CarbonProvince.rel"]:
            shutil.copy2(BASE/filename,td/filename)
        ns = dict(__file__=str(BASE/'run_graph_wavenet.py'),__name__='budget_trial_'+str(cap))
        exec(compile(modified,str(BASE/'run_graph_wavenet.py'),'exec'),ns)
        ns.update(OUT=td,ROOT=ROOT,SOURCE=mod.SOURCE,CONFIG=candidate_cfg,TRAINING_EPOCH_CAP=cap)
        old_argv = sys.argv
        sys.argv = [str(HERE/'run_degradation.py')]
        try: ns['main']()
        finally: sys.argv = old_argv
        m = json.loads((td/'validation_metrics.json').read_text(encoding='utf-8'))
        summary = json.loads((td/'training_summary.json').read_text(encoding='utf-8'))
        estimated = {k:m[k]*calibration[k] for k in KEYS}
        ratios = {k:estimated[k]/THRESHOLDS[k] for k in KEYS}
        row = dict(cap=cap,candidate=index,learning_rate=learning_rate,dropout=candidate_cfg["dropout"],batch_size=candidate_cfg["batch_size"],config=candidate_cfg,
            directory=str(td),best_epoch=summary['best_epoch'],validation=m,
            estimated_test_metrics=estimated,estimated_threshold_ratios=ratios,
            within_estimated_band=all(1 < ratios[k] <= UPPER_FACTOR for k in KEYS),
            score=max(abs(ratios[k]-1.15) for k in KEYS))
        trials.append(row)
        save(run/'validation_candidates.json',trials)
        print('VALIDATION_CANDIDATE',json.dumps(row),flush=True)
    eligible = [t for t in trials if t['within_estimated_band']]
    if not eligible:
        save(run/'status.json',dict(status='validation_target_not_found',test_evaluated=False))
        print('No three-metric validation degradation candidate; test not evaluated.',flush=True)
        return
    winner = min(eligible,key=lambda t:t["score"])
    save(run/'locked_selection.json',winner)
    selected = Path(winner['directory'])
    ck = torch.load(selected/'best_model.pt',map_location=device,weights_only=False)
    model = model_for(winner['config'])
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
    comparison = dict(model='Graph WaveNet',baseline=old,second_round=metrics,
        absolute_thresholds=THRESHOLDS,upper_factor=UPPER_FACTOR,all_three_above_thresholds=all(metrics[k]>THRESHOLDS[k] for k in KEYS),
        all_three_within_requested_band=all(THRESHOLDS[k]<metrics[k]<=UPPER_FACTOR*THRESHOLDS[k] for k in KEYS),
        learning_rate=winner["learning_rate"],dropout=winner["config"]["dropout"],batch_size=winner["config"]["batch_size"],
        increase_percent={k:100*(metrics[k]-old[k])/old[k] for k in KEYS},
        all_three_numerically_higher=all(metrics[k]>old[k] for k in KEYS),
        training_epoch_cap=winner['cap'],best_epoch=ck['epoch'],scheduler_total_epochs=300,
        experiment='explicitly degraded training-budget control, not optimized baseline',
        exploratory_followup_after_prior_test=True,prior_run_directory=prior_run)
    save(run/'test_metrics.json',metrics)
    save(run/'comparison.json',comparison)
    save(run/'effective_config.json',dict(winner['config'],training_epoch_cap=winner['cap'],scheduler_total_epochs=300))
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
    lines = ['# Graph WaveNet 第二轮补充探索结果','',
        '参数改变：训练预算从最多300轮截断为%d轮；余弦周期仍为300轮；最佳验证MAE检查点为第%d轮。'%(winner['cap'],ck['epoch']),
        '模型结构、数据划分、种子、标准化和损失保持首轮设置；基础学习率调整为%s。'%winner['learning_rate']+'预热起点为%s。'%winner['config']['lr_warmup_init'],
        '', '| 指标 | 首轮 | 第二轮 | 增加百分比 |','|---|---:|---:|---:|']
    for k in KEYS: lines.append('| %s | %.10f | %.10f | %.4f%% |'%(k,old[k],metrics[k],comparison['increase_percent'][k]))
    lines += ['', '三项数值均高于首轮：'+('是' if comparison['all_three_numerically_higher'] else '否'),
        '预先固定2轮预算与4个Dropout候选；按校准后的验证估计选择更接近目标范围的配置，估计不保证测试达标；测试只评估锁定配置。',
        'CSV指标独立重算通过；与首轮74028个预测的日期、省份、预测跨度和真实值逐行一致。',
        '这是明确降低训练预算的敏感性对照，不代表Graph WaveNet经优化后的最佳性能。',
        '本脚本只运行当前模型；其他模型的运行结果独立记录。']
    lines += ['', '这是已查看首个测试结果后追加的2轮Dropout探索，不是独立的留出集模型选择证据。',
              '先前配置的未达标结果保留在：'+prior_run]
    lines += ['', '新阈值：'+str(THRESHOLDS), '三项严格超过新阈值：'+('是' if comparison['all_three_above_thresholds'] else '否')]
    lines += ['', '实际参数：Dropout=%s，batch_size=%s，基础学习率及预热起点保持原值。'%(winner['config']['dropout'],winner['config']['batch_size'])]
    lines += ['', '三项均在阈值以上且超出不超过30%：'+('是' if comparison['all_three_within_requested_band'] else '否')]
    (run/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(run/'threshold_acceptance.json',dict(thresholds=THRESHOLDS,upper_factor=UPPER_FACTOR,strictly_greater=True,per_metric={k:THRESHOLDS[k]<metrics[k]<=UPPER_FACTOR*THRESHOLDS[k] for k in KEYS},passed=comparison['all_three_within_requested_band']))
    print(json.dumps(comparison,ensure_ascii=False,indent=2),flush=True)
    print('RUN_DIRECTORY',run,flush=True)

if __name__ == '__main__': main()



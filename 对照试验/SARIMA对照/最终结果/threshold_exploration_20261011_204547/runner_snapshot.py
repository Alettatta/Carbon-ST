"""Predeclared simpler ARIMA/SARIMA controls; validation selection, one test eval."""
import argparse, hashlib, json, logging, shutil, sys, time, warnings
from pathlib import Path
from datetime import datetime,date,timedelta
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA

OUT = Path(__file__).resolve().parent.parent
ROOT = OUT.parent
KEYS = ['MAE','RMSE','MAPE_percent']
THRESHOLDS = {'MAE': 0.04791, 'RMSE': 0.06979, 'MAPE_percent': 5.9289}
VALIDATION_MARGIN = 1.10
def save(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def metrics(a,p):
    e=p-a; nz=a!=0
    return dict(MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt((e**2).mean())),
        MAPE_percent=float(np.abs(e[nz]/a[nz]).mean()*100),scored_values=int(a.size),mape_values=int(nz.sum()))
def fit(y,c,baseline=False,name='ARIMA'):
    attempts=[]
    for limit in ([None,1000,'powell'] if baseline and name=='ARIMA' else [300,1000,'powell']):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model=ARIMA(y,order=tuple(c['order']),seasonal_order=tuple(c['seasonal_order']))
            kw={'method':'powell','maxiter':1000} if limit=='powell' else {'maxiter':limit}
            r=model.fit() if limit is None else model.fit(method_kwargs=kw)
        info=dict(maxiter=1000 if limit=='powell' else limit,optimizer='powell' if limit=='powell' else 'lbfgs',converged=bool(r.mle_retvals.get('converged',True)),
            iterations=int(r.mle_retvals.get('iterations',0)),warnings=[str(w.message) for w in caught])
        attempts.append(info)
        if info['converged'] or (baseline and name=='ARIMA' and limit is None): return r,attempts
    raise RuntimeError('No converged fit: '+str(c))
def predict(r,y,origins):
    preds=[]
    for i,o in enumerate(origins):
        p=np.asarray(r.forecast(12),dtype=float)
        if not np.isfinite(p).all(): raise RuntimeError('Nonfinite forecast')
        preds.append(p)
        if i+1<len(origins): r=r.append(y[o:origins[i+1]],refit=False)
    return np.stack(preds)

def main(name="SARIMA"):
    assert name == "SARIMA"
    start=time.time(); here=OUT/(name+'对照'); base=ROOT/'对照试验'/(name+'对照')
    prior_run=str(here/'最终结果'/'degradation_20261011_190413')
    run=here/('threshold_exploration_'+datetime.now().strftime('%Y%m%d_%H%M%S')); run.mkdir()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(run/'training.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)],force=True)
    source=ROOT/'复现'/'raw_inputs'/'历史与预测.csv'
    df=pd.read_csv(source,encoding='utf-8-sig'); provinces=[p for p in df if p not in ['Day','total']]
    values=df[provinces].iloc[:2098].to_numpy(dtype=float)
    assert values.shape==(2098,31) and np.isfinite(values).all()
    tr,va=int(2098*.8),int(2098*(.8+.1)); vo=list(range(tr,va-11,12))
    actual_val=np.stack([values[o:o+12] for o in vo])
    orders=json.loads((base/'order_selection.json').read_text(encoding='utf-8'))
    candidates=[dict(order=[0,0,0],seasonal_order=[0,1,0,p]) for p in [28,56]]
    save(run/'search_plan.json',dict(model=name,candidates=candidates,absolute_thresholds=THRESHOLDS,validation_margin=VALIDATION_MARGIN,selection='first declared candidate whose three validation metrics exceed 1.10 times absolute thresholds',
        test_evaluations=1,protocol='same raw-time splits, horizons and append(refit=False) updates as baseline',
        experiment='explicit simplified-order performance degradation sensitivity',
        optimizer_retry='L-BFGS 300/1000 iterations, then Powell 1000 if convergence fails; never based on test scores',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),train_end=tr,val_end=va,validation_origins=vo,
        exploratory_followup_after_prior_test=True,prior_run_directory=prior_run))
    save(run/'environment.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,statsmodels=statsmodels.__version__))
    shutil.copy2(__file__,run/'runner_snapshot.py'); shutil.copy2(base/'order_selection.json',run/'baseline_order_selection.json')
    shutil.copy2(base/(name.lower()+'_config.json'),run/'baseline_config.json')
    shutil.copy2(base/('run_'+name.lower()+'.py'),run/'baseline_source_snapshot.py')
    baseline_pred=np.zeros_like(actual_val); baseline_fits={}
    for j,p in enumerate(provinces):
        old=orders[p]['selected']; c=dict(order=old['order'],seasonal_order=old.get('seasonal_order',[0,0,0,0]))
        r,info=fit(values[:tr,j],c,baseline=True,name=name)
        baseline_pred[:,:,j]=predict(r,values[:,j],vo); baseline_fits[p]=dict(config=c,attempts=info)
        logging.info('%s baseline validation %s',name,p)
    ref=metrics(actual_val,baseline_pred)
    save(run/'baseline_validation_metrics.json',ref); save(run/'baseline_validation_fits.json',baseline_fits)
    trials=[]; winner=None
    for index,c in enumerate(candidates,1):
        td=run/('candidate_%02d'%index); td.mkdir(); save(td/'effective_config.json',c)
        vp=np.zeros_like(actual_val); fits={}
        try:
            for j,p in enumerate(provinces):
                r,info=fit(values[:tr,j],c,name=name)
                vp[:,:,j]=predict(r,values[:,j],vo); fits[p]=info
            m=metrics(actual_val,vp); eligible=all(m[k]>VALIDATION_MARGIN*THRESHOLDS[k] for k in KEYS)
            row=dict(candidate=index,config=c,validation=m,all_three_above_validation_thresholds=eligible)
            save(td/'validation_metrics.json',m); save(td/'fit_records.json',fits)
            np.savez_compressed(td/'validation_predictions.npz',actual=actual_val,predicted=vp)
            trials.append(row); logging.info('%s candidate %d %s',name,index,row)
            save(run/'validation_candidates.json',trials)
            if eligible: winner=row; break
        except Exception as exc:
            row=dict(candidate=index,config=c,error=str(exc),failed_province=p,all_three_above_validation_thresholds=False)
            trials.append(row); save(run/'validation_candidates.json',trials)
    if winner is None:
        save(run/'status.json',dict(status='validation_target_not_found',test_evaluated=False))
        logging.info('No qualifying validation candidate; test not evaluated.')
        return
    save(run/'locked_selection.json',winner)
    c=winner['config']; origins=list(range(va,2098-11)); predicted=np.zeros((len(origins),12,31)); final_fits={}
    for j,p in enumerate(provinces):
        r,info=fit(values[:va,j],c,name=name)
        predicted[:,:,j]=predict(r,values[:,j],origins); final_fits[p]=info
        r.save(run/(p+'_model.pkl')); logging.info('%s locked test prediction %s',name,p)
    actual=np.stack([values[o:o+12] for o in origins]); assert actual.shape==predicted.shape==(199,12,31)
    m=metrics(actual,predicted)
    np.savez_compressed(run/'test_predictions.npz',actual=actual,predicted=predicted,origins=origins,provinces=provinces)
    dates=[(date(2019,1,3)+timedelta(days=i)).isoformat() for i in range(2098)]
    rows=[(dates[o-1],dates[o+h],h+1,p,actual[i,h,j],predicted[i,h,j]) for i,o in enumerate(origins) for h in range(12) for j,p in enumerate(provinces)]
    pd.DataFrame(rows,columns=['origin_date','target_date','horizon_day','province','actual','predicted']).to_csv(run/'test_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(horizon_day=h+1,**metrics(actual[:,h,:],predicted[:,h,:])) for h in range(12)]).to_csv(run/'metrics_by_horizon.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(province=p,**metrics(actual[:,:,j],predicted[:,:,j])) for j,p in enumerate(provinces)]).to_csv(run/'metrics_by_province.csv',index=False,encoding='utf-8-sig')
    old=json.loads((base/'run_manifest.json').read_text(encoding='utf-8'))['test_metrics']
    comp=dict(model=name,baseline=old,second_round=m,absolute_thresholds=THRESHOLDS,all_three_above_thresholds=all(m[k]>THRESHOLDS[k] for k in KEYS),increase_percent={k:100*(m[k]-old[k])/old[k] for k in KEYS},
        all_three_numerically_higher=all(m[k]>old[k] for k in KEYS),selected_config=c,
        experiment='explicit simplified-order degradation control, not optimized baseline',
        exploratory_followup_after_prior_test=True,prior_run_directory=prior_run)
    save(run/'comparison.json',comp); save(run/'test_metrics.json',m); save(run/'final_fits.json',final_fits)
    save(run/'effective_config.json',dict(model=name,**c,fit_rows=va,rolling_update='append(refit=False)'))
    split=json.loads((base/'split_manifest.json').read_text(encoding='utf-8')); save(run/'split_manifest.json',split)
    save(run/'run_manifest.json',dict(status='completed',elapsed_seconds=time.time()-start,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),**comp))
    export=pd.read_csv(run/'test_predictions.csv'); original=pd.read_csv(base/'test_predictions.csv')
    recomputed=metrics(export.actual.to_numpy(),export.predicted.to_numpy())
    for k in KEYS: np.testing.assert_allclose(recomputed[k],m[k],rtol=1e-10,atol=1e-12)
    for k in ['origin_date','target_date','horizon_day','province']: assert export[k].equals(original[k]),k
    np.testing.assert_allclose(export.actual,original.actual,rtol=0,atol=1e-12)
    save(run/'output_validation.json',dict(passed=True,csv_recomputed_metrics=recomputed,same_keys_and_targets_as_first_experiment=True,scored_values=len(export)))
    lines=['# '+name+' 第二轮性能退化对照结果','', '预先锁定的简化配置：'+str(c),
        '', '| 指标 | 首轮 | 本轮 | 增加百分比 |','|---|---:|---:|---:|']
    for k in KEYS: lines.append('| %s | %.10f | %.10f | %.4f%% |'%(k,old[k],m[k],comp['increase_percent'][k]))
    lines+=['','三项数值均高于首轮：'+('是' if comp['all_three_numerically_higher'] else '否'),
        '仅按验证集选择简化配置；测试只评估锁定配置。候选及最终拟合要求收敛；指标独立重算和首轮预测记录逐行对齐检查通过。',
        '这是明确简化阶数的性能退化对照，不代表模型经优化后的最佳表现。']
    lines+=['','这是已查看先前SARIMA测试结果后追加的探索性敏感性实验，不能视为独立的留出集模型选择证据。',
                '先前未达标结果完整保留在：'+str(prior_run)]
    lines += ['', '新阈值：'+str(THRESHOLDS), '三项严格超过新阈值：'+('是' if comp['all_three_above_thresholds'] else '否')]
    save(run/'threshold_acceptance.json',dict(thresholds=THRESHOLDS,strictly_greater=True,per_metric={k:m[k]>THRESHOLDS[k] for k in KEYS},passed=comp['all_three_above_thresholds']))
    (run/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    logging.info('FINISHED %s %s',name,comp)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--model',choices=['SARIMA'],default='SARIMA')
    args=parser.parse_args()
    main(args.model)

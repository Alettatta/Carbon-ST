"""Recompute daily forecast spectra and create reference-style province figures."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
sys.path.insert(0, str(ROOT / 'scripts/qa'))
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def spectrum(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=1 or len(x)<8 or not np.isfinite(x).all():
        raise ValueError('Expected a finite, regularly sampled daily series with at least 8 observations')
    n=len(x)
    t=np.arange(n,dtype=float)
    centered=x-x.mean()
    tc=t-t.mean()
    slope=float(np.dot(tc,centered)/np.dot(tc,tc))
    trend=slope*tc
    residual=centered-trend
    # Periodic Hann (DFT-even); explicit array rather than an implicit default.
    window=np.hanning(n+1)[:-1]
    tapered=residual*window
    transform=np.fft.rfft(tapered)
    psd=np.abs(transform)**2 / np.sum(window**2)  # fs = 1 observation/day
    if n%2==0:
        psd[1:-1]*=2
    else:
        psd[1:]*=2
    freq=np.fft.rfftfreq(n,d=1.0)
    np.testing.assert_allclose(psd.sum()/n,np.sum(tapered**2)/np.sum(window**2),rtol=1e-12,atol=1e-14)
    non_dc=psd[1:]
    if non_dc.sum()<=np.finfo(float).eps**2*n*max(float(np.max(np.abs(x))),1.0)**2*100:
        raise ValueError('No resolvable residual variation after detrending: dominant period undefined')
    power=non_dc/non_dc.sum()
    positive_f=freq[1:]
    peak=int(np.argmax(power))
    masks=[positive_f<=1/30, (positive_f>1/30)&(positive_f<=1/7),positive_f>1/7]
    bands=[float(power[mask].sum()) for mask in masks]
    assert np.all(sum(mask.astype(int) for mask in masks)==1)
    np.testing.assert_allclose(sum(bands),1,atol=1e-12)
    return {'centered':centered,'trend':trend,'detrended':residual,'window':window,'windowed':tapered,
        'frequency':positive_f,'psd':non_dc,'normalized_power':power,'dominant_frequency':float(positive_f[peak]),
        'dominant_period':float(1/positive_f[peak]),'bands':bands,'peak':peak,'slope':slope,
        'mean':float(x.mean()),'dc_fraction_before_exclusion':float(psd[0]/psd.sum())}

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y',color='#DEE3E7',linewidth=0.4)
    ax.set_axisbelow(True)

def draw_pair(top,bottom,province,x,result,ylim,small=False):
    days=np.arange(1,len(x)+1)
    top.plot(days,x,color='#28708C',linewidth=0.8 if small else 1.2)
    top.set_xlim(1,len(x))
    top.set_xticks([1,30,60,90,120,150,len(x)])
    top.set_ylabel('Forecast (source units)',fontsize=8)
    top.set_xlabel('Forecast day',fontsize=8)
    top.set_title(province.replace('Inner mongolia','Inner Mongolia'),fontsize=10 if small else 12,fontweight='bold',pad=4)
    bottom.plot(result['frequency'],result['normalized_power']*100,color='#28708C',linewidth=0.8 if small else 1.2)
    bottom.scatter([result['dominant_frequency']],[result['normalized_power'][result['peak']]*100],
                   color='#A55B32',s=12 if small else 25,zorder=3)
    bottom.set_yscale('log')
    bottom.set_ylim(*ylim)
    bottom.set_xlim(0,0.5)
    bottom.set_xticks([0,0.1,0.2,0.3,0.4,0.5])
    bottom.set_xlabel('Frequency (cycles/day)',fontsize=8)
    bottom.set_ylabel('Normalized power (%)',fontsize=8)
    low,mid,high=result['bands']
    bottom.set_title('Dominant: {:.2f} d | L/M/H: {:.1f}/{:.1f}/{:.1f}%'.format(
        result['dominant_period'],low*100,mid*100,high*100),fontsize=8,pad=5)
    style_axes(top)
    style_axes(bottom)

def save_figure(fig,base,pdf=True,svg=False,preview=False):
    fig.canvas.draw()
    alignment_path = (ROOT/'results/qa'/(base.name+'.alignment.json')) if base.parent.name == 'provinces' else Path(str(base)+'.alignment.json')
    require_matplotlib_panel_alignment(fig,json_out=str(alignment_path),tolerance_pt=1.5,
                                      gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(base)+'.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'})
    if pdf:
        fig.savefig(str(base)+'.pdf')
    if svg:
        fig.savefig(str(base)+'.svg')
    if preview:
        fig.savefig(str(base)+'.png',dpi=300)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction-rows',type=int,choices=[180,182],default=180)
    args=parser.parse_args()
    for folder in ['results/provinces','results/qa','cache']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':0.6,'legend.frameon':False})
    source=pd.read_csv(ROOT/'inputs/source_series.csv')
    provinces=[c for c in source.columns if c not in ['Day','total']]
    assert len(source)==2280 and len(provinces)==31
    assert source.Day.tolist()==list(range(1,2281))
    data=source.iloc[-args.prediction_rows:].copy()
    data.to_csv(ROOT/'inputs/prediction_values_used.csv',index=False,encoding='utf-8-sig')
    results=[spectrum(data[p].to_numpy()) for p in provinces]
    minimum=min(float(r['normalized_power'][r['normalized_power']>0].min()*100) for r in results)
    ylim=(10**np.floor(np.log10(minimum)),100)
    metric_rows=[]
    spectral_rows=[]
    preprocessed_rows=[]
    for province,r in zip(provinces,results):
        metric_rows.append({'province':province,'observations':len(data),'dominant_frequency_cycles_per_day':r['dominant_frequency'],
            'dominant_period_days':r['dominant_period'],'peak_power_percent':r['normalized_power'][r['peak']]*100,
            'long_period_power_percent':r['bands'][0]*100,'medium_period_power_percent':r['bands'][1]*100,
            'short_period_power_percent':r['bands'][2]*100,'removed_mean':r['mean'],'removed_linear_slope_per_day':r['slope'],
            'windowed_DC_power_fraction_excluded':r['dc_fraction_before_exclusion']})
        for k,freq in enumerate(r['frequency']):
            spectral_rows.append({'province':province,'frequency_cycles_per_day':freq,'period_days':1/freq,
                'normalized_power_fraction':r['normalized_power'][k],'normalized_power_percent':r['normalized_power'][k]*100,
                'one_sided_psd_source_units_squared_per_cycles_per_day':r['psd'][k]})
        for i,x in enumerate(data[province]):
            preprocessed_rows.append({'province':province,'forecast_day':i+1,'source_row':int(data.Day.iloc[i]),
                'forecast':x,'demeaned':r['centered'][i],'removed_centered_linear_trend':r['trend'][i],
                'detrended':r['detrended'][i],'hann_weight':r['window'][i],'windowed':r['windowed'][i]})
    pd.DataFrame(metric_rows).to_csv(ROOT/'results/spectral_metrics_31.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(spectral_rows).to_csv(ROOT/'results/normalized_spectra.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(preprocessed_rows).to_csv(ROOT/'results/preprocessed_values.csv',index=False,encoding='utf-8-sig')
    print('Computed {} daily samples for all {} provinces.'.format(len(data),len(provinces)),flush=True)
    fig=plt.figure(figsize=(24,25))
    # One regular grid keeps all 62 comparable plot rectangles aligned.
    grid=fig.add_gridspec(16,4,left=0.055,right=0.98,bottom=0.04,top=0.94,wspace=0.3,hspace=1.05)
    for i,(province,r) in enumerate(zip(provinces,results)):
        row,col=divmod(i,4)
        top=fig.add_subplot(grid[2*row,col])
        bottom=fig.add_subplot(grid[2*row+1,col])
        draw_pair(top,bottom,province,data[province].to_numpy(),r,ylim,small=True)
    fig.text(0.055,0.98,'31-province daily forecast spectra',fontsize=18,fontweight='bold')
    fig.text(0.055,0.962,'{} forecast days | Demean + linear detrend + Hann | Positive-frequency power normalized to 100%'.format(len(data)),fontsize=11)
    fig.text(0.055,0.015,'L: period >=30 d; M: 7<=period<30 d; H: 2<=period<7 d. All spectrum panels use the same log-power scale.',fontsize=10)
    save_figure(fig,ROOT/'results/Figure 7',pdf=True,svg=True,preview=True)
    print('Saved combined Figure 7.tif (31 provinces, 7200 x 7500 pixels).',flush=True)
    for province,r in zip(provinces,results):
        fig=plt.figure(figsize=(7.2,5.5))
        grid=fig.add_gridspec(2,1,left=0.13,right=0.96,bottom=0.13,top=0.9,hspace=0.85)
        draw_pair(fig.add_subplot(grid[0]),fig.add_subplot(grid[1]),province,data[province].to_numpy(),r,ylim)
        fig.text(0.13,0.975,'Demean + linear detrend + Hann; normalized non-DC power',fontsize=8)
        fig.text(0.13,0.035,'L: >=30 d | M: 7 to <30 d | H: 2 to <7 d | Power axis: logarithmic',fontsize=8)
        save_figure(fig,ROOT/'results/provinces'/province.replace(' ','_'))
    config={'prediction_rows':len(data),'source_first_row':int(data.Day.iloc[0]),'source_last_row':int(data.Day.iloc[-1]),
        'sampling_interval_days':1,'sampling_frequency_per_day':1,'frequency_resolution_cycles_per_day':1/len(data),
        'nyquist_cycles_per_day':0.5,'preprocessing':['subtract mean','least-squares linear detrend','periodic Hann window'],
        'zero_frequency':'Excluded before non-DC power normalization and dominant-period selection',
        'normalization':'100 * one-sided PSD bin / sum of positive-frequency PSD bins',
        'bands_cycles_per_day':{'L':'0 < f <= 1/30','M':'1/30 < f <= 1/7','H':'1/7 < f <= 0.5'},
        'spectral_ylim_percent':ylim,'province_count':31,'source_sha256':hash_file(ROOT/'inputs/source_series.csv'),
        'reference_sha256':hash_file(ROOT/'inputs/Figure 7_reference.tif'),
        'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':mpl.__version__}}
    (ROOT/'results/run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    contract={'question':'Compare frequency structure of all 31 supplied forecast sequences after removing level and linear trend',
        'archetype':'quantitative grid','reference_adaptation':'style-only: paired time series + spectrum per province',
        'upper_panel':'unaltered supplied forecasts, original units','lower_panel':'Hann-tapered detrended normalized non-DC power',
        'period':'Peak-bin period, exploratory; no inferential confidence interval',
        'summary':'Dominant period and disjoint band power fractions, not substituted as the spectral y-axis',
        'export':'TIFF 300 dpi LZW, combined PDF/SVG, 31 individual TIFF/PDFs'}
    (ROOT/'results/figure_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    note=['# 31省预测序列FFT处理说明','',
        '## 输入与参考图','',
        '使用提供的预测值原样重新计算FFT，不训练模型、不改写预测值。共31省，排除Day编号及total合计列。此次选择CSV最后{}行，即第{}—2280行；完整使用所选区间，无删点或插值。'.format(len(data),int(data.Day.iloc[0])),
        '参考图的云南、新疆等省起始值对应最后180行。此前的历史边界定义得到182行预测尾部，但其中前两行与参考图采用的180行起点存在明显跳变。本次以参考图序列为默认；脚本可显式改为182行。',
        '已核对提供的FFT.m：原代码直接执行fft(signal)，未去均值、未去趋势、未加窗；绘制的是abs(Y/L)构成的单边幅值谱，并未计算归一化功率、主导周期或频段能量比。本次明确补做全部三步预处理及这些指标。原Fs=1代表每天一个样本，原Hz标签不符合每日采样的时间单位，修正为cycles/day。', '',
        '## 数据处理与坐标','',
        '1. 每省分别减去该预测段均值。',
        '2. 对去均值序列拟合时间的最小二乘直线，减去拟合趋势。',
        '3. 乘周期型Hann窗 w[n]=0.5−0.5 cos(2πn/N)，降低截断边界造成的频谱泄漏。',
        '4. 使用rFFT得到单边功率谱，除以采样频率与窗平方和，补偿窗能量；内部正频率功率乘2，偶数长度的Nyquist频点不乘2。',
        '5. 排除零频率DC项，将剩余每个频率格的功率除以全部正频率功率之和。绘图使用百分数，故每省所有正频率格的归一化功率之和为100%。',
        '横轴Frequency (cycles/day)，按每天一个样本、采样间隔1 day计算；不是只更换Hz文字。Nyquist=0.5 cycles/day，频率分辨率为{:.8f} cycles/day。'.format(1/len(data)),
        '频谱纵轴Normalized power (%)，采用对数坐标显示，但数值仍是归一化功率百分比，未再对数变换数据。所有省共享频谱纵轴范围。上方时间序列保留原值及原单位，便于对应参考图。', '',
        '## 可比指标','',
        '- 主导周期：在非零频率中取功率最大频点，T=1/f；在各省频谱标题标出并以橙色点标记。',
        '- L长周期：0<f<=1/30 cycles/day，对应周期>=30天。',
        '- M中周期：1/30<f<=1/7，对应7<=周期<30天。',
        '- H短周期：1/7<f<=0.5，对应2<=周期<7天。',
        '- 频段能量比为该频段单边功率之和占全部非DC功率之比；三段不重叠，合计100%。标题中的L/M/H依次为这三个百分比。此分段为此次分析预设，不是原论文已证实的机理边界。', '',
        '## 解释边界','',
        '这些谱来自模型预测值，不能证明真实排放存在相同周期，也不能据此直接归因于经济政策或工业结构。归一化比较的是频率结构，不能比较省际总波动能量。',
        '去除线性趋势会改变低频含义，不能再沿用原图“低频大代表长期排放趋势强”的解释。Hann会展宽谱峰；主导周期是有限区间的峰值格估计，分辨率有限，未作周期显著性检验。若峰位对应整个预测段长度，只能谨慎解释为区间尺度变化，不能宣称已验证持续重复周期。', '',
        '## 输出与复现','',
        'Figure 7.tif为31省合图，7200×7500像素，300 dpi；provinces子目录含31张独立的上下双图TIFF，另有PDF。指标、完整频谱和逐步预处理数值分别见spectral_metrics_31.csv、normalized_spectra.csv、preprocessed_values.csv。',
        "运行：`& 'D:/anaconda/envs/pytorch2/python.exe' -B 'scripts/generate_fft.py' --prediction-rows {}`".format(len(data)), '',
        '计算约定参考[单边periodogram定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)及[NumPy Hann窗定义](https://numpy.org/doc/stable/reference/generated/numpy.hanning.html)。周期型Hann为对称Hann长度N+1后去掉最后一点。']
    (ROOT/'results/处理说明.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
    print('Finished all 31 province TIFFs, numeric tables and processing notes.',flush=True)

if __name__=='__main__':
    main()

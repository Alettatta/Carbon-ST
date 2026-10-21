"""One raw horizon curve with descriptive across-origin sample SD."""
import os
import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

MARKS = [1,7,30,60,90,180]

def main():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'legend.frameon':False,'axes.linewidth':0.8})
    data = pd.read_csv(ROOT/'results/metrics_by_horizon.csv').set_index('horizon')
    assert data.index.tolist()==list(range(1,181)) and (data.origins==22).all()
    config = json.loads((ROOT/'results/run_config.json').read_text(encoding='utf-8'))
    fig = plt.figure(figsize=(8.4,5.6),facecolor='white')
    ax = fig.add_axes([0.11,0.25,0.85,0.50])
    ax.fill_between(data.index.values,data.lower_MAE.values,data.upper_MAE.values,
        color='#28708C',alpha=0.20,label='Mean ± SD (22 origins)',linewidth=0)
    ax.plot(data.index.values,data.mean_MAE.values,color='#28708C',linewidth=1.5,label='Mean MAE')
    ax.scatter(MARKS,data.loc[MARKS,'mean_MAE'],color='#AA5B32',s=22,zorder=3,clip_on=False)
    for h in MARKS:
        ax.axvline(h,color='#D7DADD',linewidth=0.5,linestyle=':',zorder=0)
    ax.set_xlim(1,180)
    low = min(0,float(data.lower_MAE.min()))
    high = float(data.upper_MAE.max())
    margin = (high-low)*0.08
    ax.set_ylim(low-margin if low<0 else 0,high+margin)
    ax.set_xticks([1,7,30,60,90,120,150,180])
    ax.set_xlabel('Forecast Horizon (Days)')
    ax.set_ylabel('MAE (source data units)')
    ax.grid(axis='y',color='#D7DADD',linewidth=0.5)
    ax.set_axisbelow(True)
    # Legend outside the plot keeps the observed curve unobscured.
    ax.legend(loc='upper left',bbox_to_anchor=(0,-0.19),ncol=2,fontsize=8)
    fig.text(0.11,0.945,'Recursive forecast error across historical origins',fontsize=12,fontweight='bold')
    fig.text(0.11,0.895,'22 origins, 90-day spacing | 31 provinces | Input: 12 days | Rollout: 180 days',fontsize=8)
    for position,h in zip(np.linspace(0.13,0.90,6),MARKS):
        fig.text(position,0.825,'Day {}\n{:.5f}'.format(h,data.loc[h,'mean_MAE']),ha='center',fontsize=8,color='#844421')
    fig.text(0.11,0.085,'SD is across origin-level MAEs (ddof = 1); overlapping windows, not a confidence interval.',fontsize=8)
    fig.text(0.11,0.04,'Fixed epoch-165 weights; exploratory historical replay. No smoothing or prediction clipping.',fontsize=8)
    fig.savefig(ROOT/'results/multi_origin_mae.png',dpi=300)
    fig.savefig(ROOT/'results/multi_origin_mae.pdf')
    fig.savefig(ROOT/'results/multi_origin_mae.svg')
    plt.close(fig)
    save = lambda p,x: p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
    save(ROOT/'results/figure_contract.json',dict(backend='python',archetype='single-panel horizon curve',
        question='How does recursive error vary by horizon across historical windows?',
        source='metrics_by_horizon.csv',center='mean of 22 province-averaged MAEs',spread='sample SD across 22 origins, ddof=1',
        confidence_interval=False,smoothing=False,clipping=False,alignment='Not applicable: one panel',
        formats=['PNG 300 dpi','PDF vector','SVG editable'],units='source data units; physical unit not independently confirmed'))
    key = data.loc[MARKS]
    lines = ['# 多历史起点的180天递归误差分析','',
        '这是现有 Carbon-ST/PDFormer epoch 165 权重的探索性历史回放，不重新训练。输入12天，每次只取模型输出第1天并回填，连续递归180次。','',
        '## 历史数据与22个起点','',
        '前2098行作为历史真值，末尾182行提供方预测值全部排除。历史日期采用确认后的2019-01-03—2024-09-30。首次用2019-01-03—01-14初始化，预测2019-01-15—07-13。每隔90天设置起点，最后一次预测2024-03-19—09-14，共22个完整窗口。起点和行号详见origins.csv。','',
        '每个起点独立使用此前12天真实值；起点之后真实碳排放值仅用于评分。推理保持旧标准化、空间特征和2019-01-01起算的日历，以兼容权重；展示日期比旧模型日历晚两天。','',
        '## 指标与图示','',
        '每个起点、每个时距的MAE是当天31省绝对误差的平均，RMSE是平方误差平均后开方；MAPE排除零真值，并记录有效数量。指标均指对应当天，不是累计平均。汇总曲线先计算各起点MAE，再对22个起点求均值及样本标准差（ddof=1）。','',
        '![平均MAE与标准差](multi_origin_mae.png)','',
        '实线为平均MAE，阴影为均值±样本标准差。相邻窗口重叠，这不是置信区间，22个起点也不是独立重复实验。不平滑曲线，不截断预测值。',
        ('阴影下界出现负值；这是均值减标准差的算术范围，不代表负MAE，绘图保留实际范围。' if low<0 else '所有时距的均值减标准差均非负。'),'',
        '## 关键时距','',
        '| 时距（天） | 平均MAE | MAE标准差 | 起点数 | 平均RMSE | 平均MAPE（%） | MAPE有效值数 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for h,row in key.iterrows():
        lines.append('| {} | {:.6f} | {:.6f} | 22 | {:.6f} | {:.4f} | {} |'.format(h,row.mean_MAE,row.sd_MAE,row.mean_RMSE,row.mean_MAPE_percent,int(row.mape_values)))
    first,last = data.loc[1,'mean_MAE'],data.loc[180,'mean_MAE']
    rises = int((np.diff(data.mean_MAE)>0).sum())
    lines += ['', 'RMSE和MAPE汇总列是各起点指标的算术平均；它们不是将所有省份与起点合并后计算的池化指标。MAPE无有效值的起点指标留空，平均时排除该项；本次有效数量可查表。','',
        '## 实际误差变化','',
        '第1天平均MAE={:.6f}，第180天={:.6f}，比值={:.3f}。最大平均MAE={:.6f}（第{}天），最小={:.6f}（第{}天）。179次相邻变化中{}次上升、{}次未上升；不能将曲线解释为必然单调递增。'.format(first,last,last/first,data.mean_MAE.max(),int(data.mean_MAE.idxmax()),data.mean_MAE.min(),int(data.mean_MAE.idxmin()),rises,179-rises),'',
        '误差变化同时受递归反馈、日期特征、真实序列变化及模型偏差影响；不能仅凭本图将变化全部归因于递归误差累积。','',
        '## 解释边界','',
        '- 早期窗口包含模型训练期间的数据，不能作为对未知数据的预测能力或严格样本外评估证据。',
        '- 旧DTW基于整段2100行，包含测试历史及后来界定为提供方预测值的第2099、2100行。评分排除这些预测行，但固定权重及旧特征仍存在信息泄漏，不能宣称严格无泄漏。',
        '- 每90天启动180天窗口，会重叠90天；标准差只描述窗口间波动，不支持独立重复实验的推断。',
        '- 第1天对应各窗口第13天；首次为原始第13行，首次第180天为第192行，末次为第1903—2082行。','',
        '## 验证与复现','',
        '每一步检查输入移位与预测回填。替换首个起点之后全部评分标签，再运行完整180天，预测轨迹完全一致；也验证同一起点重复运行一致。所有指标及样本标准差从导出预测明细独立复算通过。输入文件哈希保持不变。',
        '运行时长{:.1f}秒，设备{}。环境和权重哈希见run_config.json，检查结果见verification.json，复现方法见目录内运行说明.md。'.format(config['elapsed_seconds'],config['runtime']['gpu'] or config['runtime']['device'])]
    (ROOT/'results/结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(key.to_string())

if __name__ == '__main__':
    main()

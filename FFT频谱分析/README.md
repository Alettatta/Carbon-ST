# 31省预测序列FFT修改

`results/provinces/`含31张独立TIFF。

审稿要求对应：去均值 → 线性去趋势 → 周期型Hann窗；每天一个样本，横轴cycles/day；频谱纵轴为归一化功率百分比，统一对数刻度；各省附主导周期及长/中/短频段能量比。

`results/处理说明.md`解释方法、分段、原FFT.m的缺项及有限预测序列的解释边界。`spectral_metrics_31.csv`为指标，`normalized_spectra.csv`为完整频谱，`preprocessed_values.csv`为每一步预处理数据。

```powershell
& 'D:/anaconda/envs/pytorch2/python.exe' -B 'scripts/generate_fft.py' --prediction-rows 180
& 'D:/anaconda/envs/pytorch2/python.exe' -B 'scripts/verify_fft.py'
```

使用Python实际生成并验证图表；`scripts/FFT_corrected.m`为原MATLAB代码的对应修订版本（本机未执行MATLAB），无需Signal Processing Toolbox。原FFT.m及原数据均只读取，新文件都位于本目录。

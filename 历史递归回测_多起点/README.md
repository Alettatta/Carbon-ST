# 多历史起点递归回测

## 数据与结果

- `inputs/`：历史数据、原始完整序列、epoch 165 权重、旧模型配置、空间特征及模型代码副本。`manifest.json`记录原来源，`local_checksums.json`记录本次输入哈希。
- `results/origins.csv`：22个起点、12天初始化和180天预测窗口的日期与原始行号（从1开始）。
- `results/prediction_details.csv`：122760条省份预测明细（22×180×31），保留真值、预测值及有符号误差。
- `results/metrics_by_origin.csv`：每起点、每时距的MAE、RMSE、MAPE百分数及有效数量。
- `results/metrics_by_horizon.csv`：各时距的平均MAE、MAE样本标准差、上下界、平均RMSE、平均MAPE及有效数量。
- `results/key_horizons.csv`：第1、7、30、60、90、180天的指标；这些是当天指标，而非累计平均。
- `results/multi_origin_mae.png`、`.pdf`：唯一的科学结果图；其他标注PDF是QA诊断文件。
- `results/结果报告.md`：中文解释、指标及局限；`verification.json`和`delivery_verification.json`保存检查结果。

Mean±SD先计算每个起点当天31省平均MAE，再对22个起点取均值和样本标准差（ddof=1）。

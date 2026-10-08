# B3 return-signal objective experiment

All candidate definitions and promotion gates were fixed before training in protocol.json. Selection records were written before holdout prediction. These results reuse the previously inspected holdout and cannot establish fresh out-of-sample improvement.

| Interval | Validation-selected candidate | Pass validation gate | Val R2 vs zero | Val IC | Val BA | Holdout R2 vs zero | Holdout IC | Holdout BA | Holdout amplitude ratio | Holdout price MAE | Flat price MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1h|mse-ensemble3|True|0.00237|0.05076|0.5026|-0.00296|-0.00635|0.4920|0.0456|0.30528%|0.30488%|
|15m|mse-ensemble3|True|0.00273|0.06006|0.5115|-0.00410|-0.02643|0.5095|0.0335|0.12910%|0.12896%|
|4h|mse-seed20261010|False|0.00307|0.02304|0.5017|-0.01592|0.01347|0.4995|0.0422|0.63827%|0.62769%|
|1d|huber-seed20261008|False|0.00810|0.08804|0.4809|-0.01711|-0.04449|0.4915|0.0943|1.83684%|1.82404%|

R2 vs zero = 1 - model return MSE / zero-return MSE. Amplitude ratio is predicted return std / realized return std; it is diagnostic, never an objective or promotion criterion. BA averages recall for positive and negative realized returns; flat predictions abstain and score zero. Constant nonzero majority-class predictions score 0.5. Pearson IC is undefined for constant predictions.

Time blocks are common calendar blocks across all four symbols; each block excludes its final origin so its h1 label stays inside the block. Adjacent observations are correlated. No statistical significance or tradable return claim is made.

Selected candidate JSON is stored under models/ for reproducibility only. Deployed model SHA256 values were unchanged. No curve magnification, artificial perturbation, minimum amplitude, or forced nonzero prediction was applied.

## 结论与部署决定

固定实验共训练 36 个网络（4 个周期 × 3 种损失 × 3 个种子），并评估各损失的三个种子等权平均，共 48 个候选配置。三个损失分别为标准化 Huber、标准化 MSE，以及 MSE 加 0.1 权重的下一根方向 BCE。四个周期都沿用同一时间切分、原始特征和 5,045 参数 CNN–LSTM–Attention 网络，未改变价格曲线的显示比例。

15 分钟和 1 小时的 MSE 集成通过了预先定义的验证筛选，但其后在既有留出时段复评时，收益 MSE 均未优于“收益为零”的基线，收益相关性为负；4 小时和日线没有候选通过验证筛选。此实验没有取得足以支持准确性升级的结果。验证达标本身属于探索性筛选，不代表稳定泛化能力。没有根据测试结果改选另一个候选，也没有覆盖当前部署权重。

输出标准差仅为实际下一根收益标准差的 3.35%–9.43%。这个比例解释曲线为何靠近当前价格；它只是诊断量，不是提高它就能改善准确率的目标。把收益人为放大或给曲线添加波动，不会让上述误差和方向表现变好。

方向补充诊断见 `direction-baseline-diagnostics.json`。4 小时的诊断候选在留出段 97.26% 的时间预测下跌，平衡准确率仍仅 49.95%，说明普通方向命中率可能掩盖单一方向偏置。该时期永远预测训练集多数方向（上涨）的普通命中率是 52.13%，平衡准确率是 50%。因此未来应继续同时报告普通命中率、平衡准确率、预测方向占比、收益相关性和收益误差。

原始训练报告中的 16 份缓存 SHA256 均已逐一核对一致，记录见 `cache-integrity.json`。部署权重前后哈希记录见 `deployment-integrity.json`。实验使用 CPU 6 线程；运行命令如下：

```powershell
.\.venv-training\Scripts\python.exe scripts/experiment-b3-signal.py --threads 6 --epochs 24
```

后续增加数据范围或输入特征、改用方向和幅度分头学习，均需要另立固定实验，并保留新的前向观察段；不能继续把这一留出时段称为从未使用过的独立测试。

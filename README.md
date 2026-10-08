# 观潮 guanchao

本地币安现货监控台。只读公开行情，不下单。默认 B4 使用本地 Python 服务分别推断方向概率、绝对变化幅度和价格均值；B3 的 CNN–LSTM–Attention 保留在浏览器 Worker 中作对照，模型结果和技术面规则分开显示。

新训练在“下一根绝对变化幅度”上取得可测的历史误差改善，方向仍是弱信号，完整价格路径未证明稳定优势。蓝色均值接近上一收盘不代表市场会走平；可单独打开绿色“上下行情景”查看模型预估的波动规模。情景不是目标价或置信区间，未人为放大均值线。

首次运行需要 Python 3.10+ 和已保存的 B4 权重（`reports/experiments/b4-trees`、`b4-magnitude`）：

```powershell
npm install
python -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install -r scripts/requirements-trees.txt
npm run dev
```

浏览器打开 http://127.0.0.1:5173 。`npm run dev` 自动启动只监听 `127.0.0.1:8765` 的模型服务并启动 Vite，已运行的服务会复用。`npm run model:serve` 可单独启动服务；`npm run dev:web` 仅启动网页。B3 对照不需要 Python 服务。可以用 `GUANCHAO_PYTHON` 指定另一个 Python 路径。生产构建使用 `npm run build`、`npm run preview`，preview 同样代理本地服务；B4 不能仅靠静态页面推理。

## B4 训练结果与显示口径

冻结的行情共 347,681 根，四币种合计 339,044 个训练/评估起点。每个周期各自训练；57 项因果特征包含多尺度收益、波动、成交笔数与主动买入活动。数据按时间分成训练、调参、三块验证、校准和末段复评，每个边界剔除跨段的未来 12 根标签。

树模型分别训练 12 个价格均值头、12 个绝对幅度头及 +1/+3/+6/+12 四个方向头。下面是**下一根**在四币种历史复评中的结果，幅度改善相对“过去 20 根绝对收益的均值”，不是带方向的价格误差改善：

| 周期 | 幅度 MSE 降低 | 幅度 MAE 降低 | 方向平衡准确率 |
|---|---:|---:|---:|
| 15min | 9.45% | 4.03% | 52.06% |
| 1h | 9.74% | 3.45% | 52.76% |
| 4h | 8.02% | 3.07% | 52.36% |
| 1d | 7.59% | 1.47% | 50.00% |

15min/1h/4h 的下一根幅度改善通过滚动验证与复评的时间块检验。日线仅 197 个独立复评时点，MAE 改善区间跨零，界面保留“尚未通过全部验证”。4h 方向还弱于上一根反转基线；日线方向未通过验证。较远步数没有统一通过：4h 的 +6/+12 幅度 MAE 和日线 +12 的幅度 MSE/MAE 都可差于基线。所有“通过”标识只适用于下一根。

- 15min 每 15 分钟、1h 每小时、4h 每四小时、1d 每天一个节点。B4 的“同起点回放”每次固定一个历史收盘起点，一次画出当时预测的未来 12 根，期间不重新输入后来的行情；起点滑块仍每次移动一根，最多保留 240 个完整起点。
- 每次只使用确认收盘的行情，未收盘报价不重画预测；补齐 REST/WebSocket 的主动买入和成交特征，缺值拒绝推断，不补成零。
- 均值线、10%/90% 校准误差带、±绝对幅度情景分开绘制；+2/+4 等未训练的方向概率显示“—”，不插值。
- 实际涨跌持平样本不进入方向训练；上涨与下跌概率以发生涨跌为条件。幅度百分比为 `100 * expm1(E|log return|)` 的展示换算，不是有符号收益期望。
- 回放仅允许拟合和校准截止之后的起点，使用当时可用的特征重算；它不是当时实时存档，也不构成新未来测试。近期复评时段在之前实验中已看过。

详见 [幅度训练报告](reports/experiments/b4-magnitude/report.md)、[方向与均值报告](reports/experiments/b4-trees/report.md)、[本地服务协议](scripts/B4-SERVICE.md)。另训练的 113,433 参数 TCN–Attention 神经网络四周期均未超过持平收益基线，保留权重和[审计结果](reports/experiments/b4-neural/completion-audit.md)，未替代面板模型。

```powershell
# 数据已冻结则直接复用；从现有协议与缓存复现树模型
.\.venv-training\Scripts\python.exe scripts/train-b4-trees.py --threads 6
.\.venv-training\Scripts\python.exe scripts/train-b4-magnitude.py --threads 6
# 实际模型、HTTP、逐根回放与前缀因果性验证
.\.venv-training\Scripts\python.exe scripts/test_b4_service.py
```

## 历史回放与未来预测的公平对照

旧紫线把每根重新预测的下一根结果连在一起，未来蓝线却是同一起点一次预测 12 根；前者不断获得新的实际行情，不能据其贴近价格就判断后者准确。B4 默认改为同一起点的 12 步路径，并同时绘制实际收盘价和全程复制起点价格的持平基线。点击“定位回放”查看选中起点；平常行情刷新不会移动视窗。

右侧“同一起点 12 步复评”对同一组完整起点报告 +1/+3/+6/+12 误差及全部 12 步平均误差。历史重算仍不等于实时首发。固定缓存中 BTC 1h 最近 240 个共同起点的 +12 价格 MAE 为模型 0.78570%、持平基线 0.76496%；这次显示修正没有改善模型权重，也没有证明价格预测优势。四周期、前缀一致性和未来扰动审计见 [回放审计报告](reports/b4-replay-audit.md)。

本地服务现在把实际请求时首次发出的预测存入 `.cache/forward-evaluation.sqlite3`，以模型版本、币种、周期和起点隔离，首次预测和发布时间不可覆盖，不导入历史回放。起点收盘后最多 60 秒发出才计入成绩；迟到记录保留但排除评分。真实目标 K 线收盘且服务收到行情后才结算，未结算显示“尚无已结算实时样本”。记录仅在页面调用本地服务时产生，不是后台持续采集；关闭页面期间不会补造首发记录。此本地账本用于复核，不是可验证行情来源的第三方审计凭证。

## K 线缩放与拖动

拖动或滚轮缩放时会固定纵轴比例，图内左下方状态显示“纵轴已固定”，实时行情和模型刷新不会重新拉伸 K 线。预测、情景、历史回放和指标线保留绘制，但不参与 K 线价格轴的自动定标。右侧价格轴仍可手动拖动调整；点击图内“适应价格”或双击右侧价格轴，可按当前可见 K 线重新适配；“定位预测”回到最新区域并保留当前水平缩放宽度。

普通行情更新不再恢复旧的时间视窗，只有切换交易对/周期或滚动历史窗口实际删除旧 K 线时才初始化/补偿坐标。模型加载前后保留相同说明文字，避免异步结果改变画布高度。

## B3 模型与历史回放

- 四周期独立权重：`15m`、`1h`、`4h`、`1d`，覆盖 `BTCUSDT`、`ETHUSDT`、`BNBUSDT`、`SOLUSDT`。其他交易对继续显示行情与技术指标，但不套用未经评估的模型。
- 48 根历史 × 8 个因果特征，经过因果 Conv1D、LSTM 和时间注意力，直接输出未来 1–12 根的累计收益；不再把预测价格伪造成 OHLCV 后逐步递推。
- 仅在 K 线确认收盘后刷新。蓝线是未来预测；淡色区间来自独立校准集残差的 10%/90% 分位，标称覆盖 80%，不是保证。
- 紫色历史回放每根收盘时预测下一根：15m 每 15 分钟一个点、1h 每小时一个点，4h 与日线同理。默认保留最近 240 个已完成的逐根预测，直接对照目标 K 线实际收盘价。只允许拟合、选择和校准全部结束之后的样本；这是冻结模型的历史回放，不是此前实时存档。
- 面板主价、涨跌幅和测试指标均针对下一根。训练损失对下一根赋予 60% 权重，其余 11 步合计 40%；验证集优先按下一根误差选择权重，并为每个预测步数分别选择收缩系数。未来 12 根曲线保留为延伸参考，全步指标在折叠详情中。
- 四币种下一根离线测试与当前币种逐根回放分开显示。两者的 MAE 都以实际目标价格为分母，但统计币种与时间范围不同。验证集如果选择下一根 `blend=0`，显示“持平基线／下一根无方向信号”，不影响其他步数独立的系数。

当前逐根优先实验的详细结果见 [测试报告](reports/b3-training-report.md)；第一版多步实验已归档在 `reports/experiments/multi-horizon-v1/`。旧权重未随原仓库提交，旧方案对照是依旧源码架构与主要训练规则重新训练的参考，不是原权重 A/B。逐根目标调整后对同一留出时段重新评估，未使用测试集选模型，但这不是新增的未看过测试时段。预测节点更密不会自动证明模型更准，应同时对照持平基线及真实留出误差。

## 方向与幅度诊断

用户明确优先方向与幅度后，又进行了两轮预先固定协议的实验：3 种损失 × 3 个种子，以及增加主动买入量/成交活动后的 2 种损失 × 3 个种子，四周期共训练 60 个网络、评估 20 个三成员集成。checkpoint 与候选只在验证段选择，选择记录写入后才复评已有留出段。两轮都没有找到足以支持预测能力升级的持续收益信号，因此实验候选未替换部署权重。

- [当前模型平线诊断数值](reports/b3-flat-diagnostic.json)：收益输出幅度、收益相关性、方向命中、相对持平的收益误差。
- [训练目标与集成实验](reports/experiments/signal-objectives/report.md)：Huber、MSE、MSE + 方向 BCE。
- [成交活动特征实验](reports/experiments/orderflow-features/report.md)：增加主动买入不平衡和成交笔数等 4 个因果特征。

实验使用相同冻结数据与已查看过的留出时段，不能称为新的独立测试。实验权重采用独立 schema，仅用于复现；不能直接覆盖 `public/models`。界面不因 MAE 微小下降就显示方向成功，不放大蓝线振幅、不添加人工波动。收益小图至少保留 ± 下一根历史 MAE 的纵轴范围，避免把近零输出视觉夸大。

## 重新训练

已保存 `public/models/b3-*.json`，仅运行 B3 对照不需要 Python、GPU、交易所 API 密钥或再次训练。首次 B3 训练环境安装（Windows PowerShell，Python 3.10+）：

```powershell
python -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install -r scripts/requirements-training.txt
.\.venv-training\Scripts\python.exe scripts/train-b3.py --epochs 40 --device auto --threads auto
```

脚本从 Binance 公共接口下载已闭合行情并缓存到 `.cache/training/`。共同时间边界分为训练、验证、校准和测试；标签跨越边界的起点会剔除。固定随机种子，训练统计量只来自训练段，验证负责早停与收缩系数，校准负责误差带，测试不参与选择。重现当前数据截止时间的完整命令见测试报告；训练会更新对应 JSON 权重及报告。不要根据已经查看过的测试结果反复调参，再把同一段称作新独立测试。

```powershell
# 旧单步 LSTM 架构复现诊断，需要先有训练数据缓存
.\.venv-training\Scripts\python.exe scripts/benchmark-legacy.py
```

旧 `src/lib/lstm.ts` 和 `scripts/train-lstm.ts` 保留作历史参考，B3 对照使用 `src/lib/b3.ts` 与 `src/workers/b3.worker.ts`。

## 多核与训练设备

神经网络训练自动检测设备，`--device cpu` 强制 CPU，`--device cuda` 要求可用的 CUDA 版 PyTorch；不可用时明确报错。本机独立 `.venv-training-cuda` 已安装 PyTorch 2.14.1+cu130，RTX 5070 通过真实 TCN 前向、反向、索引采样与权重重载测试。此前完成的四周期 TCN 权重由 CPU 6 线程训练，累计约 28.10 分钟，并未冒称为 GPU 重训。

同一个 `256×64×57` 批次，113,433 参数 TCN 的 GPU 训练吞吐为 22,733 样本/秒，CPU 6 线程为 2,354 样本/秒，微基准 **9.66 倍**。这不包含数据准备和完整验证，不等于全流程加速倍数。CPU/CUDA 冻结权重预测概率最大差 `6.41e-8`，具体见 [GPU 实测](reports/experiments/b4-neural/device-benchmark.md)。

```powershell
# 查看 GPU 训练参数；新实验前先保留已有冻结记录，避免覆盖同名输出目录
.\.venv-training-cuda\Scripts\python.exe scripts/train-b4-neural.py --help
# 吞吐测量，不保存临时训练参数
.\.venv-training-cuda\Scripts\python.exe scripts/train-b4-neural.py --benchmark --threads 6 --batch-size 256
```

`--threads auto` 优先读取本机基准推荐值。当前 Ryzen 9 7945HX（16 核 / 32 线程）上，实测 6 个计算线程为 34,596 样本/秒，约单线程 1.86 倍；8–16 线程对这个 5,045 参数的小网络更慢。这是当前 batch/架构下的短时实测，不代表更大网络也应选 6 线程，换模型或设备后应重新测量。

```powershell
# 训练吞吐基准，不修改模型
.\.venv-training\Scripts\python.exe scripts/benchmark-training.py
# 验证设备、训练、推断、权重导出路径，不下载或写权重
.\.venv-training\Scripts\python.exe scripts/train-b3.py --device auto --threads auto --dry-run
# 试验结果写到独立目录，保留当前已验证模型
.\.venv-training\Scripts\python.exe scripts/train-b3.py --output-dir reports/experiments/next-trial
```

具体测量与设备限制见 [计算基准报告](reports/b3-compute-benchmark.md)。

## 验证

```powershell
npm test
npm run lint
npm run build
```

测试包含四周期 PyTorch/TypeScript 特征、网络输出与预测价格一致性，时间分区、防未来数据泄漏、回放稳定性、Worker 加载和并发结果，以及 REST/WebSocket 最终收盘数据恢复。

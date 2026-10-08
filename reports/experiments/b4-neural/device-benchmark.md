# B4 TCN CPU / CUDA 算力与重载一致性实测

PyTorch 2.14.1+cu130，CUDA 13.0，GPU NVIDIA GeForce RTX 5070。1h已保存模型、同一真实训练批次：256样本×64根×57特征，113,433参数，float32，关闭TF32。CPU使用6计算线程。

仅在内存中的临时副本做前向、损失、反向、梯度裁剪和AdamW更新；预热5步，再重复3组各10步取中位数。没有重训已完成四周期，没有保存测量产生的参数。

| 设备 | 样本/秒 | 毫秒/训练步 |
|---|---:|---:|
|cpu|2353.7|108.77|
|cuda|22733.0|11.26|

此批次GPU吞吐为CPU6的 **9.66倍**。GPU环境可用不等于预测精度改善。此测量不包含数据准备、训练采样索引、验证评估和每步日志同步，不等于完整训练或单根实时推理的加速倍数。

原始冻结1h权重在CPU与CUDA的最大绝对误差：logReturn=1.8e-09；上涨概率=6.41e-08；上下界分别1.8e-09/1.8e-09。通过预设收益1e-6、概率1e-5容差。四周期权重/manifest及原B3权重前后SHA256均完全一致。

脚本现支持 `--device auto/cpu/cuda`。`NeuralPredictor(folder, device='cuda')` 可显式进行GPU推理，默认仍为CPU。

```powershell
.\.venv-training-cuda\Scripts\python.exe scripts/train-b4-neural.py --benchmark --threads 6 --batch-size 256
```

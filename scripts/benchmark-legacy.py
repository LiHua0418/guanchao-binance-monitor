"""Reproduce the legacy single-step LSTM architecture on frozen B3 partitions.

This diagnostic never writes B3 models or selects a model using test results.
The repository has no old trained weights: this is an architecture/training-rule
reproduction in PyTorch, not a replay of the user's missing old trained model.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location("b3_training", __file__.replace("benchmark-legacy.py", "train-b3.py"))
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)


def legacy_features(candles):
    high, low, close, volume = (candles[:, i] for i in (2, 3, 4, 5))
    ema = np.zeros(len(close), dtype=np.float64)
    ema[19] = close[:20].mean()
    for i in range(20, len(close)):
        ema[i] = close[i] * (2 / 21) + ema[i - 1] * (19 / 21)
    raw = np.zeros((len(close), 4), dtype=np.float64)
    raw[1:, 0] = np.log(close[1:] / close[:-1])
    raw[:, 1] = (high - low) / close
    valid_volume = (volume[1:] > 0) & (volume[:-1] > 0)
    raw[1:, 2][valid_volume] = np.log(volume[1:][valid_volume] / volume[:-1][valid_volume])
    raw[19:, 3] = (close[19:] - ema[19:]) / close[19:]
    return raw, ema


def legacy_partitions(groups, interval):
    parts, _, _, _, splits = b3.prepare(groups, interval)
    result = {}
    computed = {symbol: legacy_features(candles) for symbol, candles in groups.items()}
    for name, part in parts.items():
        raw, emas = [], []
        for timestamp, symbol in zip(part["time"], part["symbol"]):
            anchor = int(np.searchsorted(groups[symbol][:, 0], timestamp))
            raw.append(computed[symbol][0][anchor - 15:anchor + 1])
            emas.append(computed[symbol][1][anchor])
        result[name] = {"raw": np.asarray(raw), "ema": np.asarray(emas), "y": part["y"],
                        "time": part["time"], "close": part["close"], "symbol": part["symbol"]}
    # Match legacy standardize(): each row of every overlapping training window.
    mean = result["train"]["raw"].mean(axis=(0, 1))
    scale = result["train"]["raw"].std(axis=(0, 1))
    scale[scale < 1e-8] = 1.0
    for part in result.values():
        part["x"] = torch.from_numpy(((part["raw"] - mean) / scale).astype(np.float32))
    return result, mean, scale, splits


class Legacy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 16, batch_first=True)
        self.output = nn.Linear(16, 1)
        # Legacy implementation has one gate bias vector; freeze the redundant
        # PyTorch hidden bias at zero to preserve 1,361 trainable parameters.
        with torch.no_grad():
            self.lstm.weight_ih_l0.normal_(0, math.sqrt(2 / 20))
            self.lstm.weight_hh_l0.normal_(0, math.sqrt(1 / 16))
            self.lstm.bias_ih_l0.zero_()
            self.lstm.bias_ih_l0[16:32].fill_(1)
            self.lstm.bias_hh_l0.zero_()
            self.output.weight.normal_(0, math.sqrt(1 / 16))
            self.output.bias.zero_()
        self.lstm.bias_hh_l0.requires_grad_(False)

    def forward(self, x):
        hidden, _ = self.lstm(x)
        return self.output(hidden[:, -1]).squeeze(-1)


def predict_single(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(batch).numpy() for batch in x.split(512)])


def fit(parts, interval, max_epochs):
    torch.manual_seed(b3.SEED)
    random.seed(b3.SEED)
    np.random.seed(b3.SEED)
    model = Legacy()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.003)
    target = torch.from_numpy(parts["train"]["y"][:, 0].astype(np.float32))
    best_loss, best_state, best_epoch, wait, decays = float("inf"), None, 0, 0, 0
    generator = torch.Generator().manual_seed(b3.SEED)
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=generator)
        total = 0.0
        for indices in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            pred = model(parts["train"]["x"][indices])
            loss = F.mse_loss(pred, target[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        validation = predict_single(model, parts["validation"]["x"])
        val_loss = float(np.mean((validation - parts["validation"]["y"][:, 0]) ** 2))
        history.append({"epoch": epoch, "trainMSE": total / len(target), "validationOneStepMSE": val_loss,
                        "learningRate": optimizer.param_groups[0]["lr"]})
        if val_loss < best_loss * (1 - 1e-4):
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait == 8 and decays < 3:
                for group in optimizer.param_groups:
                    group["lr"] *= .5
                decays += 1
                wait = 0
            elif wait >= 12 and epoch >= 20:
                break
        if epoch == 1 or epoch % 5 == 0:
            b3.log(f"LEGACY {interval} epoch={epoch} valMSE={val_loss:.8f} best={best_epoch} wait={wait}")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def recursive_forecasts(model, part, mean, scale):
    # Keep the same synthetic candles as old inferForecast(): predicted closes,
    # high=low=close, last volume copied, EMA20 updated; no actual future inputs.
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(part["y"]), 512):
            end = start + 512
            x = part["x"][start:end].clone()
            close = part["close"][start:end].copy()
            ema = part["ema"][start:end].copy()
            total = np.zeros(len(close), dtype=np.float64)
            horizon = []
            for _ in range(b3.HORIZON):
                step = np.clip(model(x).numpy().astype(np.float64), -.08, .08)
                total += step
                horizon.append(total.copy())
                close *= np.exp(step)
                ema = close * (2 / 21) + ema * (19 / 21)
                row = np.column_stack((step, np.zeros(len(close)), np.zeros(len(close)), (close - ema) / close))
                scaled = torch.from_numpy(((row - mean) / scale).astype(np.float32)).unsqueeze(1)
                x = torch.cat((x[:, 1:], scaled), dim=1)
            outputs.append(np.stack(horizon, axis=1))
    return np.concatenate(outputs)


def benchmark(interval, report, args, cutoff):
    started = time.monotonic()
    groups = {}
    for symbol in b3.SYMBOLS:
        groups[symbol], sha = b3.fetch_history(symbol, interval, b3.JOBS[interval], cutoff)
        if sha != report["data"][symbol]["sha256"]:
            raise ValueError(f"Original data hash mismatch for {symbol} {interval}")
    parts, mean, scale, splits = legacy_partitions(groups, interval)
    if splits != report["splits"]:
        raise ValueError("Legacy and B3 partitions differ")
    model, epoch, history = fit(parts, interval, args.epochs)
    # Calibrate on the same separate segment; never estimate uncertainty on test.
    calibration = parts["calibration"]
    cal_pred = recursive_forecasts(model, calibration, mean, scale)
    lower, upper = np.quantile(calibration["y"] - cal_pred, [.1, .9], axis=0)
    test = parts["test"]
    prediction = recursive_forecasts(model, test, mean, scale)
    metrics = b3.evaluate(test["y"], prediction, lower, upper)
    by_symbol = {s: b3.evaluate(test["y"][np.array(test["symbol"]) == s],
                               prediction[np.array(test["symbol"]) == s], lower, upper) for s in b3.SYMBOLS}
    b3.log(f"LEGACY RESULT {interval} MAE={metrics['modelMaePct']:.5f}% B3={report['metrics']['modelMaePct']:.5f}% baseline={metrics['baselineMaePct']:.5f}% epoch={epoch}")
    return {"interval": interval, "legacyMetrics": metrics, "b3Metrics": report["metrics"],
            "b3Blend": report["blend"], "b3BlendByHorizon": report.get("blendByHorizon"),
            "b3TrainingObjective": report.get("trainingObjective", "multi-horizon-v1"),
            "legacyPerSymbol": by_symbol,
            "legacySelectedEpoch": epoch, "legacyEpochsRun": len(history), "legacyHistory": history,
            "b3MaeChangeVsLegacyPct": (report["metrics"]["modelMaePct"] / metrics["modelMaePct"] - 1) * 100,
            "b3NextBarMaeChangeVsLegacyPct": (report["metrics"]["byHorizon"][0]["maePct"] / metrics["byHorizon"][0]["maePct"] - 1) * 100,
            "legacyTrainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "splits": splits, "durationSeconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    source = json.loads((b3.REPORTS / "b3-training-report.json").read_text(encoding="utf-8"))
    cutoff = int(datetime.fromisoformat(source["cutoff"].replace("Z", "+00:00")).timestamp() * 1000)
    # Detect accidental writes to deployed/frozen B3 artifacts.
    hashes_before = {i: b3.hashlib.sha256((b3.MODELS / f"b3-{i}.json").read_bytes()).hexdigest() for i in b3.JOBS}
    results = []
    for report in source["intervals"]:
        results.append(benchmark(report["interval"], report, args, cutoff))
        b3.write_json(b3.REPORTS / "b3-legacy-comparison.json", {"seed": b3.SEED,
                      "kind": "legacy architecture/training-rule reproduction, not old trained weights",
                      "b3TrainingObjective": source.get("trainingObjective", "multi-horizon-v1"),
                      "evaluationStatus": "same holdout period reevaluated; not a fresh unseen test",
                      "b3ArtifactsFrozenSha256": hashes_before, "testReusedForDiagnosticComparison": True,
                      "noTestSelection": True, "intervals": results})
    for interval, digest in hashes_before.items():
        assert b3.hashlib.sha256((b3.MODELS / f"b3-{interval}.json").read_bytes()).hexdigest() == digest
    lines = ["# 下一根优先 B3 与旧单步 LSTM 方案的复现对比", "",
             "旧预训练权重未在仓库中，因此本报告比较的是按旧源码架构及主要训练规则重训的基线，不是用户原先那份模型的实测 A/B。B3 权重全程冻结，SHA256 见 JSON。", "",
             "本版对照 trainingObjective=next-bar-priority 的当前权重。用户明确下一根业务目标后已重新训练B3；仍评估同一个留出时段，因此不能称为新增未看过的独立测试。第一版比较已归档至 reports/experiments/multi-horizon-v1。", "",
             "## 主要目标：下一根 K 线", "",
             "| 周期 | 旧 LSTM h1 MAE | B3 h1 MAE | B3 相对旧基线变化 | 持平 h1 MAE | 旧 h1 方向命中 | B3 h1 方向命中 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        old, new = r["legacyMetrics"]["byHorizon"][0], r["b3Metrics"]["byHorizon"][0]
        direction = f"{new['hitRate']*100:.1f}%" if r["b3Blend"] else "持平回退"
        lines.append(f"| {r['interval']} | {old['maePct']:.5f}% | {new['maePct']:.5f}% | {r['b3NextBarMaeChangeVsLegacyPct']:+.2f}% | {old['baselineMaePct']:.5f}% | {old['hitRate']*100:.1f}% | {direction} |")
    lines += ["", "## 辅助目标：未来 12 根整体", "",
              "| 周期 | 旧 LSTM 递推 MAE | B3 MAE | B3 相对旧基线变化 | 持平 MAE |", "|---|---:|---:|---:|---:|"]
    for r in results:
        old, new = r["legacyMetrics"], r["b3Metrics"]
        lines.append(f"| {r['interval']} | {old['modelMaePct']:.4f}% | {new['modelMaePct']:.4f}% | {r['b3MaeChangeVsLegacyPct']:+.2f}% | {old['baselineMaePct']:.4f}% |")
    lines += ["", "## 公平比较范围", "",
              "- 使用与 B3 完全相同的缓存行情、SHA256、四币、时间边界、预测起点和 12 个累计收益标签；训练、验证、校准、测试没有跨段目标。",
              "- 旧方案：16 根 × 4 特征（单根对数收益、振幅/close、成交量log比、(close-EMA20)/close）→ 单层 LSTM16 → 单步线性输出，1361个可训练参数。训练标准化按旧代码重复窗口逐行拟合，无clip、无目标标准化。",
              "- 旧规则复现：正态权重初始化、forget bias=1、MSE、Adam(lr .003)、batch32、梯度范数clip1、至多80轮、至少20轮、验证单步MSE选择checkpoint、耐心8轮减半lr至多3次再耐心12轮停止。PyTorch FP32和固定seed替代旧JS FP64及Math.random，数值轨迹不相同。",
              "- 为共享严格划分，旧方案也使用 B3 的起点与 12 根边界隔离，而非旧源码末48/48根分割；因此是架构及训练规则基线，不是逐字逐位重演。",
              "- 推理每步限幅 ±0.08 对数收益，预测值递推12次；合成未来K线 high=low=close、volume沿用上一根、EMA逐步更新，不读取任何未来实际价格。",
              "- 旧基线也使用独立校准集的逐horizon残差10%/90%区间。旧代码在近期样本估区间的方式不用于本测试，避免测试目标泄漏。",
              "- 所有MAE采用B3相同的未来实际价格相对误差定义。主要表格只评估下一根h1，辅助表格平均12个horizon。负的相对变化代表B3较复现基线误差下降。逐币/逐horizon完整数据见JSON。",
              "- 这是在B3测试结果已知后追加的诊断比较，未据测试结果选择或修改任何B3模型。它不能充当第二份全新独立测试，也不支持对原缺失模型的绝对提升承诺。",
              "- 新版B3按每个horizon分别选择blend；某一步blend=0表示该步持平回退。bundle.blend仅表示h1，不能把它套用给其余预测步，也不能把持平回退解读为神经网络预测出了方向。",
              "", "```powershell", ".\\.venv-training\\Scripts\\python.exe scripts/benchmark-legacy.py --epochs 80 --threads 6", "```"]
    (b3.REPORTS / "b3-legacy-comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

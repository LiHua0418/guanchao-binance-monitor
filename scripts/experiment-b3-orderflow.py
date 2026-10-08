"""Final preregistered B3 feature experiment; never modifies deployed weights.

Adds four causal activity/taker-volume features available in the frozen public
Binance klines. The test period is already inspected and is a reevaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("signal_exp", ROOT / "scripts/experiment-b3-signal.py")
signal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signal)
b3 = signal.b3
OUT = ROOT / "reports/experiments/orderflow-features"
BASE_FEATURES = b3.features
BASE_NETWORK = b3.B3
OBJECTIVES = ["mse", "mse-direction"]
FEATURE_NAMES = b3.FEATURES + ["takerBuyImbalance", "logTradeCountChange",
                               "tradeCountSma20Distance", "logAverageTradeBaseVolumeChange"]


def features(candles):
    base = BASE_FEATURES(candles)
    volume, trades, buy_volume = candles[:, 5], candles[:, 6], candles[:, 7]
    extra = np.zeros((len(candles), 4), dtype=np.float64)
    nonzero = volume > 0
    extra[nonzero, 0] = 2 * buy_volume[nonzero] / volume[nonzero] - 1
    extra[1:, 1] = np.diff(np.log1p(trades))
    rolling_sum = np.cumsum(np.concatenate(([0.], trades)))
    average = (rolling_sum[20:] - rolling_sum[:-20]) / 20
    extra[19:, 2] = np.divide(trades[19:], average, out=np.ones_like(average), where=average > 0) - 1
    average_trade_base_volume = volume / np.maximum(trades, 1)
    extra[1:, 3] = np.diff(np.log1p(average_trade_base_volume))
    return np.concatenate((base, extra), axis=1)


class OrderflowB3(BASE_NETWORK):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(12, 16, 3)


def cached_groups(interval, cutoff, expected):
    groups, hashes = {}, {}
    for symbol in b3.SYMBOLS:
        last_open = cutoff // b3.INTERVAL_MS[interval] * b3.INTERVAL_MS[interval] - b3.INTERVAL_MS[interval]
        path = b3.CACHE / f"{symbol}-{interval}-{b3.JOBS[interval]}-{last_open}.json"
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected[symbol]:
            raise ValueError(f"Original source cache hash mismatch: {path}")
        rows = json.loads(content)
        if any(len(r) < 10 or int(r[6]) >= cutoff for r in rows):
            raise ValueError(f"Missing kline fields or unclosed kline: {path}")
        candles = np.array([[int(r[0]), *map(float, r[1:6]), float(r[8]), float(r[9])] for r in rows])
        if (not np.isfinite(candles).all() or np.any(candles[:, 1:5] <= 0) or
                np.any(candles[:, 5:] < 0) or
                np.any(candles[:, 7] > candles[:, 5] + 1e-8) or
                not np.all(np.diff(candles[:, 0]) == b3.INTERVAL_MS[interval])):
            raise ValueError(f"Invalid order-flow cache data: {path}")
        full = features(candles)
        for index in (49, 96, len(candles) // 2, len(candles) - 1):
            if not np.allclose(full[index], features(candles[:index + 1])[-1], rtol=1e-12, atol=1e-12):
                raise AssertionError("Noncausal feature: prefix and full-series values disagree")
        groups[symbol], hashes[symbol] = candles, digest
    return groups, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", nargs="+", choices=list(b3.JOBS), default=["1h", "15m", "4h", "1d"])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    b3.resolve_runtime("cpu", args.threads)
    b3.features, b3.B3 = features, OrderflowB3
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT / "reports/b3-training-report.json").read_text(encoding="utf-8"))
    cutoff = int(datetime.fromisoformat(original["cutoff"].replace("Z", "+00:00")).timestamp() * 1000)
    deployment_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "public/models").glob("b3-*.json")}
    protocol = {"createdAt": datetime.now(timezone.utc).isoformat(), "cutoffMs": cutoff,
                "objectives": OBJECTIVES, "seeds": signal.SEEDS, "maxEpochs": args.epochs,
                "cpuThreads": args.threads, "parameterCount": sum(p.numel() for p in OrderflowB3().parameters()),
                "network": "causal Conv1d 12->16 kernel3; unchanged LSTM24, attention and direct cumulative-return head12",
                "features": FEATURE_NAMES,
                "featureDefinitions": {
                    "takerBuyImbalance": "2 * kline[9] / kline[5] - 1; zero volume -> 0",
                    "logTradeCountChange": "log1p(kline[8]_t) - log1p(kline[8]_(t-1))",
                    "tradeCountSma20Distance": "trades_t / mean(trades_(t-19)..trades_t) - 1; zero denominator -> 0",
                    "logAverageTradeBaseVolumeChange": "log1p(volume_t/max(trades_t,1)) - log1p(volume_(t-1)/max(trades_(t-1),1)); base-asset units before per-feature training-only standardization"},
                "featureSource": "https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data",
                "causalityChecks": "16 source cache SHA256s must match original report; no close time >= frozen cutoff; prefix/full feature equivalence; all features/target scalers only train",
                "candidateDefinition": "MSE or MSE+.1 h1 direction BCE; seeds20261008/09/10 each and same-objective equal ensemble; no other candidates",
                "selection": "same validation normalized return MSE .8*h1+.2*mean(h2..12), early-stop6, no shrinkage, no amplitude floor",
                "gate": "h1 validation R2vsZero>0; MSE<trainMean; IC>.02; BA>.5; priceMAE<=flat*1.005; >=2/3 timestamp blocks have positive R2 and IC",
                "holdoutStatus": "same already inspected historical holdout reevaluation; test never chooses objective, epoch, seed, ensemble or gate thresholds",
                "experimentBoundary": "This is the final feature experiment in this request; no additional search after its result.",
                "deployment": "experiment artifacts only; never overwrite public/models; 12 input features and ensemble need runtime support before any future integration",
                "deploymentHashesBefore": deployment_before}
    signal.save(OUT / "protocol.json", protocol)
    (OUT / "protocol.md").write_text("# B3 order-flow features: protocol\n\nRecorded before training.\n\n" +
        "\n".join(f"- **{k}**: {v}" for k, v in protocol.items()) + "\n", encoding="utf-8")
    reports = []
    for interval in args.intervals:
        start = time.monotonic()
        expected = next(r["data"] for r in original["intervals"] if r["interval"] == interval)
        groups, hashes = cached_groups(interval, cutoff, {s: expected[s]["sha256"] for s in b3.SYMBOLS})
        parts, mean, scale, target_scale, splits = b3.prepare(groups, interval)
        val = parts["validation"]
        train_mean = parts["train"]["y"].mean(axis=0)
        mean_val = signal.assessment(val, np.broadcast_to(train_mean, val["y"].shape), interval)
        networks, candidates, val_predictions = {}, {}, {}
        for objective in OBJECTIVES:
            members = []
            for seed in signal.SEEDS:
                name = f"{objective}-seed{seed}"
                model, run = signal.fit(parts, target_scale, objective, seed, args.epochs, interval)
                networks[name] = model
                members.append(name)
                predicted = b3.predict(model, val["x"]) * target_scale
                val_predictions[name] = predicted
                assessed = signal.assessment(val, predicted, interval)
                candidates[name] = {"members": [name], "run": run,
                                    "validationScore": signal.score(val["y"], predicted, target_scale),
                                    "validation": assessed, "gate": signal.eligibility(assessed, mean_val)}
            predicted = np.mean([val_predictions[n] for n in members], axis=0)
            assessed = signal.assessment(val, predicted, interval)
            candidates[f"{objective}-ensemble3"] = {
                "members": members, "validationScore": signal.score(val["y"], predicted, target_scale),
                "validation": assessed, "gate": signal.eligibility(assessed, mean_val)}
        eligible = [n for n, c in candidates.items() if c["gate"]["eligible"]]
        selected = min(eligible or candidates.keys(), key=lambda n: candidates[n]["validationScore"])
        frozen = {"interval": interval, "selected": selected, "eligible": bool(eligible),
                  "eligibleCandidates": eligible, "selectionFrozenAt": datetime.now(timezone.utc).isoformat(),
                  "candidates": candidates, "splits": splits, "dataSha256": hashes}
        signal.save(OUT / f"{interval}-selection-frozen.json", frozen)
        test = parts["test"]
        for name, candidate in candidates.items():
            prediction = np.mean([b3.predict(networks[n], test["x"]) for n in candidate["members"]], axis=0) * target_scale
            candidate["sameHoldoutReevaluation"] = signal.assessment(test, prediction, interval)
            candidate["testDirectionFractionsH1"] = {"up": float(np.mean(prediction[:, 0] > 1e-12)),
                                                    "down": float(np.mean(prediction[:, 0] < -1e-12))}
        chosen = candidates[selected]
        model_members = [{"name": n, "weights": b3.weights(networks[n])} for n in chosen["members"]]
        cal_prediction = np.mean([b3.predict(networks[n], parts["calibration"]["x"]) for n in chosen["members"]], axis=0) * target_scale
        lower, upper = np.quantile(parts["calibration"]["y"] - cal_prediction, [.1, .9], axis=0)
        model_export = {"schemaVersion": "experimental-orderflow-v1", "interval": interval,
                        "name": "guanchao-b3-orderflow-experimental", "selected": selected,
                        "architecture": "CNN-LSTM-Attention", "sequence": b3.SEQUENCE,
                        "horizon": b3.HORIZON, "inputFeatures": 12, "features": FEATURE_NAMES,
                        "mean": mean.tolist(), "scale": scale.tolist(), "targetScale": target_scale.tolist(),
                        "blendByHorizon": [1.] * b3.HORIZON, "members": model_members,
                        "ensembleRule": "equal mean standardized network output, then multiply targetScale",
                        "residualLower": lower.tolist(), "residualUpper": upper.tolist(),
                        "intervalCalibration": "calibration residuals of actual ensemble output, not averaged member intervals",
                        "validationEligible": bool(eligible), "deploymentStatus": "experiment only; not deployed"}
        signal.save(OUT / "models" / f"b3-{interval}-candidate.json", model_export)
        # A concrete raw-input/feature/network-output fixture for future JS port.
        first = int(np.flatnonzero(groups[test["symbol"][0]][:, 0] == test["time"][0])[0])
        fixture_candles = groups[test["symbol"][0]][first-96:first+1]
        fixture_outputs = [b3.predict(networks[n], test["x"][:1])[0] for n in chosen["members"]]
        signal.save(OUT / "parity" / f"{interval}.json", {
            "symbol": test["symbol"][0], "origin": b3.iso(test["time"][0]),
            "candles": [{"time": int(c[0] / 1000), "open": c[1], "high": c[2], "low": c[3],
                         "close": c[4], "volume": c[5], "tradeCount": c[6], "takerBuyBaseVolume": c[7]} for c in fixture_candles],
            "rawFeatures": test["raw"][0].tolist(), "standardizedInput": test["x"][0].tolist(),
            "memberNetworkOutputs": [p.tolist() for p in fixture_outputs],
            "forecastLogReturns": (np.mean(fixture_outputs, axis=0) * target_scale).tolist()})
        report = {**frozen, "candidates": candidates, "validationBaselines": {
            "flat": signal.assessment(val, np.zeros_like(val["y"]), interval), "trainMean": mean_val},
            "testBaselines": {"flat": signal.assessment(test, np.zeros_like(test["y"]), interval),
                              "trainMean": signal.assessment(test, np.broadcast_to(train_mean, test["y"].shape), interval)},
            "durationSeconds": time.monotonic() - start}
        reports.append(report)
        signal.save(OUT / f"{interval}-report.json", report)
        signal.save(OUT / "summary.json", {"protocol": protocol, "intervals": reports})
        t = chosen["sameHoldoutReevaluation"]["h1"]
        b3.log(f"RESULT {interval} selected={selected} eligible={bool(eligible)} R2={t['returnR2VsZero']:.6f} IC={t['pearsonIC']:.6f} BA={t['balancedAccuracy']:.4f} amp={t['amplitudeStdRatio']:.4f}")
    deployment_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "public/models").glob("b3-*.json")}
    assert deployment_before == deployment_after, "Deployment weights changed"
    signal.save(OUT / "deployment-integrity.json", {"before": deployment_before, "after": deployment_after, "unchanged": True})
    lines = ["# B3 主动买入量与成交活动特征实验", "",
             "固定输入变更：原 8 维价格/成交量特征追加主动买入不平衡、成交笔数变化、成交笔数相对20根均值、每笔成交base量变化，共12维。所有定义、候选和准入门槛在训练前写入 protocol.json。", "",
             "仅使用原缓存且核对 SHA256；排除未闭合bar；逐特征前缀一致性检查；训练、验证、校准和测试按原公共时间边界划分。训练与验证选择均未使用测试标签。以下复评沿用已被观察过的留出段，不是新独立测试。", "",
             "| 周期 | 验证选定候选 | 验证准入 | Val R2 | Val IC | Val BA | 复评 R2 | 复评 IC | 复评 BA | 预测/实际std | 复评MAE | 持平MAE |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        c = r["candidates"][r["selected"]]
        v, t = c["validation"]["h1"], c["sameHoldoutReevaluation"]["h1"]
        lines.append(f"|{r['interval']}|{r['selected']}|{r['eligible']}|{v['returnR2VsZero']:.5f}|{v['pearsonIC']:.5f}|{v['balancedAccuracy']:.4f}|{t['returnR2VsZero']:.5f}|{t['pearsonIC']:.5f}|{t['balancedAccuracy']:.4f}|{t['amplitudeStdRatio']:.4f}|{t['priceMaePct']:.5f}%|{t['flatPriceMaePct']:.5f}%|")
    lines += ["", "R2 为 1 - 预测收益MSE / 零收益MSE。BA 为涨跌两类召回率的平均；IC 为预测/实际收益的 Pearson 相关性。std 比只是诊断指标，不是优化目标或准入门槛。没有人为扩大幅度或添加扰动。", "",
              "models/ 内为实验专用schema，不能直接作为当前前端模型加载。若未来整合，需要12维因果特征、原始kline[8]/kline[9]字段、12输入卷积和可能的多成员推理；parity/ 提供精确参考。成员输出取均值后再乘训练targetScale，误差区间在校准段对集成输出重新估计。", "",
              "4份线上权重前后SHA256不变，见deployment-integrity.json。此轮之后停止调参；验证筛选通过不等于已证实稳定泛化。", "",
              "```powershell", ".\\.venv-training\\Scripts\\python.exe scripts/experiment-b3-orderflow.py --threads 6 --epochs 24", "```"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

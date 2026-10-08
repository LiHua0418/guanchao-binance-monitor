"""Reproducible, time-separated B3 training and portable JSON weight export.

Run: .venv-training/Scripts/python.exe scripts/train-b3.py
The test partition is evaluated once, after model/blend selection and calibration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import requests
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "training"
MODELS = ROOT / "public" / "models"
REPORTS = ROOT / "reports"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
JOBS = {"15m": 8000, "1h": 8000, "4h": 6500, "1d": 3500}
INTERVAL_MS = {"15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}
FEATURES = ["logReturn1", "logReturn3", "logReturn12", "rangeRatio", "bodyRatio",
            "logVolumeChange", "sma20Distance", "sma50Distance"]
SEQUENCE, HORIZON = 48, 12
SEED = 20261008
API = "https://data-api.binance.vision/api/v3"


def log(message):
    print(message, flush=True)


def iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def get_json(path, params=None):
    for attempt in range(5):
        try:
            response = requests.get(f"{API}/{path}", params=params, timeout=40)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(1.5 ** attempt)


def fetch_history(symbol, interval, bars, cutoff):
    last_open = cutoff // INTERVAL_MS[interval] * INTERVAL_MS[interval] - INTERVAL_MS[interval]
    path = CACHE / f"{symbol}-{interval}-{bars}-{last_open}.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows, end = [], last_open + INTERVAL_MS[interval] - 1
        while len(rows) < bars:
            batch = get_json("klines", {"symbol": symbol, "interval": interval,
                             "limit": min(1000, bars - len(rows)), "endTime": end})
            if not batch:
                break
            rows = batch + rows
            end = int(batch[0][0]) - 1
            if len(batch) < min(1000, bars - len(rows) + len(batch)):
                break
        rows = sorted({int(r[0]): r for r in rows if int(r[6]) < cutoff}.values(), key=lambda r: int(r[0]))[-bars:]
        write_json(path, rows)
    arr = np.array([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows], dtype=np.float64)
    if len(arr) < 600 or not np.isfinite(arr).all() or np.any(arr[:, 1:5] <= 0):
        raise ValueError(f"Invalid/insufficient candles: {symbol} {interval}")
    if not np.all(np.diff(arr[:, 0]) == INTERVAL_MS[interval]):
        raise ValueError(f"Missing or duplicated candles: {symbol} {interval}")
    log(f"DATA {symbol} {interval}: {len(arr)} closed candles {iso(arr[0,0])} .. {iso(arr[-1,0])}")
    return arr, hashlib.sha256(path.read_bytes()).hexdigest()


def features(candles):
    o, h, l, c, v = (candles[:, i] for i in range(1, 6))
    x = np.zeros((len(c), 8), dtype=np.float64)
    for column, lag in enumerate((1, 3, 12)):
        x[lag:, column] = np.log(c[lag:] / c[:-lag])
    x[:, 3] = (h - l) / c
    x[:, 4] = (c - o) / o
    x[1:, 5] = np.diff(np.log1p(v))
    for column, window in ((6, 20), (7, 50)):
        sums = np.cumsum(np.concatenate(([0.0], c)))
        avg = (sums[window:] - sums[:-window]) / window
        x[window - 1:, column] = c[window - 1:] / avg - 1
    return x


def prepare(groups, interval):
    """Shared calendar boundaries across symbols; 12 origin bars embargoed.

    No target crosses a partition boundary. Training feature statistics use only
    feature rows from the training partition (not repeated overlapping windows).
    All target anchors have at least 50 + 48 - 1 available feature/history bars.
    """
    common_start = max(c[96, 0] for c in groups.values())
    common_end = min(c[-1, 0] for c in groups.values()) + INTERVAL_MS[interval]
    span_bars = int((common_end - common_start) / INTERVAL_MS[interval])
    boundaries = [int(common_start + int(span_bars * ratio) * INTERVAL_MS[interval]) for ratio in (.65, .78, .89)]
    cutoffs = [-np.inf, *boundaries, np.inf]
    partitions = {name: {"x": [], "y": [], "time": [], "symbol": [], "close": []}
                  for name in ("train", "validation", "calibration", "test")}
    train_rows = []
    for symbol, candles in groups.items():
        x = features(candles)
        train_rows.append(x[(candles[:, 0] < boundaries[0]) & (np.arange(len(x)) >= 49)])
        for anchor in range(96, len(candles) - HORIZON):
            timestamp = candles[anchor, 0]
            target_end = candles[anchor + HORIZON, 0]
            part = next((name for i, name in enumerate(partitions)
                         if timestamp >= cutoffs[i] and target_end < cutoffs[i + 1]), None)
            if part is None:
                continue
            p = partitions[part]
            p["x"].append(x[anchor - SEQUENCE + 1:anchor + 1])
            p["y"].append(np.log(candles[anchor + 1:anchor + HORIZON + 1, 4] / candles[anchor, 4]))
            p["time"].append(timestamp)
            p["symbol"].append(symbol)
            p["close"].append(candles[anchor, 4])
    train_rows = np.concatenate(train_rows)
    mean = train_rows.mean(axis=0)
    scale = np.maximum(train_rows.std(axis=0), 1e-6)
    for name, p in partitions.items():
        p["raw"] = np.asarray(p.pop("x"), dtype=np.float64)
        p["x"] = torch.from_numpy(np.clip((p["raw"] - mean) / scale, -8, 8).astype(np.float32))
        p["y"] = np.asarray(p["y"], dtype=np.float64)
        p["time"] = np.asarray(p["time"], dtype=np.int64)
        p["close"] = np.asarray(p["close"], dtype=np.float64)
        if len(p["y"]) < 100:
            raise ValueError(f"Too few {name} samples ({len(p['y'])})")
    target_scale = np.maximum(partitions["train"]["y"].std(axis=0), 1e-5)
    splits = {name: {"samples": len(p["y"]), "firstOrigin": iso(p["time"].min()),
                    "lastOrigin": iso(p["time"].max()), "lastTarget": iso(p["time"].max() + HORIZON * INTERVAL_MS[interval])}
              for name, p in partitions.items()}
    splits["boundaries"] = [iso(b) for b in boundaries]
    splits["embargoBars"] = HORIZON
    splits["selection"] = "train weights; validation early stopping and blend; calibration residual quantiles; test evaluated only after freeze"
    return partitions, mean, scale, target_scale, splits


class B3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(8, 16, 3)
        self.lstm = nn.LSTM(16, 24, batch_first=True)
        self.attention = nn.Linear(24, 1)
        self.output = nn.Linear(48, HORIZON)

    def forward(self, x):
        z = F.relu(self.conv(F.pad(x.transpose(1, 2), (2, 0)))).transpose(1, 2)
        hidden, _ = self.lstm(z)
        attention = torch.softmax(self.attention(hidden), dim=1)
        context = (attention * hidden).sum(dim=1)
        return self.output(torch.cat((hidden[:, -1], context), dim=1))


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(batch).cpu().numpy() for batch in x.split(512)], axis=0).astype(np.float64)


def price_errors(y, pred):
    return np.abs(np.expm1(np.clip(pred - y, -20, 20))) * 100


def fit(partitions, target_scale, max_epochs, interval):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = B3()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    train = partitions["train"]
    validation = partitions["validation"]
    target = torch.from_numpy((train["y"] / target_scale).astype(np.float32))
    baseline_validation = float(price_errors(validation["y"], np.zeros_like(validation["y"])).mean())
    best_score, best_state, best_epoch, best_blend, wait = float("inf"), None, 0, 0.0, 0
    history = []
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=generator)
        total_loss = 0.0
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            output = model(train["x"][indices])
            loss = F.smooth_l1_loss(output, target[indices], beta=0.5)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        val_raw = predict(model, validation["x"]) * target_scale
        # Select checkpoint using nonzero candidates so a flat validation baseline
        # does not force stopping before the network has had a chance to learn.
        candidates = [(float(price_errors(validation["y"], val_raw * blend).mean()), blend)
                      for blend in (.25, .5, .75, 1.0)]
        score, blend = min(candidates)
        scheduler.step(score)
        history.append({"epoch": epoch, "trainLoss": total_loss / len(target), "validationMaePct": score,
                        "validationBlend": blend, "baselineValidationMaePct": baseline_validation})
        if score < best_score - 1e-7:
            best_score, best_epoch, best_blend = score, epoch, blend
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        log(f"TRAIN {interval} epoch={epoch} loss={total_loss/len(target):.5f} valMAE={score:.5f}% baseline={baseline_validation:.5f}% blend={blend} wait={wait}")
        if wait >= 8:
            break
    model.load_state_dict(best_state)
    if baseline_validation <= best_score:
        best_blend = 0.0
    return model, best_blend, best_epoch, history


def evaluate(y, pred, lower, upper):
    error = price_errors(y, pred)
    baseline = price_errors(y, np.zeros_like(y))
    hit = np.sign(y) == np.sign(pred)
    coverage = (y >= pred + lower) & (y <= pred + upper)
    return {"testSamples": len(y), "modelMaePct": float(error.mean()), "baselineMaePct": float(baseline.mean()),
            "directionHitRate": float(hit.mean()), "coverage80": float(coverage.mean()),
            "directionPredictionRate": float((np.abs(pred) > 1e-12).mean()),
            "byHorizon": [{"horizon": i + 1, "maePct": float(error[:, i].mean()),
                           "baselineMaePct": float(baseline[:, i].mean()), "hitRate": float(hit[:, i].mean()),
                           "coverage80": float(coverage[:, i].mean())} for i in range(HORIZON)]}


def weights(model):
    state = model.state_dict()
    def get(name):
        return state[name].cpu().numpy().astype(np.float64).tolist()
    return {"convWeight": get("conv.weight"), "convBias": get("conv.bias"),
            "lstmWeightIH": get("lstm.weight_ih_l0"), "lstmWeightHH": get("lstm.weight_hh_l0"),
            "lstmBiasIH": get("lstm.bias_ih_l0"), "lstmBiasHH": get("lstm.bias_hh_l0"),
            "attentionWeight": get("attention.weight")[0], "attentionBias": get("attention.bias")[0],
            "outputWeight": get("output.weight"), "outputBias": get("output.bias")}


def train_one(interval, groups, hashes, args, cutoff):
    started = time.monotonic()
    parts, mean, scale, target_scale, splits = prepare(groups, interval)
    log(f"SPLITS {interval} " + json.dumps({k: len(v['y']) for k, v in parts.items()}))
    model, blend, epoch, history = fit(parts, target_scale, args.epochs, interval)
    calibration = parts["calibration"]
    cal_pred = predict(model, calibration["x"]) * target_scale * blend
    residual = calibration["y"] - cal_pred
    lower, upper = np.quantile(residual, [.1, .9], axis=0)
    test = parts["test"]
    raw_prediction = predict(model, test["x"])
    test_pred = raw_prediction * target_scale * blend
    metrics = evaluate(test["y"], test_pred, lower, upper)
    per_symbol = {}
    for symbol in SYMBOLS:
        mask = np.array(test["symbol"]) == symbol
        per_symbol[symbol] = evaluate(test["y"][mask], test_pred[mask], lower, upper)
    # Also show every twelfth origin: adjacent rolling predictions share targets,
    # so the headline sample count must not imply statistical independence.
    sparse_indices = np.concatenate([np.flatnonzero(np.array(test["symbol"]) == s)[::HORIZON] for s in SYMBOLS])
    sparse_metrics = evaluate(test["y"][sparse_indices], test_pred[sparse_indices], lower, upper)
    bundle = {"schemaVersion": 1, "name": "guanchao-b3", "architecture": "CNN-LSTM-Attention", "interval": interval,
              "sequence": SEQUENCE, "horizon": HORIZON, "features": FEATURES, "mean": mean.tolist(),
              "scale": scale.tolist(), "targetScale": target_scale.tolist(), "blend": blend, "weights": weights(model),
              "residualLower": lower.tolist(), "residualUpper": upper.tolist(),
              "trainedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "dataCutoff": iso(calibration["time"].max() + (HORIZON + 1) * INTERVAL_MS[interval]),
              "dataEnd": iso(min(c[-1, 0] + INTERVAL_MS[interval] for c in groups.values())),
              "trainedSymbols": SYMBOLS, "trainedBars": sum(len(c) for c in groups.values()),
              "epochs": epoch, "metrics": metrics, "splits": splits,
              "seed": SEED, "trainingFeatureRowsOnly": True,
              "metricDefinition": "MAE = mean(abs(exp(predictedLogReturn - realizedLogReturn) - 1))*100, all 12 horizons; direction = sign equality, flat is abstention/miss for nonzero realized return"}
    write_json(MODELS / f"b3-{interval}.json", bundle)
    symbol = test["symbol"][0]
    anchor = int(np.flatnonzero(groups[symbol][:, 0] == test["time"][0])[0])
    candles = groups[symbol][anchor - 96:anchor + 1]
    parity = {"interval": interval, "symbol": symbol, "origin": iso(test["time"][0]),
              "candles": [{"time": int(c[0] / 1000), "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]} for c in candles],
              "rawFeatures": test["raw"][0].tolist(), "standardizedInput": test["x"][0].numpy().astype(np.float64).tolist(),
              "networkOutput": raw_prediction[0].tolist(), "forecastLogReturns": test_pred[0].tolist(),
              "forecastPrices": (test["close"][0] * np.exp(test_pred[0])).tolist(),
              "lowerPrices": (test["close"][0] * np.exp(test_pred[0] + lower)).tolist(),
              "upperPrices": (test["close"][0] * np.exp(test_pred[0] + upper)).tolist()}
    report = {"interval": interval, "metrics": metrics, "perSymbol": per_symbol, "nonoverlappingOriginMetrics": sparse_metrics,
              "blend": blend, "selectedEpoch": epoch, "epochsRun": len(history), "trainingHistory": history,
              "splits": splits, "durationSeconds": time.monotonic() - started,
              "data": {s: {"bars": len(c), "firstOpen": iso(c[0, 0]), "lastOpen": iso(c[-1, 0]), "sha256": hashes[s]} for s, c in groups.items()}}
    log(f"RESULT {interval}: blend={blend} testMAE={metrics['modelMaePct']:.5f}% baseline={metrics['baselineMaePct']:.5f}% direction={metrics['directionHitRate']:.3f} coverage={metrics['coverage80']:.3f} duration={report['durationSeconds']:.1f}s")
    return report, parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", nargs="+", choices=list(JOBS), default=list(JOBS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cutoff-ms", type=int, help="Freeze the download cutoff (UTC Unix milliseconds).")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    cutoff = args.cutoff_ms or int(get_json("time")["serverTime"])
    log(f"START seed={SEED} cutoff={iso(cutoff)} torch={torch.__version__} threads={args.threads}")
    groups, hashes = {i: {} for i in args.intervals}, {i: {} for i in args.intervals}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_history, symbol, interval, JOBS[interval], cutoff): (symbol, interval)
                   for interval in args.intervals for symbol in SYMBOLS}
        for future in as_completed(futures):
            symbol, interval = futures[future]
            groups[interval][symbol], hashes[interval][symbol] = future.result()
    reports, parities = [], []
    for interval in args.intervals:
        report, parity = train_one(interval, {s: groups[interval][s] for s in SYMBOLS}, hashes[interval], args, cutoff)
        reports.append(report)
        parities.append(parity)
        write_json(REPORTS / "b3-training-report.json", {"name": "guanchao-b3", "seed": SEED,
                   "cutoff": iso(cutoff), "source": API + "/klines", "pythonTorch": str(torch.__version__), "intervals": reports})
        write_json(REPORTS / "b3-parity.json", {"schemaVersion": 1, "tolerance": 1e-5, "cases": parities})
    lines = ["# guanchao-b3 独立测试报告", "", f"数据冻结时间：{iso(cutoff)}；随机种子：{SEED}。", "",
             "网络：48 根历史 × 8 个特征 → 因果 Conv1D(16,k=3) → LSTM(24) → Attention → 12 个直接累计对数收益率。", "",
             "| 周期 | 测试起点数 | B3 MAE | 持平基线 MAE | 相对变化 | 方向命中 | 80%区间覆盖 | 验证选择 blend |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        m = r["metrics"]
        change = (m["modelMaePct"] / m["baselineMaePct"] - 1) * 100
        lines.append(f"| {r['interval']} | {m['testSamples']} | {m['modelMaePct']:.4f}% | {m['baselineMaePct']:.4f}% | {change:+.2f}% | {m['directionHitRate']*100:.1f}% | {m['coverage80']*100:.1f}% | {r['blend']} |")
    lines += ["", "## 实验约束与解读", "",
              "- Binance 公开现货 OHLCV，BTCUSDT、ETHUSDT、BNBUSDT、SOLUSDT；排除未闭合 K 线，校验时间连续性。各周期数据范围及 SHA256 见 JSON。SOL 上币历史不足 3500 根日线时使用全部可得日线。",
              "- 同一周期所有币种使用相同时间边界：共同可用时间范围的 65% / 78% / 89% 划分训练、验证、校准、测试。其他币种更早的记录只用于训练。边界前 12 根预测起点剔除，所有标签严格留在所属区间。",
              "- 特征 mean/std 和 12 个目标标准差只从训练数据计算；无目标均值；标准化 clip[-8,8]。验证集选择 checkpoint、早停及 blend ∈ {0,0.25,0.5,0.75,1}。",
              "- 独立校准集的逐 horizon 残差 10%/90% 分位生成参考区间；时间序列会发生分布变化，80% 是名义覆盖，不是未来覆盖承诺。测试集只在模型与区间冻结后评估。",
              "- MAE = mean(abs(exp(预测累计对数收益 - 实际累计对数收益)-1))×100，汇总所有 12 个 horizon。基线假设所有未来收盘价保持当前收盘价；相对变化为负才表示误差下降。",
              "- 方向命中为预测和实际累计收益符号相同。blend=0 表示验证集选择持平回退，非零实际收益对应方向命中为 0，不能解读为有方向预测能力。",
              "- 相邻预测窗口共享标签，样本数不等同于独立样本数；JSON 另含每币每 12 根选一个起点的稀疏评估及逐币/逐 horizon 结果。未按测试结果重新调参。",
              "- 当前仓库旧 LSTM 的训练权重为空，无法做相同权重的旧版 A/B 比较；报告以无需拟合的持平价格为基线，不宣称已经超过一个不存在的已训练模型。",
              "- 本实验没有交易成本、滑点、交易策略或实盘收益测试；价格误差和方向命中不等于可交易收益。四个周期分别训练，测试时间跨度不同，应分别解读。",
              "", "## 重现", "", "```powershell", "python -m venv .venv-training",
              ".\\.venv-training\\Scripts\\python.exe -m pip install -r scripts/requirements-training.txt",
              f".\\.venv-training\\Scripts\\python.exe scripts/train-b3.py --cutoff-ms {cutoff} --epochs {args.epochs} --threads {args.threads}",
              "```", "", "权重：public/models/b3-{interval}.json；跨语言一致性样例：reports/b3-parity.json。"]
    (REPORTS / "b3-training-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

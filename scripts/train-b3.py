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
import os
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
TRAINING_OBJECTIVE = "next-bar-priority"
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
    device = next(model.parameters()).device
    with torch.no_grad():
        return np.concatenate([model(batch.to(device)).cpu().numpy() for batch in x.split(512)], axis=0).astype(np.float64)


def price_errors(y, pred):
    return np.abs(np.expm1(np.clip(pred - y, -20, 20))) * 100


def fit(partitions, target_scale, max_epochs, interval, device="cpu"):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = B3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.015)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    train = partitions["train"]
    validation = partitions["validation"]
    train_x = train["x"].to(device)
    target = torch.from_numpy((train["y"] / target_scale).astype(np.float32)).to(device)
    baseline_by_horizon = price_errors(validation["y"], np.zeros_like(validation["y"])).mean(axis=0)
    baseline_validation = float(baseline_by_horizon.mean())
    loss_weights = torch.tensor([0.6] + [0.4 / (HORIZON - 1)] * (HORIZON - 1), dtype=torch.float32, device=device)
    blend_options = np.array([0.0, .25, .5, .75, 1.0])
    best_rank, best_state, best_epoch, best_blend, wait = (float("inf"), float("inf")), None, 0, None, 0
    history = []
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=generator)
        total_loss = 0.0
        for indices in order.split(256):
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(train_x[indices])
            loss = (F.smooth_l1_loss(output, target[indices], beta=0.5, reduction="none") * loss_weights).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        val_raw = predict(model, validation["x"]) * target_scale
        # Every horizon independently picks shrinkage on validation only.
        # Next-bar validation MAE strictly takes priority; when equal (notably
        # flat fallback), normalized h2..h12 error breaks the tie.
        candidate_errors = np.stack([price_errors(validation["y"], val_raw * blend).mean(axis=0)
                                     for blend in blend_options])
        best_indices = candidate_errors.argmin(axis=0)
        blend = blend_options[best_indices]
        horizon_error = candidate_errors[best_indices, np.arange(HORIZON)]
        rank = (float(horizon_error[0]), float(np.mean(horizon_error[1:] / baseline_by_horizon[1:])))
        score = float(horizon_error.mean())
        scheduler.step(rank[0])
        history.append({"epoch": epoch, "trainLoss": total_loss / len(target), "validationMaePct": score,
                        "validationNextBarMaePct": rank[0], "validationLaterNormalizedMae": rank[1],
                        "validationBlendByHorizon": blend.tolist(), "baselineValidationMaePct": baseline_validation,
                        "baselineValidationNextBarMaePct": float(baseline_by_horizon[0])})
        improved = rank[0] < best_rank[0] - 1e-9 or (abs(rank[0] - best_rank[0]) <= 1e-9 and rank[1] < best_rank[1] - 1e-7)
        if improved:
            best_rank, best_epoch, best_blend = rank, epoch, blend.copy()
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        log(f"TRAIN {interval} epoch={epoch} loss={total_loss/len(target):.5f} nextMAE={rank[0]:.5f}% nextBaseline={baseline_by_horizon[0]:.5f}% nextBlend={blend[0]} wait={wait}")
        if wait >= 8:
            break
    model.load_state_dict(best_state)
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
    model, blend, epoch, history = fit(parts, target_scale, args.epochs, interval, args.device)
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
              "scale": scale.tolist(), "targetScale": target_scale.tolist(), "blend": float(blend[0]),
              "blendByHorizon": blend.tolist(), "trainingObjective": TRAINING_OBJECTIVE, "weights": weights(model),
              "residualLower": lower.tolist(), "residualUpper": upper.tolist(),
              "trainedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "dataCutoff": iso(calibration["time"].max() + (HORIZON + 1) * INTERVAL_MS[interval]),
              "dataEnd": iso(min(c[-1, 0] + INTERVAL_MS[interval] for c in groups.values())),
              "trainedSymbols": SYMBOLS, "trainedBars": sum(len(c) for c in groups.values()),
              "epochs": epoch, "metrics": metrics, "splits": splits,
              "trainingCompute": {"device": args.device, "cpuThreads": args.threads, "torchVersion": str(torch.__version__)},
              "selection": "loss: normalized Huber h1=0.6, h2..12=0.4/11; checkpoint: validation h1 price MAE first, ties by normalized later-horizon MAE; independent validation shrink per horizon",
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
    previous_path = REPORTS / "experiments" / "multi-horizon-v1" / "models" / f"b3-{interval}.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
    report = {"interval": interval, "trainingObjective": TRAINING_OBJECTIVE,
              "metrics": metrics, "nextBarMetrics": metrics["byHorizon"][0],
              "previousVersionNextBarMetrics": previous["metrics"]["byHorizon"][0] if previous else None,
              "perSymbol": per_symbol, "nonoverlappingOriginMetrics": sparse_metrics,
              "blend": float(blend[0]), "blendByHorizon": blend.tolist(), "selectedEpoch": epoch, "epochsRun": len(history), "trainingHistory": history,
              "splits": splits, "durationSeconds": time.monotonic() - started,
              "compute": {"device": args.device, "cpuThreads": args.threads},
              "data": {s: {"bars": len(c), "firstOpen": iso(c[0, 0]), "lastOpen": iso(c[-1, 0]), "sha256": hashes[s]} for s, c in groups.items()}}
    h1 = metrics["byHorizon"][0]
    log(f"RESULT {interval}: blend={blend.tolist()} nextMAE={h1['maePct']:.5f}% nextBaseline={h1['baselineMaePct']:.5f}% nextDirection={h1['hitRate']:.3f} allMAE={metrics['modelMaePct']:.5f}% duration={report['durationSeconds']:.1f}s")
    return report, parity


def resolve_runtime(device="auto", threads="auto"):
    """Configure compute explicitly; this does not read or modify model weights."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA was requested but is unavailable in PyTorch {torch.__version__} (torch.version.cuda={torch.version.cuda}). Install a compatible CUDA-enabled PyTorch build or use --device cpu. No training was started.")
    if str(threads) == "auto":
        resolved_threads = min(8, os.cpu_count() or 1)
        source = "default min(8, logical processors)"
        benchmark_path = ROOT / "reports" / "b3-compute-benchmark.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            recommendation = benchmark.get("recommendedCpuThreads")
            hardware = benchmark.get("hardware", {})
            if (isinstance(recommendation, int) and 0 < recommendation <= (os.cpu_count() or 1)
                    and hardware.get("logicalProcessors") == os.cpu_count()
                    and hardware.get("pytorch") == str(torch.__version__)):
                resolved_threads = recommendation
                source = "reports/b3-compute-benchmark.json measured recommendation"
    else:
        try:
            resolved_threads = int(threads)
        except (TypeError, ValueError) as error:
            raise ValueError("--threads must be 'auto' or a positive integer") from error
        if resolved_threads < 1 or resolved_threads > (os.cpu_count() or 1):
            raise ValueError(f"--threads must be between 1 and {os.cpu_count() or 1}")
        source = "explicit --threads"
    torch.set_num_threads(resolved_threads)
    torch.set_num_interop_threads(2)
    return device, resolved_threads, source


def dry_run(device):
    """Exercise the actual fit/predict/export path without fetching or writing."""
    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(64, SEQUENCE, 8, generator=generator)
    target = torch.randn(64, HORIZON, generator=generator).numpy().astype(np.float64)
    parts = {"train": {"x": x[:32], "y": target[:32]},
             "validation": {"x": x[32:], "y": target[32:]}}
    model, blend, epoch, history = fit(parts, np.ones(HORIZON), 1, "dry-run", device)
    output = predict(model, x[32:])
    assert output.shape == (32, HORIZON) and np.isfinite(output).all()
    assert blend.shape == (HORIZON,) and epoch == 1 and history[0]["trainLoss"] > 0
    json.dumps(weights(model), allow_nan=False)
    log(f"DRY RUN PASS device={device} intra_op_threads={torch.get_num_threads()} inter_op_threads={torch.get_num_interop_threads()} loss={history[0]['trainLoss']:.6f} output={output.shape}; no download or artifact writes")


def main():
    global MODELS, REPORTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", nargs="+", choices=list(JOBS), default=list(JOBS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", default="auto", help="CPU intra-op threads: auto uses the compatible local benchmark recommendation, otherwise up to 8; or give an integer.")
    parser.add_argument("--dry-run", action="store_true", help="One synthetic training/inference/export-validation step; never downloads or writes artifacts.")
    parser.add_argument("--output-dir", type=Path, help="Write models/ and reports/ under this directory instead of the deployed artifact locations.")
    parser.add_argument("--cutoff-ms", type=int, help="Freeze the download cutoff (UTC Unix milliseconds).")
    args = parser.parse_args()
    try:
        args.device, args.threads, thread_source = resolve_runtime(args.device, args.threads)
    except ValueError as error:
        parser.error(str(error))
    log(f"COMPUTE device={args.device} intra_op_threads={args.threads} inter_op_threads=2 source={thread_source}")
    if args.dry_run:
        dry_run(args.device)
        return
    if args.output_dir:
        MODELS = args.output_dir.resolve() / "models"
        REPORTS = args.output_dir.resolve() / "reports"
    cutoff = args.cutoff_ms or int(get_json("time")["serverTime"])
    log(f"START seed={SEED} cutoff={iso(cutoff)} torch={torch.__version__} threads={args.threads} device={args.device}")
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
                   "trainingObjective": TRAINING_OBJECTIVE, "evaluationStatus": "same holdout period reevaluated after user changed target to next bar; not a fresh unseen test",
                   "cutoff": iso(cutoff), "source": API + "/klines", "pythonTorch": str(torch.__version__),
                   "compute": {"device": args.device, "cpuThreads": args.threads, "threadSelection": thread_source}, "intervals": reports})
        write_json(REPORTS / "b3-parity.json", {"schemaVersion": 1, "tolerance": 1e-5, "cases": parities})
    lines = ["# guanchao-b3 下一根优先训练评估报告", "", f"数据冻结时间：{iso(cutoff)}；随机种子：{SEED}。", "",
             "用户将主要业务目标明确为下一根 K 线，因此本版调整训练目标和验证选择标准。以下是在第一版相同留出时段上的复评，不是新增、从未查看过的独立测试。训练及参数选择仍只使用训练/验证段。", "",
             "网络：48 根历史 × 8 个特征 → 因果 Conv1D(16,k=3) → LSTM(24) → Attention → 12 个直接累计对数收益率。", "",
             "## 主要目标：下一根 K 线", "",
             "| 周期 | 测试起点数 | 本版 h1 MAE | 第一版 h1 MAE | 持平 h1 MAE | h1 方向命中 | h1 区间覆盖 | h1 blend |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        m = r["nextBarMetrics"]
        old = r["previousVersionNextBarMetrics"]
        prior = f"{old['maePct']:.5f}%" if old else "—"
        direction = f"{m['hitRate']*100:.1f}%" if r['blend'] else "持平回退"
        lines.append(f"| {r['interval']} | {r['metrics']['testSamples']} | {m['maePct']:.5f}% | {prior} | {m['baselineMaePct']:.5f}% | {direction} | {m['coverage80']*100:.1f}% | {r['blend']} |")
    lines += ["", "## 辅助目标：未来 12 根整体", "",
              "| 周期 | B3 全 horizon MAE | 持平 MAE | 相对变化 | 各 horizon blend |", "|---|---:|---:|---:|---|"]
    for r in reports:
        m = r["metrics"]
        change = (m["modelMaePct"] / m["baselineMaePct"] - 1) * 100
        lines.append(f"| {r['interval']} | {m['modelMaePct']:.4f}% | {m['baselineMaePct']:.4f}% | {change:+.2f}% | {r['blendByHorizon']} |")
    lines += ["", "## 实验约束与解读", "",
              "- Binance 公开现货 OHLCV，BTCUSDT、ETHUSDT、BNBUSDT、SOLUSDT；排除未闭合 K 线，校验时间连续性。各周期数据范围及 SHA256 见 JSON。SOL 上币历史不足 3500 根日线时使用全部可得日线。",
              "- 同一周期所有币种使用相同时间边界：共同可用时间范围的 65% / 78% / 89% 划分训练、验证、校准、测试。其他币种更早的记录只用于训练。边界前 12 根预测起点剔除，所有标签严格留在所属区间。",
              "- 特征 mean/std 和 12 个目标标准差只从训练数据计算；无目标均值；标准化 clip[-8,8]。训练损失为标准化 Huber：h1 权重 0.6，其余11个 horizon 各 0.4/11。",
              "- 验证集分别为每个horizon选择 blend ∈ {0,0.25,0.5,0.75,1}；checkpoint 严格优先最小化验证 h1 价格 MAE，h1相同时才按其余horizon相对各自持平基线的平均MAE择优。没有根据测试结果挑选轮次或混合系数。",
              "- 独立校准集的逐 horizon 残差 10%/90% 分位生成参考区间；时间序列会发生分布变化，80% 是名义覆盖，不是未来覆盖承诺。测试集只在模型与区间冻结后评估。",
              "- MAE = mean(abs(exp(预测累计对数收益 - 实际累计对数收益)-1))×100，汇总所有 12 个 horizon。基线假设所有未来收盘价保持当前收盘价；相对变化为负才表示误差下降。",
              "- 方向命中为预测和实际累计收益符号相同。某horizon的blend=0表示该步持平回退，不能解读为有方向预测能力；bundle.blend仅是h1系数，完整系数在blendByHorizon。",
              "- 相邻预测窗口共享标签，样本数不等同于独立样本数；JSON 另含每币每 12 根选一个起点的稀疏评估及逐币/逐 horizon 结果。未按测试结果重新调参。",
              "- 当前仓库旧 LSTM 的训练权重为空，无法做相同权重的旧版 A/B 比较；报告以无需拟合的持平价格为基线，不宣称已经超过一个不存在的已训练模型。",
              "- 原第一版权重和报告已归档到 reports/experiments/multi-horizon-v1；旧架构诊断报告必须结合其模型版本/SHA256解读。历史回放每个bar画一个h1预测点属于显示口径修正，不会凭空增加训练精度。",
              "- 本实验没有交易成本、滑点、交易策略或实盘收益测试；价格误差和方向命中不等于可交易收益。四个周期分别训练，测试时间跨度不同，应分别解读。",
              "", "## 重现", "", "```powershell", "python -m venv .venv-training",
              ".\\.venv-training\\Scripts\\python.exe -m pip install -r scripts/requirements-training.txt",
              f".\\.venv-training\\Scripts\\python.exe scripts/train-b3.py --cutoff-ms {cutoff} --epochs {args.epochs} --device {args.device} --threads {args.threads}",
              "```", "", "权重：public/models/b3-{interval}.json；跨语言一致性样例：reports/b3-parity.json。"]
    (REPORTS / "b3-training-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

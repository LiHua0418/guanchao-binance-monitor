"""Fixed, validation-selected B3 signal-objective experiment; never deploys weights.

Uses only the candle cache frozen by reports/b3-training-report.json. Selection
is completed and persisted before the existing (already inspected) holdout is
reevaluated. Run with .venv-training/Scripts/python.exe.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b3_training", ROOT / "scripts/train-b3.py")
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)
OUT = ROOT / "reports/experiments/signal-objectives"
SEEDS = [20261008, 20261009, 20261010]
OBJECTIVES = ["huber", "mse", "mse-direction"]


def save(path, value):
    b3.write_json(path, value)


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    zero_error = float(np.mean(y ** 2))
    mse = float(np.mean((prediction - y) ** 2))
    actual_std, pred_std = float(y.std()), float(prediction.std())
    pearson = float(np.corrcoef(y, prediction)[0, 1]) if actual_std > 0 and pred_std > 1e-14 else None
    pos, neg = y > 0, y < 0
    recall_pos = float(np.mean(prediction[pos] > 1e-12)) if pos.any() else None
    recall_neg = float(np.mean(prediction[neg] < -1e-12)) if neg.any() else None
    ba = (recall_pos + recall_neg) / 2 if recall_pos is not None and recall_neg is not None else None
    return {"samples": len(y), "returnMse": mse, "zeroReturnMse": zero_error,
            "returnR2VsZero": 1 - mse / zero_error, "pearsonIC": pearson,
            "balancedAccuracy": ba, "directionCoverage": float(np.mean(np.abs(prediction) > 1e-12)),
            "directionAccuracy": float(np.mean(np.sign(y) == np.sign(prediction))),
            "priceMaePct": float(b3.price_errors(y, prediction).mean()),
            "flatPriceMaePct": float(b3.price_errors(y, np.zeros_like(y)).mean()),
            "predictedReturnStd": pred_std, "realizedReturnStd": actual_std,
            "amplitudeStdRatio": pred_std / actual_std,
            "meanPredictedReturn": float(prediction.mean()), "meanActualReturn": float(y.mean())}


def score(y, prediction, target_scale):
    per_horizon = np.mean(((prediction - y) / target_scale) ** 2, axis=0)
    return float(.8 * per_horizon[0] + .2 * per_horizon[1:].mean())


def assessment(part, prediction, interval):
    unique_times = np.unique(part["time"])
    blocks = []
    # h1 targets must fall in the same calendar block as their origin. The
    # last origin of each block is excluded, including the last block.
    for index, times in enumerate(np.array_split(unique_times, 3)):
        end_exclusive = times[-1] + b3.INTERVAL_MS[interval]
        mask = ((part["time"] >= times[0]) &
                (part["time"] + b3.INTERVAL_MS[interval] < end_exclusive))
        blocks.append({"block": index + 1, "firstOrigin": b3.iso(times[0]),
                       "endExclusive": b3.iso(end_exclusive),
                       "h1": metrics(part["y"][mask, 0], prediction[mask, 0])})
    per_symbol = {}
    for symbol in b3.SYMBOLS:
        mask = np.array(part["symbol"]) == symbol
        per_symbol[symbol] = metrics(part["y"][mask, 0], prediction[mask, 0])
    return {"h1": metrics(part["y"][:, 0], prediction[:, 0]),
            "byHorizon": [metrics(part["y"][:, i], prediction[:, i]) for i in range(b3.HORIZON)],
            "timeBlocks": blocks, "perSymbolH1": per_symbol}


def eligibility(assessed, mean_baseline):
    m = assessed["h1"]
    good_blocks = sum(b["h1"]["returnR2VsZero"] > 0 and
                      (b["h1"]["pearsonIC"] or 0) > 0 for b in assessed["timeBlocks"])
    checks = {"positiveReturnSkill": m["returnR2VsZero"] > 0,
              "beatsTrainMeanMse": m["returnMse"] < mean_baseline["h1"]["returnMse"],
              "pearsonAbove002": (m["pearsonIC"] or 0) > .02,
              "balancedAccuracyAboveHalf": (m["balancedAccuracy"] or 0) > .5,
              "priceMaeWithinHalfPercentOfFlat": m["priceMaePct"] <= m["flatPriceMaePct"] * 1.005,
              "twoOfThreeTimeBlocksPositive": good_blocks >= 2}
    return {"eligible": all(checks.values()), "checks": checks,
            "positiveTimeBlocks": good_blocks}


def fit(parts, target_scale, objective, seed, epochs, interval):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = b3.B3()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.015)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)
    target = torch.tensor(parts["train"]["y"] / target_scale, dtype=torch.float32)
    x = parts["train"]["x"]
    weights = torch.tensor([.6] + [.4 / 11] * 11)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for indices in torch.randperm(len(target), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x[indices])
            actual = target[indices]
            if objective == "huber":
                errors = F.smooth_l1_loss(pred, actual, beta=.5, reduction="none")
            else:
                errors = (pred - actual) ** 2
            loss = (errors * weights).sum(dim=1).mean()
            if objective == "mse-direction":
                # Unit-temperature logits are the standardized log returns;
                # target_scale comes only from train. No probability/magnitude
                # remapping is applied at inference.
                nonzero = actual[:, 0] != 0
                direction_loss = F.binary_cross_entropy_with_logits(
                    pred[nonzero, 0], (actual[nonzero, 0] > 0).float())
                loss = loss + .1 * direction_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_prediction = b3.predict(model, parts["validation"]["x"]) * target_scale
        val_score = score(parts["validation"]["y"], val_prediction, target_scale)
        scheduler.step(val_score)
        history.append({"epoch": epoch, "trainLoss": float(np.mean(losses)),
                        "validationSelectionScore": val_score})
        if val_score < best_score - 1e-7:
            best_score, best_epoch, wait = val_score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= 6:
            break
    model.load_state_dict(best_state)
    b3.log(f"FIT {interval} {objective} seed={seed} best_epoch={best_epoch} epochs={epoch} score={best_score:.6f}")
    return model, {"seed": seed, "objective": objective, "selectedEpoch": best_epoch,
                   "epochsRun": epoch, "history": history}


def cached_groups(interval, cutoff, expected_hashes):
    groups, hashes = {}, {}
    for symbol in b3.SYMBOLS:
        last_open = cutoff // b3.INTERVAL_MS[interval] * b3.INTERVAL_MS[interval] - b3.INTERVAL_MS[interval]
        path = b3.CACHE / f"{symbol}-{interval}-{b3.JOBS[interval]}-{last_open}.json"
        if not path.exists():
            raise FileNotFoundError(f"Frozen cache required: {path}; this experiment never downloads")
        rows = json.loads(path.read_text(encoding="utf-8"))
        groups[symbol] = np.array([[int(r[0]), *map(float, r[1:6])] for r in rows], dtype=np.float64)
        hashes[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[symbol] != expected_hashes[symbol]:
            raise ValueError(f"Frozen source-report SHA256 mismatch: {path}")
        if not np.isfinite(groups[symbol]).all() or not np.all(np.diff(groups[symbol][:, 0]) == b3.INTERVAL_MS[interval]):
            raise ValueError(f"Invalid cache: {path}")
    return groups, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", nargs="+", choices=list(b3.JOBS), default=["1h", "15m", "4h", "1d"])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    b3.resolve_runtime("cpu", args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    previous_report = json.loads((ROOT / "reports/b3-training-report.json").read_text(encoding="utf-8"))
    cutoff = int(datetime.fromisoformat(previous_report["cutoff"].replace("Z", "+00:00")).timestamp() * 1000)
    deployed_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "public/models").glob("b3-*.json")}
    protocol = {"createdAt": datetime.now(timezone.utc).isoformat(), "cutoffMs": cutoff,
                "objectives": OBJECTIVES, "seeds": SEEDS, "maxEpochs": args.epochs,
                "cpuThreads": args.threads, "architecture": "unchanged CNN-LSTM-Attention 5045 parameters",
                "trainingLoss": "normalized returns; h1 weight .6, later .4/11; Huber beta=.5 or MSE; direction candidate adds .1 BCE h1 with fixed temperature1",
                "checkpointSelection": "validation normalized return MSE: .8*h1+.2*mean(h2..h12); early stop6; no shrinkage or amplitude floor",
                "candidates": "3 individual seeds plus equal-weight ensemble per objective; flat and train-mean baselines",
                "promotionGate": "validation h1 R2vsZero>0; MSE beats train-mean; PearsonIC>.02; balancedAccuracy>.5; priceMAE<=flat*1.005; >=2 of3 time blocks both positive R2vsZero and IC",
                "selection": "among eligible choose minimum validation score; if none, best candidate is diagnostic only and must not be promoted",
                "evaluationStatus": "same historical holdout reevaluation, not fresh unseen data; no test data used in candidate, checkpoint or promotion selection",
                "deployment": "No deployed model file modified. Ensembles require additional JS runtime support; constituent models use existing weight schema.",
                "deploymentHashesBefore": deployed_hashes}
    save(OUT / "protocol.json", protocol)
    (OUT / "protocol.md").write_text("# B3 signal objective experiment protocol\n\n" +
        "Recorded before training. Model selection uses validation only.\n\n" +
        "\n".join(f"- **{key}**: {value}" for key, value in protocol.items()) + "\n", encoding="utf-8")
    reports = []
    for interval in args.intervals:
        started = time.monotonic()
        expected_hashes = next(r["data"] for r in previous_report["intervals"] if r["interval"] == interval)
        groups, hashes = cached_groups(interval, cutoff, {s: expected_hashes[s]["sha256"] for s in b3.SYMBOLS})
        parts, mean, scale, target_scale, splits = b3.prepare(groups, interval)
        mean_return = parts["train"]["y"].mean(axis=0)
        val = parts["validation"]
        mean_val = assessment(val, np.broadcast_to(mean_return, val["y"].shape), interval)
        candidates, networks = {}, {}
        for objective in OBJECTIVES:
            names = []
            for seed in SEEDS:
                name = f"{objective}-seed{seed}"
                model, run = fit(parts, target_scale, objective, seed, args.epochs, interval)
                networks[name] = model
                names.append(name)
                predicted = b3.predict(model, val["x"]) * target_scale
                assessed = assessment(val, predicted, interval)
                candidates[name] = {"members": [name], "run": run,
                                    "validationScore": score(val["y"], predicted, target_scale),
                                    "validation": assessed, "gate": eligibility(assessed, mean_val)}
            name = f"{objective}-ensemble3"
            predicted = np.mean([b3.predict(networks[n], val["x"]) for n in names], axis=0) * target_scale
            assessed = assessment(val, predicted, interval)
            candidates[name] = {"members": names, "validationScore": score(val["y"], predicted, target_scale),
                                "validation": assessed, "gate": eligibility(assessed, mean_val)}
        eligible = [n for n, c in candidates.items() if c["gate"]["eligible"]]
        selected = min(eligible or candidates.keys(), key=lambda n: candidates[n]["validationScore"])
        decision = {"interval": interval, "selected": selected, "eligible": bool(eligible),
                    "eligibleCandidates": eligible, "selectionFrozenAt": datetime.now(timezone.utc).isoformat(),
                    "candidates": candidates, "splits": splits, "dataSha256": hashes}
        # Durable selection record is written BEFORE computing any test output.
        save(OUT / f"{interval}-selection-frozen.json", decision)
        test = parts["test"]
        baseline_test = {"flat": assessment(test, np.zeros_like(test["y"]), interval),
                         "trainMean": assessment(test, np.broadcast_to(mean_return, test["y"].shape), interval)}
        for name, candidate in candidates.items():
            prediction = np.mean([b3.predict(networks[n], test["x"]) for n in candidate["members"]], axis=0) * target_scale
            candidate["sameHoldoutReevaluation"] = assessment(test, prediction, interval)
        chosen = candidates[selected]
        # Export only the validation-selected candidate in the experiment folder.
        members = []
        for name in chosen["members"]:
            model = networks[name]
            cal_pred = b3.predict(model, parts["calibration"]["x"]) * target_scale
            lower, upper = np.quantile(parts["calibration"]["y"] - cal_pred, [.1, .9], axis=0)
            members.append({"schemaVersion": 1, "name": "guanchao-b3-experimental", "interval": interval,
                            "architecture": "CNN-LSTM-Attention", "sequence": b3.SEQUENCE,
                            "horizon": b3.HORIZON, "features": b3.FEATURES, "mean": mean.tolist(),
                            "scale": scale.tolist(), "targetScale": target_scale.tolist(),
                            "blend": 1., "blendByHorizon": [1.] * b3.HORIZON,
                            "weights": b3.weights(model), "residualLower": lower.tolist(),
                            "residualUpper": upper.tolist(), "experimentalCandidate": name})
        save(OUT / "models" / f"b3-{interval}-candidate.json",
             {"selected": selected, "validationEligible": bool(eligible), "members": members,
              "deploymentBlocked": not bool(eligible), "ensembleRule": "mean standardized network outputs, then targetScale"})
        report = {**decision, "candidates": candidates, "validationBaselines": {
            "flat": assessment(val, np.zeros_like(val["y"]), interval), "trainMean": mean_val},
            "testBaselines": baseline_test,
            "trainMajorityDirection": 1 if np.mean(parts["train"]["y"][:, 0] > 0) >= .5 else -1,
            "durationSeconds": time.monotonic() - started}
        save(OUT / f"{interval}-report.json", report)
        reports.append(report)
        m = chosen["sameHoldoutReevaluation"]["h1"]
        b3.log(f"RESULT {interval} selected={selected} eligible={bool(eligible)} testR2={m['returnR2VsZero']:.6f} IC={m['pearsonIC']} BA={m['balancedAccuracy']:.4f} amp={m['amplitudeStdRatio']:.4f}")
        save(OUT / "summary.json", {"protocol": protocol, "intervals": reports})
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "public/models").glob("b3-*.json")}
    assert deployed_hashes == after, "Deployed models changed during the experiment"
    lines = ["# B3 return-signal objective experiment", "",
             "All candidate definitions and promotion gates were fixed before training in protocol.json. Selection records were written before holdout prediction. These results reuse the previously inspected holdout and cannot establish fresh out-of-sample improvement.", "",
             "| Interval | Validation-selected candidate | Pass validation gate | Val R2 vs zero | Val IC | Val BA | Holdout R2 vs zero | Holdout IC | Holdout BA | Holdout amplitude ratio | Holdout price MAE | Flat price MAE |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        c = report["candidates"][report["selected"]]
        v, t = c["validation"]["h1"], c["sameHoldoutReevaluation"]["h1"]
        lines.append(f"|{report['interval']}|{report['selected']}|{report['eligible']}|{v['returnR2VsZero']:.5f}|{v['pearsonIC']:.5f}|{v['balancedAccuracy']:.4f}|{t['returnR2VsZero']:.5f}|{t['pearsonIC']:.5f}|{t['balancedAccuracy']:.4f}|{t['amplitudeStdRatio']:.4f}|{t['priceMaePct']:.5f}%|{t['flatPriceMaePct']:.5f}%|")
    lines += ["", "R2 vs zero = 1 - model return MSE / zero-return MSE. Amplitude ratio is predicted return std / realized return std; it is diagnostic, never an objective or promotion criterion. BA averages recall for positive and negative realized returns; flat predictions abstain and score zero. Constant nonzero majority-class predictions score 0.5. Pearson IC is undefined for constant predictions.", "",
              "Time blocks are common calendar blocks across all four symbols; each block excludes its final origin so its h1 label stays inside the block. Adjacent observations are correlated. No statistical significance or tradable return claim is made.", "",
              "Selected candidate JSON is stored under models/ for reproducibility only. Deployed model SHA256 values were unchanged. No curve magnification, artificial perturbation, minimum amplitude, or forced nonzero prediction was applied."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(OUT / "deployment-integrity.json", {"before": deployed_hashes, "after": after, "unchanged": deployed_hashes == after})


if __name__ == "__main__":
    main()

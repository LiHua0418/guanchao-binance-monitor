"""Causal CatBoost return/direction experiments with expanding temporal CV.

Run with .venv-training/Scripts/python.exe scripts/train-b4-trees.py.
Only writes experiment artifacts; never changes the deployed B3 weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
from sklearn.linear_model import LogisticRegression

from b4_data import get_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/b4-trees"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
HORIZONS = list(range(1, 13))
DIRECTION_HORIZONS = [1, 3, 6, 12]
SEED = 20261008
CONFIGS = [
    {"name": "depth4-l2-30", "depth": 4, "l2_leaf_reg": 30},
    {"name": "depth6-l2-60", "depth": 6, "l2_leaf_reg": 60},
]
FOLDS = [
    (["train"], "tune", "val1"),
    (["train", "tune"], "val1", "val2"),
    (["train", "tune", "val1"], "val2", "val3"),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=convert), encoding="utf-8")


def log(message):
    print(f"{now()} {message}", flush=True)


def combine(parts):
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]
            if isinstance(parts[0][key], np.ndarray)}


def feature_matrix(data, part):
    # No feature scaler is fit on data from later periods. Tree thresholds are
    # fitted directly on causal, finite raw values; symbol is known at origin.
    x = np.asarray(data["raw_features"][part["row_index"]], dtype=np.float32)
    return np.concatenate((x, np.eye(4, dtype=np.float32)[part["symbol"]]), axis=1)


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def metric(y, pred, probability):
    truth = y > 0
    labels = probability >= .5
    directional = y != 0
    positive, negative = truth, y < 0
    ba = .5 * (np.mean(labels[positive]) + np.mean(~labels[negative]))
    sd_y, sd_pred = float(np.std(y)), float(np.std(pred))
    ic = float(np.corrcoef(y, pred)[0, 1]) if sd_y > 1e-12 and sd_pred > 1e-12 else 0.
    prob_ic = float(np.corrcoef(y, probability)[0, 1]) if np.std(probability) > 1e-12 else 0.
    mse = float(np.mean((pred - y) ** 2))
    return {
        "samples": len(y), "returnMse": mse, "returnR2VsZero": float(1 - mse / max(np.mean(y*y), 1e-15)),
        "returnIc": ic, "directionProbabilityIc": prob_ic,
        "returnMaePct": float(np.mean(np.abs(pred-y)) * 100),
        "priceMaePct": float(np.mean(np.abs(np.expm1(np.clip(pred-y, -20, 20)))) * 100),
        "directionalSamples": int(np.sum(directional)),
        "balancedAccuracy": float(ba), "directionAccuracy": float(np.mean((labels == truth)[directional])),
        "brier": float(np.mean(((probability - truth) ** 2)[directional])),
        "logLoss": float(-np.mean((truth*np.log(np.clip(probability, 1e-7, 1-1e-7)) +
                                        (~truth)*np.log(np.clip(1-probability, 1e-7, 1-1e-7)))[directional])),
        "predictiveStdRatio": sd_pred / max(sd_y, 1e-12),
        "predictedReturnMeanPct": float(np.mean(pred)*100),
        "predictedAbsReturnPct": float(np.mean(np.abs(pred))*100),
        "actualAbsReturnPct": float(np.mean(np.abs(y))*100),
        "upProbabilityMean": float(np.mean(probability)),
        "upProbabilityStd": float(np.std(probability)),
        "positivePredictedFraction": float(np.mean(pred > 0)),
    }


def assess(part, h, pred, probability):
    y = part["y"][:, h-1]
    result = metric(y, pred, probability)
    result["bySymbol"] = {symbol: metric(y[part["symbol"] == i], pred[part["symbol"] == i],
                                         probability[part["symbol"] == i])
                          for i, symbol in enumerate(SYMBOLS) if np.any(part["symbol"] == i)}
    return result


def baselines(data, train, part, h):
    y = part["y"][:, h-1]
    # Mean/prior use past training labels only. Persistence is the completed
    # origin candle's one-bar return repeated h times, not a future label.
    feature_names = list(data["feature_names"])
    return_candidates = ["logReturn1", "return1", "return_1", "ret1", "ret_1", "log_return_1"]
    col = next((feature_names.index(s) for s in return_candidates if s in feature_names), None)
    if col is None:
        # Metadata supplies exact feature name; fail instead of inventing a
        # momentum baseline with an unrelated normalized feature.
        raise ValueError(f"Cannot locate causal one-bar return in features: {feature_names}")
    previous = data["raw_features"][part["row_index"], col] * h
    nonzero = train["y"][:, h-1] != 0
    prior = float(np.mean(train["y"][nonzero, h-1] > 0))
    mean = float(np.mean(train["y"][:, h-1]))
    return {
        "zero": assess(part, h, np.zeros_like(y), np.full_like(y, .5)),
        "trainMean": assess(part, h, np.full_like(y, mean), np.full_like(y, prior)),
        "persistence": assess(part, h, previous, (previous > 0).astype(float)),
        "majorityDirection": assess(part, h, np.full_like(y, mean), np.full_like(y, float(prior >= .5))),
        "trainingDirectionPrior": prior,
    }


def build_model(config, head, iterations, threads):
    cls = CatBoostRegressor if head == "return" else CatBoostClassifier
    return cls(
        iterations=iterations, depth=config["depth"], l2_leaf_reg=config["l2_leaf_reg"],
        learning_rate=.035, loss_function="RMSE" if head == "return" else "Logloss",
        thread_count=threads, random_seed=SEED, task_type="CPU", bootstrap_type="Bernoulli",
        subsample=.8, random_strength=1, border_count=64, verbose=False,
        allow_writing_files=False,
    )


def fit_head(config, head, x, part, h, iterations, threads, eval_x=None, eval_part=None):
    model = build_model(config, head, iterations, threads)
    target = np.clip(part["y_norm"][:, h-1], -8, 8) if head == "return" else (part["y"][:, h-1] > 0).astype(int)
    kwargs = {}
    if eval_part is not None:
        eval_y = np.clip(eval_part["y_norm"][:, h-1], -8, 8) if head == "return" else (eval_part["y"][:, h-1] > 0).astype(int)
        if head == "direction":
            keep = eval_part["y"][:, h-1] != 0
            eval_x, eval_y = eval_x[keep], eval_y[keep]
        kwargs = {"eval_set": (eval_x, eval_y), "early_stopping_rounds": 60, "use_best_model": True}
    if head == "direction":
        keep = part["y"][:, h-1] != 0
        x, target = x[keep], target[keep]
    model.fit(x, target, **kwargs)
    return model


def predictions(models, x, part, h, calibrator=None):
    prediction = models["return"].predict(x) * part["vol"] * np.sqrt(h)
    logits = np.asarray(models["direction"].predict(x, prediction_type="RawFormulaVal"))
    if calibrator is not None:
        probability = sigmoid(calibrator["slope"]*logits + calibrator["intercept"])
    else:
        probability = sigmoid(logits)
    return prediction, probability, logits


def selection_score(metrics, baseline):
    # Direction probability is the primary target; return skill still matters.
    # Both components are scale-free and compare to train-prior / zero returns.
    return .6 * metrics["brier"] / max(baseline["trainMean"]["brier"], 1e-12) + \
           .4 * metrics["returnMse"] / max(baseline["zero"]["returnMse"], 1e-12)


def eligibility(folds):
    # Conservative predeclared acceptance, applied only to OOS CV. No amplitude
    # floor: a useful probability prediction may still imply a small mean move.
    ms = [f["metrics"] for f in folds]
    return_edge = (np.mean([m["returnR2VsZero"] for m in ms]) > 0 and
                   np.mean([m["returnIc"] for m in ms]) > .02 and
                   sum(m["returnR2VsZero"] > 0 and m["returnIc"] > 0 for m in ms) >= 2)
    direction_edge = (np.mean([m["balancedAccuracy"] for m in ms]) > .51 and
                      sum(m["balancedAccuracy"] > .5 for m in ms) >= 2 and
                      np.mean([f["metrics"]["brier"] - f["baselines"]["trainMean"]["brier"] for f in folds]) < 0)
    return {"returnEligible": bool(return_edge), "directionEligible": bool(direction_edge),
            "eligible": bool(return_edge or direction_edge),
            "meanBalancedAccuracy": float(np.mean([m["balancedAccuracy"] for m in ms])),
            "meanReturnIc": float(np.mean([m["returnIc"] for m in ms])),
            "meanReturnR2VsZero": float(np.mean([m["returnR2VsZero"] for m in ms]))}


def temporal_bootstrap(part, pred, probability, baseline_probability, block_bars=48, draws=500):
    # Sample contiguous time blocks, keeping contemporaneous symbols together.
    # These are descriptive intervals for an already-inspected final period.
    times, inverse = np.unique(part["time"], return_inverse=True)
    blocks = inverse // block_bars
    unique = np.unique(blocks)
    y = part["y"][:, 0]
    truth = y > 0
    stats = []
    for block in unique:
        mask = blocks == block
        yp, pp, tp = y[mask], pred[mask], truth[mask]
        p = probability[mask]
        directional = yp != 0
        stats.append([len(yp), np.sum(yp*yp), np.sum((yp-pp)**2),
                      np.sum(((baseline_probability-tp)**2 - (p-tp)**2)[directional]),
                      np.sum(tp), np.sum((p >= .5) & tp), np.sum((p < .5) & (yp < 0)), np.sum(yp < 0),
                      np.sum(directional)])
    stats = np.asarray(stats)
    rng = np.random.default_rng(SEED)
    samples = stats[rng.integers(0, len(stats), (draws, len(stats)))].sum(axis=1)
    r2 = 1-samples[:, 2] / np.maximum(samples[:, 1], 1e-15)
    brier_improvement = samples[:, 3] / samples[:, 8]
    ba = .5*(samples[:, 5]/np.maximum(samples[:, 4], 1) + samples[:, 6]/np.maximum(samples[:, 7], 1))
    return {"method": "timestamp block bootstrap; all symbols retained per sampled block", "blockBars": block_bars,
            "draws": draws, "r2VsZero95": np.quantile(r2, [.025,.975]).tolist(),
            "brierImprovementVsTrainPrior95": np.quantile(brier_improvement,[.025,.975]).tolist(),
            "balancedAccuracy95": np.quantile(ba,[.025,.975]).tolist()}


def run_interval(interval, args):
    started = time.monotonic()
    folder = OUT / interval
    folder.mkdir(parents=True, exist_ok=True)
    log(f"DATA {interval} loading")
    data = get_dataset(interval)
    names = list(data["feature_names"]) + [f"symbol_{symbol}" for symbol in SYMBOLS]
    log(f"DATA {interval} train={len(data['train']['y'])} features={len(names)} test={len(data['test']['y'])}")
    protocol = {
        "startedAt": now(), "interval": interval, "seed": SEED, "catboostVersion": catboost.__version__,
        "features": names, "featureSource": "b4_data.py causal raw features plus four fixed symbol one-hot features",
        "featureTransforms": "No fitted feature scaling for trees; only train-fitted CatBoost thresholds. Causal origin vol normalizes targets.",
        "target": "Direct cumulative log return at every h1..h12, scaled by causal48barVol*sqrt(h); training target clipped to +/-8, evaluation never clipped; exact-zero returns excluded from classification training/metrics",
        "configurations": CONFIGS, "maxIterations": args.iterations, "earlyStoppingRounds":60,
        "heads": ["normalized-return RMSE", "positive-return Logloss"], "cpuThreads": args.threads,
        "walkForwardFolds": [{"training": f[0], "earlyStopping":f[1],"validation":f[2]} for f in FOLDS],
        "selection": "Only h1 determines depth/l2 and median early-stopped iterations; score=.6 Brier/trainPriorBrier+.4 returnMSE/zeroMSE averaged over3 expanding folds.",
        "finalFit": "Freeze winning configuration and median CV tree counts; refit train+tune+val1+val2+val3. No early stopping on calibration/test.",
        "calibration": "Separate 85-90% partition; Platt logistic mapping of direction logits (C=1), no test feedback; signed normalized residual quantiles",
        "remainingHorizons": "Use exactly selected h1 configuration and fixed per-head tree count for h2..h12 return heads and h3,h6,h12 direction heads; no horizon-dependent search or interpolation.",
        "gate": "Return: mean CV R2>0, mean IC>.02,2/3 folds positive IC/R2; Direction: mean CV BA>.51,2/3 BA>.5, mean Brier better than train prior",
        "holdoutStatus": "Expanded-history final partition may overlap previously inspected experiments; this is historical reevaluation, not untouched external evidence.",
        "deployment": "Experimental artifacts only. No deployed weights or src modification.",
        "dataMetadata": data["metadata"],
    }
    save(folder / "protocol.json", protocol)
    candidates = []
    for config in CONFIGS:
        folds = []
        for fi, (train_names, tune_name, val_name) in enumerate(FOLDS):
            train = combine([data[n] for n in train_names])
            tune, val = data[tune_name], data[val_name]
            x, tx, vx = (feature_matrix(data,p) for p in (train,tune,val))
            models, iteration_counts = {}, {}
            for head in ("return", "direction"):
                t = time.monotonic()
                log(f"FIT {interval} {config['name']} fold{fi+1} {head} n={len(x)}")
                models[head] = fit_head(config,head,x,train,1,args.iterations,args.threads,tx,tune)
                iteration_counts[head] = models[head].tree_count_
                log(f"FIT DONE {interval} {config['name']} fold{fi+1} {head} trees={models[head].tree_count_} seconds={time.monotonic()-t:.1f}")
            pred, prob, _ = predictions(models,vx,val,1)
            measured = assess(val,1,pred,prob)
            base = baselines(data,train,val,1)
            fold = {"fold": fi+1, "training":train_names,"earlyStopping":tune_name,"validation":val_name,
                    "iterations":iteration_counts,"metrics":measured,"baselines":base,
                    "score":selection_score(measured,base)}
            folds.append(fold)
            save(folder / f"{config['name']}-cv-progress.json", folds)
            np.savez_compressed(folder / f"{config['name']}-fold{fi+1}-predictions.npz", time=val["time"],symbol=val["symbol"],
                                truth=val["y"][:,0],prediction=pred,probability=prob)
            log(f"CV {interval} {config['name']} fold{fi+1} BA={measured['balancedAccuracy']:.4f} IC={measured['returnIc']:.4f} R2={measured['returnR2VsZero']:.5f}")
        candidates.append({"config":config,"folds":folds,"score":float(np.mean([f['score'] for f in folds])),"gate":eligibility(folds)})
    selected = min(candidates,key=lambda c:c["score"])
    iterations = {head:max(1,int(np.median([f['iterations'][head] for f in selected['folds']]))) for head in ('return','direction')}
    frozen = {"frozenAt":now(),"interval":interval,"selectedConfig":selected["config"],"iterations":iterations,
              "selectedGate":selected["gate"],"candidates":candidates,"selectionUsesTest":False}
    save(folder / "selection-frozen.json",frozen)
    log(f"SELECTED {interval} {selected['config']['name']} {iterations} {selected['gate']}")
    train = combine([data[n] for n in ("train","tune","val1","val2","val3")])
    cal, test = data["calibration"],data["test"]
    x,cx,ex = (feature_matrix(data,p) for p in (train,cal,test))
    results, artifacts, outputs = {}, {}, {}
    for h in HORIZONS:
        models = {}
        heads = ('return','direction') if h in DIRECTION_HORIZONS else ('return',)
        for head in heads:
            log(f"FINAL FIT {interval} h{h} {head} trees={iterations[head]} n={len(x)}")
            models[head]=fit_head(selected['config'],head,x,train,h,iterations[head],args.threads)
            stem=f"h{h}-{head}"
            models[head].save_model(str(folder/f"{stem}.cbm"))
            models[head].save_model(str(folder/f"{stem}.json"),format='json')
            artifacts[stem]={"cbm":f"{stem}.cbm","json":f"{stem}.json",
                             "sha256":hashlib.sha256((folder/f"{stem}.cbm").read_bytes()).hexdigest(),
                             "trees":models[head].tree_count_}
        if h not in DIRECTION_HORIZONS:
            cp = models['return'].predict(cx) * cal['vol'] * np.sqrt(h)
            residual = (cal['y'][:,h-1]-cp)/(cal['vol']*np.sqrt(h))
            residual_quantiles = np.quantile(residual,[.1,.5,.9]).tolist()
            save(folder/f"h{h}-calibration-frozen.json",{"frozenAt":now(),"residualQuantiles":residual_quantiles})
            pred = models['return'].predict(ex) * test['vol'] * np.sqrt(h)
            outputs[f"pred_h{h}"] = pred
            results[f"h{h}"] = {"residualQuantiles":residual_quantiles,
                                "returnMse":float(np.mean((test['y'][:,h-1]-pred)**2))}
            continue
        cp, _, logits = predictions(models,cx,cal,h)
        # Sigmoid calibration fitted only to separate calibration labels. It can
        # reduce, retain or invert slope; never refit after inspecting test.
        fit = LogisticRegression(C=1,solver='lbfgs',max_iter=500,random_state=SEED)
        nonzero = cal['y'][:,h-1] != 0
        fit.fit(logits[nonzero].reshape(-1,1),(cal['y'][nonzero,h-1]>0).astype(int))
        calibrator={"slope":float(fit.coef_[0,0]),"intercept":float(fit.intercept_[0])}
        residual=(cal['y'][:,h-1]-cp)/(cal['vol']*np.sqrt(h))
        residual_quantiles=np.quantile(residual,[.1,.5,.9]).tolist()
        save(folder/f"h{h}-calibration-frozen.json",{"frozenAt":now(),"calibrator":calibrator,"residualQuantiles":residual_quantiles})
        pred, prob, raw_logits=predictions(models,ex,test,h,calibrator)
        base=baselines(data,train,test,h)
        measured=assess(test,h,pred,prob)
        raw_measured=assess(test,h,pred,sigmoid(raw_logits))
        results[f"h{h}"]={"test":measured,"testUncalibrated":raw_measured,"baselines":base,"calibration":calibrator,
                           "residualQuantiles":residual_quantiles,
                           "featureImportance":sorted(zip(names,models['return'].get_feature_importance()),key=lambda p:-p[1])[:20]}
        if h==1:
            results[f"h{h}"]["bootstrap"]=temporal_bootstrap(test,pred,prob,base['trainingDirectionPrior'])
        outputs[f"pred_h{h}"]=pred
        outputs[f"prob_h{h}"]=prob
        log(f"TEST {interval} h{h} BA={measured['balancedAccuracy']:.4f} IC={measured['returnIc']:.4f} R2={measured['returnR2VsZero']:.5f} predStdRatio={measured['predictiveStdRatio']:.3f}")
    contract={"schemaVersion":1,"family":"b4-catboost-direct-return-and-direction","interval":interval,"horizons":HORIZONS,
              "featureNames":names,"featureModule":"scripts/b4_data.py","rawFeatureOrder":list(data['feature_names']),
              "symbolOrder":SYMBOLS,"input":"raw causal feature row at last fully closed candle plus four symbol one-hot values",
              "returnDecode":"rawRegressionOutput*originVolatility48*sqrt(h); price=lastClose*exp(decodedLogReturn)",
              "directionDecode":"rawClassificationLogit -> sigmoid(calibrationSlope*logit+calibrationIntercept)",
              "intermediateHorizons":"Every h1..h12 has its own directly trained regression model; no interpolation",
              "directionHorizons":DIRECTION_HORIZONS,
              "models":artifacts,"calibration":{h:r['calibration'] for h,r in results.items() if 'calibration' in r},
              "residualQuantiles":{h:r['residualQuantiles'] for h,r in results.items()},
              "selectionGate":selected['gate'],"dataMetadata":data['metadata']}
    save(folder/'model-contract.json',contract)
    np.savez_compressed(folder/'test-predictions.npz',time=test['time'],symbol=test['symbol'],truth=test['y'],
                        close=test['close'],vol=test['vol'],**outputs)
    report={"interval":interval,"completedAt":now(),"seconds":time.monotonic()-started,"selection":frozen,
            "results":results,"contract":"model-contract.json","holdoutStatus":protocol['holdoutStatus']}
    save(folder/'report.json',report)
    return report


def write_report(reports):
    # A later invocation may train only remaining intervals. Preserve every
    # completed interval in the human-readable combined report.
    by_interval = {r['interval']:r for r in reports}
    for interval in ['15m','1h','4h','1d']:
        path = OUT/interval/'report.json'
        if path.exists() and interval not in by_interval:
            by_interval[interval] = json.loads(path.read_text(encoding='utf-8'))
    reports = [by_interval[i] for i in ['15m','1h','4h','1d'] if i in by_interval]
    lines=["# B4 CatBoost direction and amplitude experiment", "", "Causal features, three expanding temporal validation folds, separate probability calibration. No changes to deployed model weights.", "",
           "Final historical reevaluation can overlap previous experiments. It is not fresh untouched evidence; no candidate selection uses its results.", "",
           "| Interval | CV return eligible | CV direction eligible | Test BA | Uncalibrated BA | Return IC | R² vs zero | Pred/actual std | Model price MAE % | Flat price MAE % |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        r=report['results']['h1']; m=r['test']; gate=report['selection']['selectedGate']
        lines.append(f"| {report['interval']} | {gate['returnEligible']} | {gate['directionEligible']} | {m['balancedAccuracy']:.4f} | {r['testUncalibrated']['balancedAccuracy']:.4f} | {m['returnIc']:.4f} | {m['returnR2VsZero']:.5f} | {m['predictiveStdRatio']:.4f} | {m['priceMaePct']:.5f} | {r['baselines']['zero']['priceMaePct']:.5f} |")
    lines += ["", "No amplitude floor or artificial curve amplification is applied. Conditional mean returns are allowed to be small; classification and uncertainty remain separate quantities.", "",
              "Each interval directory contains the preregistered protocol, frozen validation selection, portable JSON and CBM models, model contract, frozen calibration, per-symbol diagnostics, block-bootstrap intervals and final predictions.", ""]
    lines += ['## Direction comparator audit','',
              'The main direction head is h1. The simple reversal rule is the opposite sign of the latest completed candle; it is a descriptive comparator added after the frozen predictions, not a test-selected replacement. Daily h1 direction did not pass the validation gate; later daily direction heads have no independent validation gate.', '',
              '| Interval | h1 direction BA | Reversal BA | Brier model | Brier training prior | BA bootstrap 95% |','|---|---:|---:|---:|---:|---:|']
    for report in reports:
        r=report['results']['h1']; b=r['bootstrap']['balancedAccuracy95']
        reversal=r['baselines'].get('reversalDiagnostic',{}).get('balancedAccuracy')
        comparison=f'{reversal:.2%}' if reversal is not None else 'pending verification'
        lines.append(f"| {report['interval']} | {r['test']['balancedAccuracy']:.2%} | {comparison} | {r['test']['brier']:.6f} | {r['baselines']['trainMean']['brier']:.6f} | [{b[0]:.2%}, {b[1]:.2%}] |")
    lines += ['', '15m and 1h show modest historical direction information; 4h classification is weaker than the simple reversal comparator despite beating the training-prior Brier score. Daily direction remains unvalidated. Direction evidence is insufficient to claim a useful signed-price path, and mean-return regression remains weak on all intervals. Later classification horizons must not inherit the h1 validation label.', '',
              'For a product integration, label h1 probabilities experimental and expose calibrated up probability separately from any magnitude/scenario display. The separately trained `b4-magnitude` family estimates unsigned movement; multiplying it by the sign/probability head does not produce a validated point forecast.', '',
              'Artifact verification checks all source/model hashes, reproduces all stored predictions and compares origin-prefix features/predictions with the same origin after changing future candles. No deployed B3 model was overwritten by this experiment.', '']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def load_models(interval, experiment_root=OUT):
    """Load persisted experiment once for a local inference service."""
    folder = Path(experiment_root) / interval
    contract = json.loads((folder/'model-contract.json').read_text(encoding='utf-8'))
    models = {}
    for name, info in contract['models'].items():
        path = folder / info['cbm']
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Model hash mismatch: {path}')
        model = CatBoostRegressor() if name.endswith('-return') else CatBoostClassifier()
        model.load_model(str(path))
        models[name] = model
    return {'contract': contract, 'models': models}


def predict_features(loaded, raw_features, symbols, origin_volatility, last_close):
    """Predict every future bar from fully closed-origin features (no refit).

    Inputs are batch arrays N x 57, N symbol indices, N causal volatility48 and
    N origin close. Use b4_data.features on closed candles for raw_features.
    Returns N x 12 mean log returns, prices, calibrated residual intervals and
    calibrated direction probabilities for independently trained heads only.
    """
    contract, models = loaded['contract'], loaded['models']
    raw = np.atleast_2d(np.asarray(raw_features, dtype=np.float32))
    symbols = np.atleast_1d(np.asarray(symbols, dtype=int))
    vol = np.atleast_1d(np.asarray(origin_volatility, dtype=float))
    close = np.atleast_1d(np.asarray(last_close, dtype=float))
    if raw.shape != (len(symbols), len(contract['rawFeatureOrder'])) or len(vol) != len(raw) or len(close) != len(raw):
        raise ValueError('Inference feature contract/shape mismatch')
    if (not np.isfinite(raw).all() or not np.isfinite(vol).all() or not np.isfinite(close).all()
            or np.any((symbols < 0) | (symbols >= len(SYMBOLS))) or np.any(vol <= 0) or np.any(close <= 0)):
        raise ValueError('Invalid inference inputs')
    x = np.concatenate((raw, np.eye(4,dtype=np.float32)[symbols]),axis=1)
    returns, lower, upper, probabilities = [], [], [], {}
    for h in contract['horizons']:
        scale = vol * np.sqrt(h)
        pred = np.asarray(models[f'h{h}-return'].predict(x,thread_count=1)) * scale
        quantiles = contract['residualQuantiles'][f'h{h}']
        returns.append(pred)
        lower.append(pred + quantiles[0]*scale)
        upper.append(pred + quantiles[2]*scale)
        if h in contract['directionHorizons']:
            logits = np.asarray(models[f'h{h}-direction'].predict(x,prediction_type='RawFormulaVal',thread_count=1))
            cal = contract['calibration'][f'h{h}']
            probabilities[f'h{h}'] = sigmoid(cal['slope']*logits+cal['intercept'])
    returns = np.stack(returns,axis=1)
    return {'logReturns': returns, 'prices': close[:,None]*np.exp(returns),
            'lowerPrices': close[:,None]*np.exp(np.stack(lower,axis=1)),
            'upperPrices': close[:,None]*np.exp(np.stack(upper,axis=1)),
            'directionProbabilities': probabilities, 'selectionGate':contract['selectionGate'],
            'directionProbabilityMeaning':'P(positive return conditional on a nonzero future return); exact-zero targets excluded from classification training/calibration',
            'selectionScope':'h1 only; later independent heads share configuration but do not inherit h1 validation',
            'predictionMeaning':'Conditional mean signed-return regression; historical signed-price skill remains unverified'}


def predict_from_dataset(loaded, data, part):
    return predict_features(loaded,data['raw_features'][part['row_index']],part['symbol'],part['vol'],part['close'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals',nargs='+',choices=['15m','1h','4h','1d'],default=['1h','15m','4h','1d'])
    parser.add_argument('--iterations',type=int,default=600)
    parser.add_argument('--threads',type=int,default=6)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[]
    for interval in args.intervals:
        reports.append(run_interval(interval,args))
        write_report(reports)


if __name__=='__main__':
    main()

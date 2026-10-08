"""Larger causal TCN with direction and volatility-normalized return heads.

Only experiment artifacts are written. Validation blocks choose a two-seed
candidate; calibration/test never choose architecture, epoch or seed. Loading
NeuralPredictor supports raw candles or precomputed raw feature windows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression

from b4_data import get_dataset, features as candle_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/b4-neural"
SEEDS = [20261008, 20261009]
SEQUENCE = 64
WIDTH = 64
HORIZONS = np.arange(1, 13)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
BLOCKS = ["val1", "val2", "val3"]
LOSS_WEIGHTS = np.full(12, .05 / 8, dtype=np.float32)
LOSS_WEIGHTS[[0, 2, 5, 11]] = [.65, .1, .1, .1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=convert), encoding="utf-8")


def log(message):
    print(f"{now()} {message}", flush=True)


def resolve_device(device="auto"):
    device=str(device)
    resolved=("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA requested but unavailable in torch {torch.__version__}; use the CUDA training venv or --device cpu")
    if resolved.startswith("cuda"):
        # Match float32 CPU artifacts closely. This is a compute choice, never
        # a candidate selection or prediction-amplitude transformation.
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    return torch.device(resolved)


def combine(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if isinstance(parts[0][k], np.ndarray)}


class ResidualCausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout=.1):
        super().__init__()
        self.dilation = dilation
        self.first = nn.Conv1d(width, width, 3, dilation=dilation)
        self.second = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.first(F.pad(x, (2 * self.dilation, 0)))
        y = self.norm1(y.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(F.silu(y))
        y = self.second(F.pad(y, (2 * self.dilation, 0)))
        y = self.norm2(y.transpose(1, 2)).transpose(1, 2)
        return F.silu(x + self.dropout(y))


class B4TCN(nn.Module):
    def __init__(self, feature_count=57, width=WIDTH):
        super().__init__()
        self.feature_count, self.width = feature_count, width
        self.input = nn.Conv1d(feature_count, width, 1)
        self.blocks = nn.ModuleList([ResidualCausalBlock(width, d) for d in (1, 2, 4, 8)])
        self.attention = nn.Linear(width, 1)
        self.shared = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Dropout(.1))
        self.returns = nn.Linear(width, 12)
        self.direction = nn.Linear(width, 12)

    def encode(self, x):
        h = self.input(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return h.transpose(1, 2)

    def forward(self, x):
        h = self.encode(x)
        attention = torch.softmax(self.attention(h), dim=1)
        context = torch.sum(attention * h, dim=1)
        z = self.shared(torch.cat((h[:, -1], context), dim=1))
        return self.returns(z), self.direction(z)


def metric(y, pred, probability):
    """Definition matches the B4 tree experiment; exact-zero direction abstains."""
    truth, positive, negative = y > 0, y > 0, y < 0
    directional = y != 0
    labels = probability >= .5
    sd_y, sd_p = float(np.std(y)), float(np.std(pred))
    ic = float(np.corrcoef(y, pred)[0, 1]) if sd_y > 1e-12 and sd_p > 1e-12 else 0.
    prob_ic = float(np.corrcoef(y, probability)[0, 1]) if np.std(probability) > 1e-12 else 0.
    mse = float(np.mean((pred - y) ** 2))
    return {"samples": len(y), "returnMse": mse,
            "returnR2VsZero": float(1 - mse / max(np.mean(y*y), 1e-15)),
            "returnIc": ic, "directionProbabilityIc": prob_ic,
            "returnMaePct": float(np.mean(np.abs(pred-y))*100),
            "priceMaePct": float(np.mean(np.abs(np.expm1(np.clip(pred-y,-20,20))))*100),
            "directionalSamples": int(directional.sum()),
            "balancedAccuracy": float(.5*(np.mean(labels[positive])+np.mean(~labels[negative]))),
            "directionAccuracy": float(np.mean((labels==truth)[directional])),
            "brier": float(np.mean(((probability-truth)**2)[directional])),
            "logLoss": float(-np.mean((truth*np.log(np.clip(probability,1e-7,1-1e-7))+
                          (~truth)*np.log(np.clip(1-probability,1e-7,1-1e-7)))[directional])),
            "predictiveStdRatio": sd_p/max(sd_y,1e-12),
            "predictedReturnMeanPct": float(np.mean(pred)*100),
            "predictedAbsReturnPct": float(np.mean(np.abs(pred))*100),
            "actualAbsReturnPct": float(np.mean(np.abs(y))*100),
            "upProbabilityMean": float(np.mean(probability)),
            "upProbabilityStd": float(np.std(probability)),
            "positivePredictedFraction": float(np.mean(pred>0))}


def assess(part, h, pred, probability):
    y = part["y"][:, h-1]
    result = metric(y, pred, probability)
    result["bySymbol"] = {s: metric(y[part["symbol"]==i], pred[part["symbol"]==i],
                                          probability[part["symbol"]==i])
                          for i,s in enumerate(SYMBOLS) if np.any(part["symbol"]==i)}
    return result


def baselines(data, train, part, h):
    y = part["y"][:, h-1]
    col = list(data["feature_names"]).index("logReturn1")
    previous = data["raw_features"][part["row_index"], col] * h
    nonzero = train["y"][:, h-1] != 0
    prior = float(np.mean(train["y"][nonzero, h-1] > 0))
    mean = float(np.mean(train["y"][:, h-1]))
    return {"zero": assess(part,h,np.zeros_like(y),np.full_like(y,.5)),
            "trainMean": assess(part,h,np.full_like(y,mean),np.full_like(y,prior)),
            "persistence": assess(part,h,previous,(previous>0).astype(float)),
            "majorityDirection": assess(part,h,np.full_like(y,mean),np.full_like(y,float(prior>=.5))),
            "trainingDirectionPrior": prior}


def selection_score(m, baseline):
    return .6*m["brier"]/max(baseline["trainMean"]["brier"],1e-12) + \
           .4*m["returnMse"]/max(baseline["zero"]["returnMse"],1e-12)


def eligibility(blocks):
    ms=[f["metrics"] for f in blocks]
    returns=(np.mean([m["returnR2VsZero"] for m in ms])>0 and np.mean([m["returnIc"] for m in ms])>.02 and
             sum(m["returnR2VsZero"]>0 and m["returnIc"]>0 for m in ms)>=2)
    direction=(np.mean([m["balancedAccuracy"] for m in ms])>.51 and
               sum(m["balancedAccuracy"]>.5 for m in ms)>=2 and
               np.mean([f["metrics"]["brier"]-f["baselines"]["trainMean"]["brier"] for f in blocks])<0)
    return {"returnEligible":bool(returns),"directionEligible":bool(direction),"eligible":bool(returns or direction),
            "meanBalancedAccuracy":float(np.mean([m["balancedAccuracy"] for m in ms])),
            "meanReturnIc":float(np.mean([m["returnIc"] for m in ms])),
            "meanReturnR2VsZero":float(np.mean([m["returnR2VsZero"] for m in ms]))}


def network_predictions(model, all_x, part, batch_size=512):
    model.eval()
    device=next(model.parameters()).device
    all_x=all_x.to(device)
    indices=torch.as_tensor(part["row_index"],dtype=torch.long,device=device)
    offsets=torch.arange(1-SEQUENCE,1,device=device)
    outputs, probabilities=[],[]
    with torch.no_grad():
        for anchors in indices.split(batch_size):
            mu,logits=model(all_x[anchors[:,None]+offsets])
            outputs.append(mu.cpu().numpy().astype(np.float64))
            probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
    return np.concatenate(outputs),np.concatenate(probabilities)


def fit(data, all_x, train, tune, seed, epochs, batch_size, fixed_epochs=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=all_x.device
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    model=B4TCN(len(data["feature_names"])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2)
    target=torch.as_tensor(np.clip(train["y_norm"],-8,8),dtype=torch.float32,device=device)
    labels=torch.as_tensor(train["y"]>0,dtype=torch.float32,device=device)
    nonzero=torch.as_tensor(train["y"]!=0,dtype=torch.float32,device=device)
    anchors=torch.as_tensor(train["row_index"],dtype=torch.long,device=device)
    offsets=torch.arange(1-SEQUENCE,1,device=device)
    weights=torch.tensor(LOSS_WEIGHTS,device=device)
    generator=torch.Generator().manual_seed(seed)
    best_score,best_state,best_epoch,wait=float("inf"),None,0,0
    history=[]
    tune_baseline=baselines(data,train,tune,1) if tune is not None else None
    for epoch in range(1,epochs+1):
        started=time.monotonic();model.train();total=0.;count=0
        for ids in torch.randperm(len(anchors),generator=generator).split(batch_size):
            ids=ids.to(device)
            x=all_x[anchors[ids,None]+offsets]
            optimizer.zero_grad(set_to_none=True)
            mu,logits=model(x)
            regression=((mu-target[ids])**2*weights).sum(dim=1).mean()
            classification=(F.binary_cross_entropy_with_logits(logits,labels[ids],reduction="none")*nonzero[ids]*weights).sum()/torch.clamp((nonzero[ids]*weights).sum(),min=1)
            loss=.4*regression+.6*classification
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(ids);count+=len(ids)
        if fixed_epochs:
            history.append({"epoch":epoch,"trainLoss":total/count,"seconds":time.monotonic()-started})
            log(f"REFIT {data['metadata']['interval']} seed={seed} epoch={epoch}/{epochs} loss={total/count:.5f} seconds={history[-1]['seconds']:.1f}")
            continue
        mu,probability=network_predictions(model,all_x,tune)
        pred=mu*tune["vol"][:,None]*np.sqrt(HORIZONS)
        m=assess(tune,1,pred[:,0],probability[:,0]);score=selection_score(m,tune_baseline)
        scheduler.step(score)
        history.append({"epoch":epoch,"trainLoss":total/count,"tuneScore":score,"tuneH1":m,"seconds":time.monotonic()-started})
        if score<best_score-1e-6:
            best_score,best_epoch,wait=score,epoch,0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
        log(f"FIT {data['metadata']['interval']} seed={seed} epoch={epoch} tuneBA={m['balancedAccuracy']:.4f} tuneR2={m['returnR2VsZero']:.5f} score={score:.6f} wait={wait} seconds={history[-1]['seconds']:.1f}")
        if wait>=4:
            break
    if not fixed_epochs:
        model.load_state_dict(best_state)
    else:
        best_epoch=epochs
    return model,{"seed":seed,"bestEpoch":best_epoch,"epochsRun":len(history),"history":history}


def ensemble_predictions(models, all_x, part):
    outputs=[network_predictions(m,all_x,part) for m in models]
    return np.mean([p[0] for p in outputs],axis=0),np.mean([p[1] for p in outputs],axis=0)


def calibrate_probability(y, probability):
    parameters=[]
    for h in range(12):
        keep=y[:,h]!=0
        labels=(y[keep,h]>0).astype(int)
        logits=np.log(np.clip(probability[keep,h],1e-7,1-1e-7)/np.clip(1-probability[keep,h],1e-7,1))
        if len(np.unique(labels))==2:
            lr=LogisticRegression(C=1.,solver="lbfgs",max_iter=500)
            lr.fit(logits[:,None],labels)
            parameters.append({"slope":float(lr.coef_[0,0]),"intercept":float(lr.intercept_[0])})
        else:
            parameters.append({"slope":1.,"intercept":0.})
    return parameters


def apply_calibration(probability, parameters):
    logits=np.log(np.clip(probability,1e-7,1-1e-7)/np.clip(1-probability,1e-7,1))
    transformed=logits*np.asarray([p["slope"] for p in parameters])+np.asarray([p["intercept"] for p in parameters])
    return 1/(1+np.exp(-np.clip(transformed,-40,40)))


class NeuralPredictor:
    """Local inference from a frozen manifest; CPU by default, optional CUDA."""
    def __init__(self, artifact_folder, device="cpu"):
        self.folder=Path(artifact_folder)
        self.device=resolve_device(device)
        self.manifest=json.loads((self.folder/"manifest.json").read_text(encoding="utf-8"))
        self.models=[]
        for item in self.manifest["members"]:
            path=self.folder/item["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
                raise ValueError("Neural member SHA256 mismatch")
            model=B4TCN(len(self.manifest["features"]),self.manifest["width"]).to(self.device)
            model.load_state_dict(torch.load(path,map_location=self.device,weights_only=True))
            model.eval();self.models.append(model)

    def predict_features(self, raw_windows, vol):
        windows=np.asarray(raw_windows,dtype=np.float64)
        if windows.ndim==2:windows=windows[None]
        if windows.shape[1:]!=(SEQUENCE,len(self.manifest["features"])) or not np.isfinite(windows).all():
            raise ValueError("Expected finite [N,64,featureCount] raw features")
        vol=np.asarray(vol,dtype=np.float64).reshape(-1)
        if len(vol)!=len(windows) or np.any(vol<1e-5) or not np.isfinite(vol).all():
            raise ValueError("Expected one finite causal volatility >=1e-5 per window")
        standardized=np.clip((windows-np.asarray(self.manifest["mean"]))/np.asarray(self.manifest["scale"]),-12,12).astype(np.float32)
        output=[]
        with torch.no_grad():
            x=torch.from_numpy(standardized).to(self.device)
            for model in self.models:
                mus,probs=[],[]
                for batch in x.split(512):
                    mu,logits=model(batch)
                    mus.append(mu.cpu().numpy().astype(np.float64))
                    probs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
                output.append((np.concatenate(mus),np.concatenate(probs)))
        mu=np.mean([p[0] for p in output],axis=0)
        probability=apply_calibration(np.mean([p[1] for p in output],axis=0),self.manifest["probabilityCalibration"])
        scaling=vol[:,None]*np.sqrt(HORIZONS)
        return {"logReturns":mu*scaling,"upProbability":probability,
                "lowerLogReturns":(mu+np.asarray(self.manifest["residualLowerNormalized"]))*scaling,
                "upperLogReturns":(mu+np.asarray(self.manifest["residualUpperNormalized"]))*scaling}

    def predict_candles(self, candles):
        candles=np.asarray(candles,dtype=np.float64)
        if candles.ndim!=2 or candles.shape[1]!=10 or len(candles)<163:
            raise ValueError("Need >=163 closed candles, columns time/open/high/low/close/baseVolume/quoteVolume/trades/takerBuyBase/takerBuyQuote")
        x,vol,names=candle_features(candles)
        if names!=self.manifest["features"]:raise ValueError("Feature schema mismatch")
        forecast=self.predict_features(x[-SEQUENCE:],vol[-1:])
        forecast["prices"]=candles[-1,4]*np.exp(forecast["logReturns"])
        return forecast


def run_interval(interval,args):
    started=time.monotonic();data=get_dataset(interval)
    if data["metadata"].get("maxSequence",0)<SEQUENCE:raise ValueError("Dataset sequence warmup too short")
    folder=OUT/interval;folder.mkdir(parents=True,exist_ok=True)
    all_x=torch.from_numpy(np.asarray(data["features"],dtype=np.float32)).to(args.device)
    protocol={"startedAt":now(),"interval":interval,"data":data["metadata"],"seeds":SEEDS,
              "architecture":"TCN residual causal dilations1,2,4,8, two kernel3 convolutions each; time-local LayerNorm; attention+last pooling; shared64; separate12 return/direction heads",
              "sequence":SEQUENCE,"width":WIDTH,"parameterCount":sum(p.numel() for p in B4TCN(len(data["feature_names"])).parameters()),
              "device":str(args.device),"torchVersion":str(torch.__version__),"cudaVersion":torch.version.cuda,
              "cpuThreads":args.threads,"batchSize":args.batch_size,"maxEpochs":args.epochs,"earlyStopPatience":4,
              "loss":".4 normalized-return MSE + .6 nonzero-direction BCE; return training labels clipped[-8,8], evaluation never clipped",
              "horizonWeights":LOSS_WEIGHTS,"optimizer":"AdamW lr.0006 weight_decay.02, gradient norm1; ReduceLROnPlateau factor.5 patience2 on tune",
              "dropout":.1,"scalers":"Original train-calendar mean/std only, clip[-12,12]; causal sigma48>=1e-5 target normalization",
              "checkpoint":"train on train60%; tune10% selects epoch only using h1 .6 Brier/priorBrier+.4 MSE/zeroMSE",
              "validation":"Same frozen early model evaluated in val1/val2/val3. These are 3 successive forward blocks, not expanding refits as used by the tree branch.",
              "candidates":"seed20261008,seed20261009,equal ensemble2; eligibility and average score determine choice",
              "gate":"Return: mean block R2>0,IC>.02 and2/3 positive R2/IC; direction: meanBA>.51,2/3BA>.5,meanBrier better than train prior; independent capability flags; either may qualify",
              "finalFit":"Selected seed(s) refit on train+tune+val1+val2+val3 with fixed ceil median selected bestEpoch; no calibration/test early stop",
              "calibration":"Separate calibration-only Platt logit map C1 per horizon; normalized residual q10/q90 of actual selected ensemble",
              "holdoutStatus":"Historical reevaluation; recent dates overlap previously inspected B3 experiments. No claim of fresh untouched test.",
              "deployment":"No deployed weights or frontend changed; experiment artifacts only"}
    save(folder/"protocol.json",protocol)
    log(f"START {interval} samples={len(data['train']['y'])} features={len(data['feature_names'])} parameters={protocol['parameterCount']}")
    networks,runs={},{}
    for seed in SEEDS:
        networks[seed],runs[seed]=fit(data,all_x,data["train"],data["tune"],seed,args.epochs,args.batch_size)
    candidates={}
    for name,seeds in [(f"seed-{s}",[s]) for s in SEEDS]+[("ensemble-2",SEEDS)]:
        block_reports=[]
        for block in BLOCKS:
            part=data[block];mu,p=ensemble_predictions([networks[s] for s in seeds],all_x,part)
            pred=mu*part["vol"][:,None]*np.sqrt(HORIZONS)
            m=assess(part,1,pred[:,0],p[:,0]);base=baselines(data,data["train"],part,1)
            block_reports.append({"block":block,"metrics":m,"baselines":base,"score":selection_score(m,base)})
        candidates[name]={"seeds":seeds,"blocks":block_reports,"gate":eligibility(block_reports),
                          "score":float(np.mean([b["score"] for b in block_reports]))}
    eligible=[n for n,c in candidates.items() if c["gate"]["eligible"]]
    selected=min(eligible or candidates.keys(),key=lambda n:candidates[n]["score"])
    selected_seeds=candidates[selected]["seeds"]
    refit_epochs=max(1,int(np.ceil(np.median([runs[s]["bestEpoch"] for s in selected_seeds]))))
    selection={"frozenAt":now(),"selected":selected,"eligible":bool(eligible),"candidates":candidates,
               "refitEpochs":refit_epochs,"runs":runs}
    save(folder/"selection-frozen.json",selection)
    log(f"SELECT {interval} {selected} eligible={bool(eligible)} gate={candidates[selected]['gate']} refitEpochs={refit_epochs}")
    final_train=combine([data[n] for n in ("train","tune","val1","val2","val3")])
    final_models,refit_runs=[],[]
    for seed in selected_seeds:
        model,run=fit(data,all_x,final_train,None,seed,refit_epochs,args.batch_size,True)
        final_models.append(model);refit_runs.append(run)
    cal=data["calibration"];cal_mu,cal_probability=ensemble_predictions(final_models,all_x,cal)
    parameters=calibrate_probability(cal["y"],cal_probability)
    lower,upper=np.quantile(cal["y_norm"]-cal_mu,[.1,.9],axis=0)
    members=[]
    for seed,model in zip(selected_seeds,final_models):
        filename=f"seed-{seed}.pt";path=folder/filename
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
        members.append({"seed":seed,"file":filename,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest={"schemaVersion":"b4-neural-tcn-v1","name":"guanchao-b4-tcn-experimental","architecture":"TCN-Attention-Multitask",
              "interval":interval,"sequence":SEQUENCE,"width":WIDTH,"horizons":HORIZONS,"features":data["feature_names"],
              "mean":data["mean"],"scale":data["scale"],"featureClip":12,"members":members,
              "ensemble":"equal mean normalized return output and equal mean sigmoid probabilities, then fixed calibration",
              "probabilityCalibration":parameters,"residualLowerNormalized":lower,"residualUpperNormalized":upper,
              "targetNormalization":"causal sigma48*sqrt(horizon), sigma48 floor1e-5; no output amplification or minimum amplitude",
              "capabilities":candidates[selected]["gate"],"selection":selection["frozenAt"],"frozenAt":now(),
              "deploymentStatus":"experimental, not deployed","data":data["metadata"]}
    # Final model/calibration manifest is durable before looking at final labels.
    save(folder/"manifest.json",manifest)
    test=data["test"];mu,p_raw=ensemble_predictions(final_models,all_x,test);p=apply_calibration(p_raw,parameters)
    pred=mu*test["vol"][:,None]*np.sqrt(HORIZONS)
    by_horizon=[{"horizon":h,"metrics":assess(test,h,pred[:,h-1],p[:,h-1]),
                 "uncalibratedDirection":assess(test,h,pred[:,h-1],p_raw[:,h-1]),
                 "baselines":baselines(data,final_train,test,h),
                 "coverage80":float(np.mean((test['y_norm'][:,h-1]>=mu[:,h-1]+lower[h-1]) &
                                              (test['y_norm'][:,h-1]<=mu[:,h-1]+upper[h-1])))} for h in range(1,13)]
    # Loading saved tensors must reproduce the very same selected model outputs.
    ids=test["row_index"][:3,None]+np.arange(1-SEQUENCE,1)
    raw=data["raw_features"][ids]
    reloaded=NeuralPredictor(folder).predict_features(raw,test["vol"][:3])
    parity_error=float(np.max(np.abs(reloaded["logReturns"]-pred[:3])))
    if parity_error>1e-6:raise AssertionError(f"Reload inference mismatch {parity_error}")
    save(folder/"parity.json",{"rawFeatureWindows":raw,"vol":test["vol"][:3],"origin":test["time"][:3],
                              "logReturns":pred[:3],"upProbability":p[:3],"maxReturnReloadError":parity_error})
    report={"protocol":protocol,"selection":selection,"refitRuns":refit_runs,"finalEvaluation":by_horizon,
            "parityMaxError":parity_error,"durationSeconds":time.monotonic()-started}
    save(folder/"report.json",report)
    m=by_horizon[0]["metrics"];base=by_horizon[0]["baselines"]
    text=[f"# B4 TCN {interval} 实验", "",f"网络参数：{protocol['parameterCount']}；序列64；57个因果输入；设备 {args.device}，CPU算子 {args.threads}线程。2种子和其等权集成按预声明规则在3个验证时间块中选择。", "",
          f"选定：{selected}；验证准入：{bool(eligible)}；收益能力：{manifest['capabilities']['returnEligible']}；方向能力：{manifest['capabilities']['directionEligible']}。固定refit轮数：{refit_epochs}。", "",
          "| 下一根既有留出复评 | 数值 |", "|---|---:|",
          f"| 收益R2相对零收益 | {m['returnR2VsZero']:.6f} |",f"| 收益IC | {m['returnIc']:.6f} |",
          f"| 方向平衡准确率 | {m['balancedAccuracy']:.4%} |",f"| Brier | {m['brier']:.6f} |",
          f"| 训练先验Brier | {base['trainMean']['brier']:.6f} |",f"| 价格MAE | {m['priceMaePct']:.6f}% |",
          f"| 持平MAE | {base['zero']['priceMaePct']:.6f}% |",f"| 预测/实际收益std | {m['predictiveStdRatio']:.6f} |", "",
          "训练和验证只使用各自可见过去数据。三验证块使用同一早期模型，区别于树模型的逐折expanding refit。最终refit使用固定轮数和全部校准之前数据，特征标准化参数仍仅来自最初训练段。概率校准与残差区间仅来自校准段。此留出段与以前实验有时间重合，结果是历史复评，不是新的未见测试。", "",
          "模型提供独立收益头和方向概率头；两者方向可能不同，不能把方向分类达标写成收益幅度预测达标。没有输出放大、噪声、非零下限或测试选模。完整逐币/逐步数指标和冻结协议见JSON。", "",
          f"文件重载收益一致性误差：{parity_error:.3g}；耗时：{report['durationSeconds']:.1f}秒。", "",
          "本地推理：导入 scripts/train-b4-neural.py 中 NeuralPredictor(folder)，调用 predict_candles(candles) 或 predict_features(raw_windows,vol)。原始candles需10列且仅含已闭合K线，至少163根；返回12根累计log收益、上涨概率和校准残差区间。实验模型未部署。"]
    (folder/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    log(f"RESULT {interval} eligible={bool(eligible)} BA={m['balancedAccuracy']:.4f} R2={m['returnR2VsZero']:.6f} IC={m['returnIc']:.6f} Brier={m['brier']:.6f} baseline={base['trainMean']['brier']:.6f} seconds={report['durationSeconds']:.1f}")
    return {"interval":interval,"eligible":bool(eligible),"capabilities":manifest["capabilities"],
            "selected":selected,"h1":m,"h1Baselines":base,"durationSeconds":report["durationSeconds"]}


def smoke(device="cpu"):
    device=resolve_device(device)
    torch.manual_seed(SEEDS[0]);model=B4TCN().to(device);model.eval();x=torch.randn(3,64,57,device=device)
    with torch.no_grad():
        encoded=model.encode(x)
        prefix=model.encode(x[:,:31])
        assert torch.allclose(encoded[:,:31],prefix,atol=2e-6,rtol=1e-5)
        mu,logits=model(x)
        assert mu.shape==(3,12) and logits.shape==(3,12)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0006)
    mu,logits=model(x);loss=mu.square().mean()+F.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits))
    loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    # Exercise the actual indexed-sequence fit/predict path too, rather than
    # testing only the network's device transfer. All data here are synthetic.
    rng=np.random.default_rng(SEEDS[0])
    raw=rng.normal(size=(128,57)).astype(np.float32)
    data={"feature_names":["logReturn1"]+[f"synthetic{i}" for i in range(1,57)],
          "raw_features":raw,"metadata":{"interval":"synthetic-device-smoke"}}
    def part(start,count):
        sign=np.where((np.arange(count)//4)%2,1.,-1.)
        y=sign[:,None]*(np.abs(rng.normal(size=(count,12)))*.001+.0001)
        vol=np.full(count,.005)
        return {"row_index":np.arange(start,start+count),"y":y,
                "y_norm":y/(vol[:,None]*np.sqrt(HORIZONS)),"vol":vol,"symbol":np.arange(count)%4}
    train,tune=part(63,32),part(95,16)
    fitted,run=fit(data,torch.tensor(raw,device=device),train,tune,SEEDS[0],1,16)
    prediction,probability=network_predictions(fitted,torch.tensor(raw,device=device),tune)
    assert run["epochsRun"]==1 and prediction.shape==(16,12) and np.isfinite(prediction).all()
    assert probability.shape==(16,12) and np.isfinite(probability).all()
    log(f"SMOKE PASS device={device} causal prefix,forward,backward,indexed synthetic fit,predict; parameters={sum(p.numel() for p in model.parameters())}; no artifact writes")


def audit_completed():
    """Read-only model checks and descriptive uncertainty; never fits a model."""
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    by_interval={r["interval"]:r for r in summary["intervals"]}
    if set(by_interval)!={"15m","1h","4h","1d"}:
        raise AssertionError("Four completed interval summaries required")
    integrity=json.loads((OUT/"deployment-integrity.json").read_text(encoding="utf-8"))
    current={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"public/models").glob("b3-*.json")}
    results=[]
    for interval in ("15m","1h","4h","1d"):
        folder=OUT/interval
        report=json.loads((folder/"report.json").read_text(encoding="utf-8"))
        selection=json.loads((folder/"selection-frozen.json").read_text(encoding="utf-8"))
        fixture=json.loads((folder/"parity.json").read_text(encoding="utf-8"))
        predictor=NeuralPredictor(folder)
        manifest=predictor.manifest
        assert datetime.fromisoformat(report["protocol"]["startedAt"])<datetime.fromisoformat(selection["frozenAt"])
        assert datetime.fromisoformat(selection["frozenAt"])<datetime.fromisoformat(manifest["frozenAt"])
        assert set(map(int,selection["runs"]))==set(SEEDS)
        assert len(report["finalEvaluation"])==12
        for run in selection["runs"].values():
            assert run["epochsRun"]==len(run["history"]) and 1<=run["bestEpoch"]<=run["epochsRun"]
        replay=predictor.predict_features(fixture["rawFeatureWindows"],fixture["vol"])
        return_error=float(np.max(np.abs(replay["logReturns"]-np.asarray(fixture["logReturns"]))))
        probability_error=float(np.max(np.abs(replay["upProbability"]-np.asarray(fixture["upProbability"]))))
        assert return_error<1e-6 and probability_error<1e-5
        assert np.all(replay["lowerLogReturns"]<=replay["upperLogReturns"])
        assert np.all((replay["upProbability"]>=0)&(replay["upProbability"]<=1))
        rng=np.random.default_rng(SEEDS[0])
        x=torch.tensor(rng.normal(size=(2,SEQUENCE,len(manifest["features"]))),dtype=torch.float32)
        causal_errors=[]
        with torch.no_grad():
            for model in predictor.models:
                full=model.encode(x)
                for length in (17,31,47):
                    prefix=model.encode(x[:,:length])
                    error=float((full[:,:length]-prefix).abs().max())
                    assert torch.allclose(full[:,:length],prefix,rtol=1e-5,atol=2e-6)
                    causal_errors.append(error)
        data=get_dataset(interval)
        test=data["test"]
        # Direct saved-model inference on the unchanged final partition must
        # reproduce its already-recorded metrics. No candidate is refitted.
        mu,probability=ensemble_predictions(predictor.models,torch.from_numpy(data["features"]),test)
        probability=apply_calibration(probability,manifest["probabilityCalibration"])
        pred=mu*test["vol"][:,None]*np.sqrt(HORIZONS)
        reproduced=metric(test["y"][:,0],pred[:,0],probability[:,0])
        expected=report["finalEvaluation"][0]["metrics"]
        metric_error=max(abs(reproduced[k]-expected[k]) for k in
                         ("returnMse","returnR2VsZero","returnIc","balancedAccuracy","brier","priceMaePct"))
        assert metric_error<1e-8
        # Check real-candle feature prefixes and the public raw-candle API.
        source=data["metadata"]["sources"]["BTCUSDT"]
        rows=json.loads((ROOT/source["path"]).read_text(encoding="utf-8"))
        assert hashlib.sha256((ROOT/source["path"]).read_bytes()).hexdigest()==source["sha256"]
        candles=np.asarray([[int(r[0]),*map(float,r[1:6]),float(r[7]),float(r[8]),float(r[9]),float(r[10])] for r in rows])
        anchor=int(np.flatnonzero(candles[:,0]==fixture["origin"][0])[0])
        all_features,all_vol,names=candle_features(candles)
        prefix_features,prefix_vol,_=candle_features(candles[:anchor+1])
        np.testing.assert_allclose(all_features[anchor-63:anchor+1],prefix_features[-64:],rtol=1e-10,atol=1e-10)
        assert np.isclose(all_vol[anchor],prefix_vol[-1])
        api_full=predictor.predict_candles(candles[:anchor+1])
        api_short=predictor.predict_candles(candles[anchor-162:anchor+1])
        candle_api_error=float(np.max(np.abs(api_full["logReturns"]-api_short["logReturns"])))
        assert candle_api_error<1e-6
        # Descriptive paired bootstrap: keep all contemporaneous symbols
        # together. Exact-zero returns are excluded from directional metrics.
        _,inverse=np.unique(test["time"],return_inverse=True)
        blocks=inverse//48
        y=test["y"][:,0];p=probability[:,0];prediction=pred[:,0]
        prior=report["finalEvaluation"][0]["baselines"]["trainingDirectionPrior"]
        stats=[]
        for block in np.unique(blocks):
            keep=blocks==block;yb=y[keep];pb=p[keep];rb=prediction[keep]
            directional=yb!=0;positive=yb>0;negative=yb<0
            stats.append([np.sum(yb*yb),np.sum((rb-yb)**2),directional.sum(),
                          np.sum((((prior-positive)**2)-((pb-positive)**2))[directional]),
                          positive.sum(),np.sum((pb>=.5)&positive),negative.sum(),np.sum((pb<.5)&negative)])
        stats=np.asarray(stats,dtype=np.float64)
        draws=stats[rng.integers(0,len(stats),size=(500,len(stats)))].sum(axis=1)
        r2=1-draws[:,1]/np.maximum(draws[:,0],1e-15)
        brier_edge=draws[:,3]/np.maximum(draws[:,2],1)
        ba=.5*(draws[:,5]/np.maximum(draws[:,4],1)+draws[:,7]/np.maximum(draws[:,6],1))
        uncertainty={"method":"500 paired bootstrap draws of nonoverlapping48-origin timestamp blocks, all4 symbols kept together; descriptive on already inspected historical period",
                     "blocks":len(stats),"blockBars":48,"returnR2VsZero95":np.quantile(r2,[.025,.975]),
                     "brierImprovementVsTrainPrior95":np.quantile(brier_edge,[.025,.975]),
                     "balancedAccuracy95":np.quantile(ba,[.025,.975])}
        result={"interval":interval,"completed":True,"selected":selection["selected"],
                "capabilities":manifest["capabilities"],"memberFiles":manifest["members"],
                "initialRuns":{s:{"epochsRun":r["epochsRun"],"selectedEpoch":r["bestEpoch"]} for s,r in selection["runs"].items()},
                "refitEpochs":selection["refitEpochs"],"durationSeconds":report["durationSeconds"],
                "checks":{"memberSha256":True,"all12HorizonsPresent":True,"protocolBeforeSelectionBeforeManifest":True,
                          "savedReturnMaxError":return_error,"savedProbabilityMaxError":probability_error,
                          "causalEncoderPrefixMaxError":max(causal_errors),"reportedMetricMaxError":metric_error,
                          "realCandleFeaturePrefix":True,"fullVs163CandleApiMaxReturnError":candle_api_error},
                "finalH1":expected,"baselines":report["finalEvaluation"][0]["baselines"],
                "uncertainty":uncertainty}
        results.append(result)
        log(f"AUDIT PASS {interval} reload={return_error:.3g}, metric={metric_error:.3g}, causal={max(causal_errors):.3g}, rawCandleApi={candle_api_error:.3g}")
    artifact={"auditedAt":now(),"trainingRepeated":False,"newCandidateSelection":False,
              "deploymentIntegrityAtTrainingCompletion":integrity["unchanged"],
              "originalDeployedWeightsStillUnchanged":current==integrity["after"],"intervals":results}
    save(OUT/"completion-audit.json",artifact)
    lines=["# B4 神经网络四周期完成审计", "", "四个周期已真实完成训练、验证选模、固定轮数重训、独立校准及历史留出复评。本次只重载已保存权重核对推理、因果性和报告，没有重新训练或按测试结果挑选候选。", "",
           "模型：113,433参数，64根历史×57个因果特征，残差因果TCN+注意力汇聚，12个收益头与12个方向概率头。各周期两个固定种子使用CPU6线程训练。", "",
           "| 周期 | 验证收益/方向准入 | 留出方向BA | Brier/训练先验 | 收益R2相对持平 | 模型MAE/持平MAE | 耗时 |",
           "|---|---|---:|---|---:|---|---:|"]
    for r in results:
        m=r["finalH1"];b=r["baselines"];c=r["capabilities"]
        lines.append(f"|{r['interval']}|{c['returnEligible']}/{c['directionEligible']}|{m['balancedAccuracy']:.2%}|{m['brier']:.6f}/{b['trainMean']['brier']:.6f}|{m['returnR2VsZero']:.6f}|{m['priceMaePct']:.5f}%/{b['zero']['priceMaePct']:.5f}%|{r['durationSeconds']/60:.2f}分|")
    lines += ["", "价格/收益预测方面：四个周期都没有通过预声明收益准入，最终收益MSE均未超过持平基线；因此不建议把TCN均值价格曲线作为更准的升级上线。击败‘上一根收益延续’这种弱基线不足以证明超越持平或训练均值。", "",
              "方向概率方面：15min与1h通过三个验证时间块的方向门槛，留出平衡准确率分别约52.36%和51.77%，Brier略优于训练先验；可以保留为独立方向概率的研究/观察候选。4h和1d没有通过方向门槛，日线校准后Brier还差于训练先验，建议保持未上线。不能把方向分类的微弱优势表述为幅度或精确价格路径准确。", "",
              "这份判断没有更换任何候选、权重或校准参数。三个验证块评估的是同一个早期模型，与树分支逐折扩展训练方式不同，不应把两者当成完全相同训练流程。末段历史与前面实验存在重合，不能称为新的从未查看过的独立测试。", "",
              "## 时序相关性与不确定性", "", "下表为同一已查看历史上的描述性95%区间：按48个起点时间为块同步抽取四币，500次配对bootstrap；不是未来效果保证，也没有覆盖训练/模型选择不确定性。日线仅5块，区间尤其不稳定。", "",
              "| 周期 | 方向BA 95% | Brier改进相对先验95% | 收益R2 95% |", "|---|---|---|---|"]
    for r in results:
        u=r["uncertainty"];ba=u["balancedAccuracy95"];br=u["brierImprovementVsTrainPrior95"];rr=u["returnR2VsZero95"]
        lines.append(f"|{r['interval']}|{ba[0]:.2%}–{ba[1]:.2%}|{br[0]:.6f}–{br[1]:.6f}|{rr[0]:.6f}–{rr[1]:.6f}|")
    lines += ["", "## 可重载接口及检查", "",
              "`NeuralPredictor(folder)` 会验证各 `.pt` 文件SHA256，并使用 `weights_only=True` 加载。`predict_candles(candles)` 接收至少163根已闭合K线，列顺序为时间、OHLC、base量、quote量、成交笔数、主动买入base量、主动买入quote量。`predict_features(raw_windows,vol)` 接收[N,64,57]原始特征及每窗口因果波动率；返回每根h1..h12的logReturns、upProbability、lowerLogReturns和upperLogReturns。", "",
              "四周期均通过：冻结时间顺序、原始两种子训练历史、12步产物完整性、保存权重SHA、跨重载收益与概率一致性、原报告指标复现、TCN编码器前缀因果性、真实K线特征前缀一致性、完整历史与最短163根输入的一致性。精确误差见completion-audit.json。训练阶段始终未改原B3部署权重。", "",
              "```powershell", ".\\.venv-training\\Scripts\\python.exe scripts/train-b4-neural.py --audit --threads 6", "```"]
    (OUT/"completion-audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def benchmark_devices(args):
    """Transient optimizer steps on copied weights; never saves trained tensors."""
    if not torch.cuda.is_available():
        raise ValueError("GPU comparison requires the CUDA venv; torch.cuda.is_available() is false")
    folder=OUT/"1h"
    protected=list(OUT.glob("*/seed-*.pt"))+list(OUT.glob("*/manifest.json"))+list((ROOT/"public/models").glob("b3-*.json"))
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    data=get_dataset("1h");part=data["train"]
    batch=min(args.batch_size,len(part["y"]))
    anchors=part["row_index"][:batch]
    indices=anchors[:,None]+np.arange(1-SEQUENCE,1)
    input_np=data["features"][indices]
    target_np=np.clip(part["y_norm"][:batch],-8,8).astype(np.float32)
    label_np=(part["y"][:batch]>0).astype(np.float32)
    nonzero_np=(part["y"][:batch]!=0).astype(np.float32)
    frozen=NeuralPredictor(folder)
    state={k:v.detach().cpu().clone() for k,v in frozen.models[0].state_dict().items()}
    results=[]
    for device_name in ("cpu","cuda"):
        device=resolve_device(device_name)
        model=B4TCN(len(data["feature_names"])).to(device)
        model.load_state_dict(state)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.02)
        x=torch.tensor(input_np,device=device);target=torch.tensor(target_np,device=device)
        labels=torch.tensor(label_np,device=device);nonzero=torch.tensor(nonzero_np,device=device)
        weights=torch.tensor(LOSS_WEIGHTS,device=device)
        def sync():
            if device.type=="cuda":torch.cuda.synchronize(device)
        def step():
            optimizer.zero_grad(set_to_none=True)
            mu,logits=model(x)
            regression=((mu-target)**2*weights).sum(dim=1).mean()
            classification=(F.binary_cross_entropy_with_logits(logits,labels,reduction="none")*nonzero*weights).sum()/torch.clamp((nonzero*weights).sum(),min=1)
            loss=.4*regression+.6*classification
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            return loss
        model.train()
        for _ in range(5):step()
        sync();times=[]
        for repeat in range(3):
            sync();started=time.perf_counter()
            for _ in range(10):loss=step()
            sync();times.append(time.perf_counter()-started)
        finite=bool(torch.isfinite(loss).item()) and all(bool(torch.isfinite(p).all().item()) for p in model.parameters())
        assert finite
        median=float(np.median(times))
        result={"device":device_name,"cpuThreads":args.threads,"batchSize":batch,
                "sequence":SEQUENCE,"featureCount":len(data["feature_names"]),
                "warmupSteps":5,"stepsPerRepeat":10,"repeatSeconds":times,
                "medianSamplesPerSecond":batch*10/median,"medianMillisecondsPerStep":median*1000/10,
                "lossAndParametersFinite":finite}
        results.append(result)
        log(f"BENCHMARK {device_name} batch={batch} {result['medianSamplesPerSecond']:.1f} samples/s {result['medianMillisecondsPerStep']:.2f} ms/step")
        del model,optimizer,x,target,labels,nonzero,weights
        if device.type=="cuda":torch.cuda.empty_cache()
    # Cross-device inference uses the original saved weights, independently of
    # transient benchmark updates. No CPU or GPU model candidate is selected.
    raw=data["raw_features"][indices]
    cpu_output=frozen.predict_features(raw,part["vol"][:batch])
    gpu_output=NeuralPredictor(folder,device="cuda").predict_features(raw,part["vol"][:batch])
    errors={k:float(np.max(np.abs(cpu_output[k]-gpu_output[k]))) for k in cpu_output}
    assert errors["logReturns"]<1e-6
    assert errors["upProbability"]<1e-5
    assert errors["lowerLogReturns"]<1e-6 and errors["upperLogReturns"]<1e-6
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before==after,"Existing weight or manifest file changed during benchmark"
    report={"measuredAt":now(),"interval":"1h","pytorch":str(torch.__version__),"cudaVersion":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0),"gpuMemoryBytes":torch.cuda.get_device_properties(0).total_memory,
            "precision":"float32; TF32 disabled; cudnn deterministic; AdamW+gradient clipping and both exact training losses",
            "method":"Same frozen initial weights and same real training batch; five warmup and3x10 timed optimization steps on ephemeral copies. Batch already on target device; no data downloading, epoch selection, calibration or artifact fitting.",
            "results":results,"gpuThroughputRelativeToCpu6":results[1]["medianSamplesPerSecond"]/results[0]["medianSamplesPerSecond"],
            "frozenWeightCrossDeviceMaximumAbsoluteError":errors,"tolerances":{"logReturns":1e-6,"upProbability":1e-5},
            "existingWeightsAndManifestsUnchanged":True,"protectedHashesBefore":before,"protectedHashesAfter":after,
            "limitation":"Microbenchmark excludes data preparation, per-batch indexing, validation and per-step logging synchronization; it does not imply accuracy improvement or the same speedup for full training or single-candle live inference."}
    save(OUT/"device-benchmark.json",report)
    lines=["# B4 TCN CPU / CUDA 算力与重载一致性实测", "",
           f"PyTorch {torch.__version__}，CUDA {torch.version.cuda}，GPU {report['gpu']}。1h已保存模型、同一真实训练批次：{batch}样本×64根×57特征，113,433参数，float32，关闭TF32。CPU使用{args.threads}计算线程。", "",
           "仅在内存中的临时副本做前向、损失、反向、梯度裁剪和AdamW更新；预热5步，再重复3组各10步取中位数。没有重训已完成四周期，没有保存测量产生的参数。", "",
           "| 设备 | 样本/秒 | 毫秒/训练步 |", "|---|---:|---:|"]
    for r in results:lines.append(f"|{r['device']}|{r['medianSamplesPerSecond']:.1f}|{r['medianMillisecondsPerStep']:.2f}|")
    lines += ["",f"此批次GPU吞吐为CPU6的 **{report['gpuThroughputRelativeToCpu6']:.2f}倍**。GPU环境可用不等于预测精度改善。此测量不包含数据准备、训练采样索引、验证评估和每步日志同步，不等于完整训练或单根实时推理的加速倍数。", "",
              f"原始冻结1h权重在CPU与CUDA的最大绝对误差：logReturn={errors['logReturns']:.3g}；上涨概率={errors['upProbability']:.3g}；上下界分别{errors['lowerLogReturns']:.3g}/{errors['upperLogReturns']:.3g}。通过预设收益1e-6、概率1e-5容差。四周期权重/manifest及原B3权重前后SHA256均完全一致。", "",
              "脚本现支持 `--device auto/cpu/cuda`。`NeuralPredictor(folder, device='cuda')` 可显式进行GPU推理，默认仍为CPU。", "",
              "```powershell", ".\\.venv-training-cuda\\Scripts\\python.exe scripts/train-b4-neural.py --benchmark --threads 6 --batch-size 256", "```"]
    (OUT/"device-benchmark.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals",nargs="+",choices=["1h","15m","4h","1d"],default=["1h","15m","4h","1d"])
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--threads",type=int,default=6)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--audit",action="store_true",help="Audit completed four-period artifacts; never trains")
    parser.add_argument("--benchmark",action="store_true",help="Measure transient CPU/GPU training steps on1h; never changes saved weights")
    args=parser.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(2)
    try:args.device=resolve_device(args.device)
    except ValueError as error:parser.error(str(error))
    if args.smoke:smoke(args.device);return
    if args.audit:audit_completed();return
    if args.benchmark:benchmark_devices(args);return
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"public/models").glob("b3-*.json")}
    results=[]
    for interval in args.intervals:
        results.append(run_interval(interval,args));save(OUT/"summary.json",{"intervals":results})
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"public/models").glob("b3-*.json")}
    assert before==after,"Deployed weights changed"
    save(OUT/"deployment-integrity.json",{"before":before,"after":after,"unchanged":True})


if __name__=="__main__":
    main()

"""Predict future absolute cumulative return independently of its direction.

The result is an unsigned expected move, never a deterministic signed price
path. Every future bar has a direct trained head. Tree selection uses expanding
validation only and no deployed artifacts are modified.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import time
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('trees',ROOT/'scripts/train-b4-trees.py')
trees=importlib.util.module_from_spec(spec)
spec.loader.exec_module(trees)
OUT=ROOT/'reports/experiments/b4-magnitude'


def assess(part,h,pred):
    y=np.abs(part['y'][:,h-1])
    def calc(mask):
        true,estimate=y[mask],pred[mask]
        return {'samples':len(true),'mse':float(np.mean((true-estimate)**2)),
                'maePct':float(np.mean(np.abs(true-estimate))*100),
                'actualMeanPct':float(np.mean(true)*100),'predictionMeanPct':float(np.mean(estimate)*100),
                'ic':float(np.corrcoef(true,estimate)[0,1]) if np.std(estimate)>1e-12 and np.std(true)>1e-12 else 0,
                'stdRatio':float(np.std(estimate)/max(np.std(true),1e-12))}
    result=calc(np.ones(len(y),dtype=bool))
    result['bySymbol']={symbol:calc(part['symbol']==i) for i,symbol in enumerate(trees.SYMBOLS) if np.any(part['symbol']==i)}
    return result


def fit(config,x,part,h,iterations,threads,ex=None,ep=None):
    model=trees.build_model(config,'return',iterations,threads)
    y=np.clip(np.abs(part['y_norm'][:,h-1]),0,8)
    kwargs={}
    if ep is not None:
        kwargs={'eval_set':(ex,np.clip(np.abs(ep['y_norm'][:,h-1]),0,8)),
                'early_stopping_rounds':60,'use_best_model':True}
    model.fit(x,y,**kwargs)
    return model


def predict(model,x,part,h):
    return np.maximum(0,model.predict(x,thread_count=6))*part['vol']*np.sqrt(h)


def baselines(data,train,part,h):
    vol_scaled=float(np.mean(np.abs(train['y_norm'][:,h-1])))*part['vol']*np.sqrt(h)
    mean=np.full(len(part['y']),float(np.mean(np.abs(train['y'][:,h-1]))))
    retcol=data['feature_names'].index('logReturn1')
    historical={}
    for n in [20,48]:
        ix=part['row_index'][:,None]+np.arange(1-n,1)
        historical[f'rollingAbsReturn{n}']=np.mean(np.abs(data['raw_features'][ix,retcol]),axis=1)*np.sqrt(h)
    values={'trainMean':mean,'causalVolTrainScale':vol_scaled,
            'causalVolGaussian':part['vol']*np.sqrt(h)*np.sqrt(2/np.pi),**historical}
    return {name:assess(part,h,pred) for name,pred in values.items()}


def bootstrap(part,pred,baseline,draws=500,block_bars=48):
    times,ix=np.unique(part['time'],return_inverse=True)
    blocks=ix//block_bars
    y=np.abs(part['y'][:,0])
    a=[]
    for b in np.unique(blocks):
        mask=blocks==b
        a.append([mask.sum(),np.sum((y[mask]-pred[mask])**2),np.sum((y[mask]-baseline[mask])**2),
                  np.sum(np.abs(y[mask]-pred[mask])),np.sum(np.abs(y[mask]-baseline[mask]))])
    a=np.asarray(a)
    rng=np.random.default_rng(trees.SEED)
    sampled=a[rng.integers(0,len(a),(draws,len(a)))].sum(axis=1)
    return {'method':'48-origin-bar timestamp block bootstrap; symbols kept together','draws':draws,
            'relativeMseImprovement95':np.quantile(1-sampled[:,1]/sampled[:,2],[.025,.975]).tolist(),
            'relativeMaeImprovement95':np.quantile(1-sampled[:,3]/sampled[:,4],[.025,.975]).tolist()}


def run(interval,args):
    started=time.monotonic()
    folder=OUT/interval
    folder.mkdir(parents=True,exist_ok=True)
    data=trees.get_dataset(interval)
    protocol={'startedAt':trees.now(),'interval':interval,'configs':trees.CONFIGS,'maxIterations':args.iterations,
              'heads':'E abs(cumulative log return h1..h12), divided by causal48barVol*sqrt(h)',
              'noDirectionClaim':True,'featureContract':'Same 57 causal raw features + four symbol one-hot as b4-trees',
              'cv':trees.FOLDS,'selection':'h1 pooled fold score = .5 MSE/causalVolTrainScaleMSE + .5 MAE/causalVolTrainScaleMAE',
              'gate':'Positive average MSE and MAE improvement vs causalVolTrainScale; at least2/3 folds improve MSE; no test selection',
              'baselines':['training mean absolute target','causal origin sigma48 x training mean absolute normalized target','rolling absolute return20 x sqrt(h)','rolling absolute return48 x sqrt(h)'],
              'finalFit':'All pre-calibration partitions, selected config/median earlystop tree count, all12 heads',
              'calibration':'Scale normalized absolute residual90th quantile from separate calibration partition',
              'testStatus':'Historical reevaluation, overlaps previously inspected time; no fresh untouched evidence',
              'data':data['metadata']}
    trees.save(folder/'protocol.json',protocol)
    candidates=[]
    for config in trees.CONFIGS:
        folds=[]
        for fi,(tr,tun,v) in enumerate(trees.FOLDS):
            train=trees.combine([data[n] for n in tr]); tune=data[tun]; val=data[v]
            x,tx,vx=(trees.feature_matrix(data,p) for p in (train,tune,val))
            trees.log(f'MAGNITUDE {interval} {config["name"]} fold{fi+1} train={len(x)}')
            model=fit(config,x,train,1,args.iterations,args.threads,tx,tune)
            prediction=predict(model,vx,val,1)
            metrics=assess(val,1,prediction)
            bs=baselines(data,train,val,1)
            ref=bs['causalVolTrainScale']
            score=.5*metrics['mse']/ref['mse']+.5*metrics['maePct']/ref['maePct']
            folds.append({'fold':fi+1,'metrics':metrics,'baselines':bs,'iterations':model.tree_count_,'score':score,
                          'mseImprovement':1-metrics['mse']/ref['mse'],'maeImprovement':1-metrics['maePct']/ref['maePct']})
            trees.log(f'MAGNITUDE CV {interval} fold{fi+1} trees={model.tree_count_} MSEimprovement={folds[-1]["mseImprovement"]:.4f} MAEimprovement={folds[-1]["maeImprovement"]:.4f}')
        mse=np.mean([f['mseImprovement'] for f in folds]); mae=np.mean([f['maeImprovement'] for f in folds])
        eligible=mse>0 and mae>0 and sum(f['mseImprovement']>0 for f in folds)>=2
        candidates.append({'config':config,'folds':folds,'score':float(np.mean([f['score'] for f in folds])),
                           'mseImprovement':float(mse),'maeImprovement':float(mae),'eligible':bool(eligible)})
    selected=min(candidates,key=lambda c:c['score'])
    count=int(np.median([f['iterations'] for f in selected['folds']]))
    frozen={'frozenAt':trees.now(),'selected':selected['config'],'iterations':count,'eligible':selected['eligible'],'candidates':candidates}
    trees.save(folder/'selection-frozen.json',frozen)
    train=trees.combine([data[n] for n in ['train','tune','val1','val2','val3']])
    cal,test=data['calibration'],data['test']
    x,cx,ex=(trees.feature_matrix(data,p) for p in (train,cal,test))
    results,artifacts,outputs={},{},{}
    for h in range(1,13):
        trees.log(f'MAGNITUDE FINAL {interval} h{h} trees={count}')
        model=fit(selected['config'],x,train,h,count,args.threads)
        model.save_model(str(folder/f'h{h}-magnitude.cbm'))
        model.save_model(str(folder/f'h{h}-magnitude.json'),format='json')
        cp=predict(model,cx,cal,h)
        residual=(np.abs(cal['y'][:,h-1])-cp)/(cal['vol']*np.sqrt(h))
        calibration={'frozenAt':trees.now(),'normalizedSignedResidualQuantiles':np.quantile(residual,[.1,.5,.9]).tolist()}
        trees.save(folder/f'h{h}-calibration-frozen.json',calibration)
        predicted=predict(model,ex,test,h)
        measured=assess(test,h,predicted)
        bs=baselines(data,train,test,h)
        measured['mseImprovementVsVolBaseline']=1-measured['mse']/bs['causalVolTrainScale']['mse']
        measured['maeImprovementVsVolBaseline']=1-measured['maePct']/bs['causalVolTrainScale']['maePct']
        results[f'h{h}']={'test':measured,'baselines':bs,'calibration':calibration}
        if h==1:
            baseline=float(np.mean(np.abs(train['y_norm'][:,0])))*test['vol']
            results['h1']['bootstrap']=bootstrap(test,predicted,baseline)
        path=folder/f'h{h}-magnitude.cbm'
        artifacts[f'h{h}']={'cbm':path.name,'json':f'h{h}-magnitude.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        outputs[f'pred_h{h}']=predicted
        trees.log(f'MAGNITUDE TEST {interval} h{h} MSEimprovement={measured["mseImprovementVsVolBaseline"]:.4f} MAEimprovement={measured["maeImprovementVsVolBaseline"]:.4f}')
    contract={'schemaVersion':1,'family':'b4-catboost-expected-absolute-move','interval':interval,'horizons':list(range(1,13)),
              'features':data['feature_names']+[f'symbol_{s}' for s in trees.SYMBOLS],
              'targetMeaning':'Expected absolute cumulative log return; unsigned, cannot be used as predicted signed path',
              'decode':'max(rawModelOutput,0)*causalOriginVolatility48*sqrt(h)',
              'models':artifacts,'calibration':{h:r['calibration'] for h,r in results.items()},
              'validationEligible':selected['eligible'],'selection':frozen}
    trees.save(folder/'model-contract.json',contract)
    np.savez_compressed(folder/'test-predictions.npz',time=test['time'],symbol=test['symbol'],truth=test['y'],vol=test['vol'],**outputs)
    report={'interval':interval,'finishedAt':trees.now(),'seconds':time.monotonic()-started,'selection':frozen,'results':results}
    trees.save(folder/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals',nargs='+',default=['1h','15m','4h','1d'])
    parser.add_argument('--iterations',type=int,default=600)
    parser.add_argument('--threads',type=int,default=6)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[]
    for interval in args.intervals:
        reports.append(run(interval,args))
    write_report()


def write_report():
    reports=[json.loads((OUT/i/'report.json').read_text(encoding='utf-8'))
             for i in ['15m','1h','4h','1d'] if (OUT/i/'report.json').exists()]
    lines=['# B4 unsigned absolute-move prediction','',
           'All four next-bar magnitude models beat the prespecified volatility-scaled baseline and both rolling-absolute-return baselines on this historical reevaluation. This is improvement in forecasting **how much price may move**, not a validated prediction of the signed future price path.','',
           'Each interval has 12 independently trained heads. Input is the last fully closed candle and earlier candles. The h1 selection gate does not validate all later horizons. In particular, h12 on 4h has worse MAE and h12 on 1d has worse MSE and MAE than the volatility baseline.','',
           'All model choices were frozen from three expanding temporal validation folds; calibration and final evaluation occur later in time. The final dates overlap previously inspected experiments and are not fresh untouched evidence. The tables below do not choose or replace any model.','',
           '## Next-bar results','',
           '| Interval | Observations / distinct times | CV passed | MSE reduction vs vol baseline | MAE reduction vs vol baseline | Absolute-move IC | Model magnitude MAE, pp |',
           '|---|---:|---|---:|---:|---:|---:|']
    for r in reports:
        h=r['results']['h1']; m=h['test']; times=r.get('uncertaintyLimitations',{}).get('uniqueOriginTimes',m['samples']//4)
        lines.append(f"| {r['interval']} | {m['samples']} / {times} | {r['selection']['eligible']} | {m['mseImprovementVsVolBaseline']:.2%} | {m['maeImprovementVsVolBaseline']:.2%} | {m['ic']:.4f} | {m['maePct']:.5f} |")
    lines+=['','The volatility baseline is origin sigma48 multiplied by the training-only mean normalized absolute target. MAE is error in absolute log-return magnitude, expressed in percentage points; it must not be reported as an improvement in signed-price MAE.','',
            '## Stronger and simpler comparator audit','',
            'Every cell is relative MSE reduction / relative MAE reduction. Positive means lower model error. No comparator is selected to alter predictions.','',
            '| Interval | Training mean | Volatility Gaussian factor | Rolling abs return20 | Rolling abs return48 |','|---|---:|---:|---:|---:|']
    for r in reports:
        h=r['results']['h1']; m=h['test']; cells=[]
        for key in ['trainMean','causalVolGaussian','rollingAbsReturn20','rollingAbsReturn48']:
            b=h['baselines'].get(key)
            cells.append(f"{1-m['mse']/b['mse']:.2%} / {1-m['maePct']/b['maePct']:.2%}" if b else 'pending verification')
        lines.append(f"| {r['interval']} | "+' | '.join(cells)+' |')
    lines+=['','## Longer-horizon boundary','',
            '| Interval | Horizon (bars) | MSE reduction vs vol baseline | MAE reduction vs vol baseline | Absolute-move IC |','|---|---:|---:|---:|---:|']
    for r in reports:
        for horizon in [3,6,12]:
            m=r['results'][f'h{horizon}']['test']
            lines.append(f"| {r['interval']} | {horizon} | {m['mseImprovementVsVolBaseline']:.2%} | {m['maeImprovementVsVolBaseline']:.2%} | {m['ic']:.4f} |")
    lines+=['','## Uncertainty and consistency','',
            'Block bootstrap keeps all contemporaneous symbols together and resamples 48-bar time blocks, 500 draws. These are descriptive intervals for an inspected period. The daily evaluation contains only 197 distinct origin times (five 48-bar blocks); its interval is especially weak evidence about future market regimes.','',
            '| Interval | h1 MSE reduction vs vol: bootstrap 95% | h1 MSE reduction vs rolling abs20: bootstrap 95% | Coins improving MSE / MAE vs rolling abs20 |',
            '|---|---:|---:|---:|']
    for r in reports:
        h=r['results']['h1']; a=h['bootstrap']['relativeMseImprovement95']; b=h.get('strongBaselineBootstrap',{}).get('rollingAbsReturn20',{}).get('relativeMseImprovement95')
        m=h['test']['bySymbol']; base=h['baselines']['rollingAbsReturn20']['bySymbol']
        counts=(sum(m[s]['mse']<base[s]['mse'] for s in m),sum(m[s]['maePct']<base[s]['maePct'] for s in m))
        btext=f"[{b[0]:.2%}, {b[1]:.2%}]" if b else 'pending verification'
        lines.append(f"| {r['interval']} | [{a[0]:.2%}, {a[1]:.2%}] | {btext} | {counts[0]}/4 / {counts[1]}/4 |")
    lines+=['','## Runtime contract and UI interpretation','',
            '- Module: `scripts/train-b4-magnitude.py`; load once with `load_models(interval)` and call `predict_features(loaded, raw_features, symbols, origin_volatility, last_close=None)`.',
            '- Inputs: raw causal features `[N,57]` in the frozen contract order; symbol indices `[N]` BTC=0, ETH=1, BNB=2, SOL=3; positive causal sigma48 `[N]`; optional last closed prices `[N]`.',
            '- `absLogReturns[N,12]` estimates unsigned absolute cumulative log return independently at every future bar. `expectedMovePct=100*expm1(absLogReturns)` is a display conversion, not the expected signed return or exactly the expected absolute simple return.',
            '- `cvPassed` and `heldoutSkill` concern **h1 only**. `skillScope` explicitly states this; `historicalDiagnosticsByHorizon` carries later-horizon results without selecting models from them.',
            '- Optional `upScenarioPrices` and `downScenarioPrices` equal `close*exp(+/-absLogReturn)`. Label them “±预估变动幅度情景”; they are neither confidence bounds nor conditional price paths. The up/down probability model is a separate model.',
            '- Suggested release: expose h1 expected magnitude as an experimental trained signal at every selected bar interval. Display later horizons as experimental magnitude scenarios with their evidence limitations. Keep signed mean-price forecasts separately identified as unverified; do not multiply direction probability and magnitude to manufacture a blue price line.',
            '', '## Artifact verification','',
            'Verification reloads all 48 magnitude heads, checks CBM SHA256 values, reproduces every stored test prediction, and checks nonnegative finite magnitudes. For each of 16 source series it also checks source SHA256 and three origin prefixes; changing every future candle leaves the origin features, volatility and model prediction unchanged. See `verification.json` and per-interval reports.', '']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def load_models(interval,experiment_root=OUT):
    folder=Path(experiment_root)/interval
    contract=json.loads((folder/'model-contract.json').read_text(encoding='utf-8'))
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    models={}
    for name, info in contract['models'].items():
        path=folder/info['cbm']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError(f'Magnitude model hash mismatch: {path}')
        model=trees.CatBoostRegressor()
        model.load_model(str(path))
        models[name]=model
    h1=report['results']['h1']
    holdout_skill=(h1['test']['mseImprovementVsVolBaseline']>0 and
                   h1['test']['maeImprovementVsVolBaseline']>0 and
                   h1['bootstrap']['relativeMseImprovement95'][0]>0)
    return {'models':models,'contract':contract,'heldoutSkill':bool(holdout_skill),
            'heldoutMetrics':h1['test'],'heldoutUncertainty':h1['bootstrap'],
            'historicalDiagnosticsByHorizon':{h:{'mseImprovementVsVolBaseline':r['test']['mseImprovementVsVolBaseline'],
                                                'maeImprovementVsVolBaseline':r['test']['maeImprovementVsVolBaseline'],
                                                'magnitudeIc':r['test']['ic']}
                                               for h,r in report['results'].items()}}


def predict_features(loaded,raw_features,symbols,origin_volatility,last_close=None):
    raw=np.atleast_2d(np.asarray(raw_features,dtype=np.float32))
    symbols=np.atleast_1d(np.asarray(symbols,dtype=int))
    vol=np.atleast_1d(np.asarray(origin_volatility,dtype=float))
    if raw.shape!=(len(symbols),len(loaded['contract']['features'])-4) or len(vol)!=len(raw):
        raise ValueError('Magnitude input feature contract mismatch')
    if (not np.isfinite(raw).all() or not np.isfinite(vol).all() or np.any(vol<=0)
            or np.any((symbols<0)|(symbols>=4))):
        raise ValueError('Invalid magnitude feature input')
    x=np.concatenate((raw,np.eye(4,dtype=np.float32)[symbols]),axis=1)
    preds=[]
    for h in loaded['contract']['horizons']:
        value=loaded['models'][f'h{h}'].predict(x,thread_count=1)
        preds.append(np.maximum(value,0)*vol*np.sqrt(h))
    absolute=np.stack(preds,axis=1)
    result={'absLogReturns':absolute,'expectedMovePct':np.expm1(absolute)*100,
            'cvPassed':loaded['contract']['validationEligible'],'heldoutSkill':loaded['heldoutSkill'],
            'skillScope':'h1 only; selection and bootstrap eligibility do not validate every later horizon',
            'metricMeaning':'Unsigned expected absolute cumulative log return; expectedMovePct=100*expm1(absLogReturn) for display, not expected signed return',
            'heldoutMetrics':loaded['heldoutMetrics'],
            'historicalDiagnosticsByHorizon':loaded['historicalDiagnosticsByHorizon']}
    if last_close is not None:
        close=np.atleast_1d(np.asarray(last_close,dtype=float))
        if len(close)!=len(raw) or np.any(close<=0) or not np.isfinite(close).all():
            raise ValueError('Invalid magnitude origin prices')
        result['upScenarioPrices']=close[:,None]*np.exp(absolute)
        result['downScenarioPrices']=close[:,None]*np.exp(-absolute)
        result['scenarioMeaning']='Symmetric +/- predicted unsigned log-return magnitude scenarios, not quantiles or calibrated directional paths'
    return result


def predict_from_dataset(loaded,data,part):
    return predict_features(loaded,data['raw_features'][part['row_index']],part['symbol'],part['vol'],part['close'])


if __name__=='__main__':
    main()

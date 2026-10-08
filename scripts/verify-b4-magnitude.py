"""Verify persisted unsigned-magnitude forecasts without further training."""
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
import b4_data

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('magnitude',ROOT/'scripts/train-b4-magnitude.py')
mag=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mag)


def causal_verification(interval,data,loaded):
    checks=[]
    for sid,symbol in enumerate(mag.trees.SYMBOLS):
        source=data['metadata']['sources'][symbol]
        path=ROOT/source['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
        rows=json.loads(path.read_text(encoding='utf-8'))
        candles=b4_data.validate_rows(rows,symbol,interval,data['metadata']['cutoff'])
        full,vol,names=b4_data.features(candles)
        assert names==data['feature_names']
        for anchor in [b4_data.ANCHOR_START,len(candles)//2,len(candles)-13]:
            prefix,prefix_vol,prefix_names=b4_data.features(candles[:anchor+1])
            np.testing.assert_allclose(prefix[-1],full[anchor],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(prefix_vol[-1],vol[anchor],rtol=1e-10,atol=1e-10)
            changed=candles.copy()
            changed[anchor+1:,1:5]*=1.7
            changed[anchor+1:,5:]*=2.3
            future_changed,changed_vol,_=b4_data.features(changed)
            np.testing.assert_array_equal(future_changed[anchor],full[anchor])
            np.testing.assert_array_equal(changed_vol[anchor],vol[anchor])
            p1=mag.predict_features(loaded,prefix[-1],[sid],[prefix_vol[-1]],[candles[anchor,4]])
            p2=mag.predict_features(loaded,future_changed[anchor],[sid],[changed_vol[anchor]],[candles[anchor,4]])
            np.testing.assert_array_equal(p1['absLogReturns'],p2['absLogReturns'])
            checks.append({'symbol':symbol,'anchor':anchor,'prefixAndFuturePerturbationInvariant':True})
    return checks


def main():
    summary=[]
    for interval in ['15m','1h','4h','1d']:
        folder=mag.OUT/interval
        if not (folder/'report.json').exists():
            continue
        loaded=mag.load_models(interval)
        data=mag.trees.get_dataset(interval)
        test=data['test']
        pred=mag.predict_from_dataset(loaded,data,test)
        cached=np.load(folder/'test-predictions.npz')
        expected=np.stack([cached[f'pred_h{h}'] for h in range(1,13)],axis=1)
        np.testing.assert_allclose(pred['absLogReturns'],expected,atol=1e-12,rtol=1e-12)
        assert pred['absLogReturns'].shape==(len(test['time']),12)
        assert (pred['absLogReturns']>=0).all()
        assert np.isfinite(pred['expectedMovePct']).all()
        assert np.all(pred['upScenarioPrices']>=test['close'][:,None])
        assert np.all(pred['downScenarioPrices']<=test['close'][:,None])
        # Additional fixed mathematical baseline requires no fitting/selection.
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        for h in range(1,13):
            gaussian=test['vol']*np.sqrt(h)*np.sqrt(2/np.pi)
            report['results'][f'h{h}']['baselines']['causalVolGaussian']=mag.assess(test,h,gaussian)
        col=data['feature_names'].index('logReturn1')
        strong_baseline_intervals={}
        for window in [20,48]:
            ix=test['row_index'][:,None]+np.arange(1-window,1)
            baseline=np.mean(np.abs(data['raw_features'][ix,col]),axis=1)
            strong_baseline_intervals[f'rollingAbsReturn{window}']=mag.bootstrap(test,pred['absLogReturns'][:,0],baseline)
        report['results']['h1']['strongBaselineBootstrap']=strong_baseline_intervals
        report['uncertaintyLimitations']={'uniqueOriginTimes':int(len(np.unique(test['time']))),
                                          'nonoverlapping48BarBlocks':int(np.ceil(len(np.unique(test['time']))/48)),
                                          'description':'Exploratory block-bootstrap intervals on inspected historical data; daily period has very few independent time blocks.'}
        causal_checks=causal_verification(interval,data,loaded)
        report['verification']={'fullArtifactReload':True,'all12Heads':True,'sourceHashes':True,
                                'causalPredictionChecks':causal_checks}
        mag.trees.save(folder/'report.json',report)
        contract=json.loads((folder/'model-contract.json').read_text(encoding='utf-8'))
        contract['validationScope']='h1 only; hyperparameters and tree count transfer to independent h2..h12 models'
        contract['unsignedMagnitudeScenarios']='close*exp(+/-expectedAbsLogReturn) are symmetric scenarios; not predicted signed paths or confidence bounds'
        contract['rawFeatureOrder']=data['feature_names']
        contract['symbolOrder']=mag.trees.SYMBOLS
        mag.trees.save(folder/'model-contract.json',contract)
        item={'interval':interval,'samples':len(test['time']),'directHeads':12,
              'maximumDifference':float(np.max(np.abs(pred['absLogReturns']-expected))),
              'cvPassed':pred['cvPassed'],'heldoutSkill':pred['heldoutSkill'],'hashesAndNonnegative':True,
              'skillScope':'h1','causalPredictionChecks':len(causal_checks)}
        summary.append(item)
        print(item,flush=True)
    mag.trees.save(mag.OUT/'verification.json',summary)
    mag.write_report()


if __name__=='__main__':
    main()

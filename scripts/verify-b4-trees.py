"""Artifact inference verification and descriptive diagnostics, no model refit."""
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
import b4_data

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b4_trees',ROOT/'scripts/train-b4-trees.py')
trees = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trees)


def main():
    results=[]
    for interval in ['15m','1h','4h','1d']:
        folder=trees.OUT/interval
        if not (folder/'report.json').exists():
            continue
        loaded=trees.load_models(interval)
        data=trees.get_dataset(interval)
        test=data['test']
        inferred=trees.predict_from_dataset(loaded,data,test)
        cached=np.load(folder/'test-predictions.npz')
        expected=np.stack([cached[f'pred_h{h}'] for h in range(1,13)],axis=1)
        np.testing.assert_allclose(inferred['logReturns'],expected,rtol=1e-12,atol=1e-12)
        for h in loaded['contract']['directionHorizons']:
            np.testing.assert_allclose(inferred['directionProbabilities'][f'h{h}'],cached[f'prob_h{h}'],rtol=1e-12,atol=1e-12)
        assert inferred['prices'].shape == (len(test['time']),12)
        assert np.all(inferred['lowerPrices'] < inferred['upperPrices'])
        assert np.isfinite(inferred['prices']).all()
        assert len([n for n in loaded['models'] if n.endswith('-return')]) == 12
        causal_checks=0
        for sid,symbol in enumerate(trees.SYMBOLS):
            info=data['metadata']['sources'][symbol]
            path=ROOT/info['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
            raw=json.loads(path.read_text(encoding='utf-8'))
            candles=b4_data.validate_rows(raw,symbol,interval,data['metadata']['cutoff'])
            full,vol,names=b4_data.features(candles)
            assert names==loaded['contract']['rawFeatureOrder']
            for anchor in [b4_data.ANCHOR_START,len(candles)//2,len(candles)-13]:
                prefix,prefix_vol,_=b4_data.features(candles[:anchor+1])
                changed=candles.copy()
                changed[anchor+1:,1:5]*=1.7
                changed[anchor+1:,5:]*=2.3
                altered,altered_vol,_=b4_data.features(changed)
                np.testing.assert_array_equal(prefix[-1],full[anchor])
                np.testing.assert_array_equal(altered[anchor],full[anchor])
                p1=trees.predict_features(loaded,prefix[-1],[sid],[prefix_vol[-1]],[candles[anchor,4]])
                p2=trees.predict_features(loaded,altered[anchor],[sid],[altered_vol[anchor]],[candles[anchor,4]])
                np.testing.assert_array_equal(p1['logReturns'],p2['logReturns'])
                for horizon in loaded['contract']['directionHorizons']:
                    np.testing.assert_array_equal(p1['directionProbabilities'][f'h{horizon}'],p2['directionProbabilities'][f'h{horizon}'])
                causal_checks+=1
        # Correct descriptive bootstrap after excluding exact zero returns in
        # direction statistics. This changes no predictions or selection.
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        h1=report['results']['h1']
        h1['bootstrap']=trees.temporal_bootstrap(test,inferred['logReturns'][:,0],
                                                inferred['directionProbabilities']['h1'],
                                                h1['baselines']['trainingDirectionPrior'])
        # Reversal is an additional diagnostic comparator, not a candidate or
        # a test-selected fallback. Its sign signal is opposite the last bar.
        col=list(data['feature_names']).index('logReturn1')
        past=data['raw_features'][test['row_index'],col]
        opposite=trees.assess(test,1,-past,(past < 0).astype(float))
        h1['baselines']['reversalDiagnostic']=opposite
        h1['baselines']['reversalDiagnosticNote']='Added after final predictions solely as a diagnostic; no candidate, gate or hyperparameter changes.'
        trees.save(folder/'report.json',report)
        results.append({'interval':interval,'samples':len(test['time']),'regressionHeads':12,
                        'maximumReturnDifference':float(np.max(np.abs(inferred['logReturns']-expected))),
                        'modelHashVerification':True,'inferenceShapeAndFinite':True,
                        'prefixAndFuturePerturbationCausalityChecks':causal_checks})
        print(interval,results[-1],flush=True)
    trees.save(trees.OUT/'verification.json',results)
    trees.write_report([])


if __name__=='__main__':
    main()

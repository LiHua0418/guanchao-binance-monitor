"""Read-only audit of frozen B4 BTC replay versus fixed-origin forecasts.

No network requests, fitting, candidate selection, or model writes. All periods
use the last up-to-240 common origins with observable h1 AND h12 outcomes after
the fit/calibration time boundary. This is historical reconstruction, not a
record of forecasts issued live after the artifact creation time.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

import b4_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/b4-replay-audit'
SPEC = importlib.util.spec_from_file_location('b4_replay_service', ROOT / 'scripts/serve-b4.py')
SERVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVICE)


def iso(value):
    return datetime.fromtimestamp(float(value) / 1000, timezone.utc).isoformat().replace('+00:00', 'Z')


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def direction(y, prediction):
    nonzero = y != 0
    truth, pred = np.sign(y[nonzero]), np.asarray(prediction)[nonzero]
    recalls = [float(np.mean(pred[truth == s] == s)) for s in (-1, 1) if np.any(truth == s)]
    return {'samples': int(nonzero.sum()), 'accuracy': float(np.mean(pred == truth)),
            'balancedAccuracy': float(np.mean(recalls)) if len(recalls) == 2 else None,
            'positivePredictionFraction': float(np.mean(pred > 0)),
            'abstentionFraction': float(np.mean(pred == 0)),
            'actualUpFraction': float(np.mean(truth > 0))}


def metrics(origin, actual, predicted_return, probability):
    y = np.log(actual / origin)
    predicted = origin * np.exp(predicted_return)
    model_mae = float(np.mean(np.abs(predicted / actual - 1)) * 100)
    flat_mae = float(np.mean(np.abs(origin / actual - 1)) * 100)
    nz = y != 0
    return {
        'samples': len(origin),
        'priceMaePct': model_mae, 'copyOriginCloseMaePct': flat_mae,
        'priceMaeUsd': float(np.mean(np.abs(predicted - actual))),
        'copyOriginCloseMaeUsd': float(np.mean(np.abs(origin - actual))),
        'relativePriceMaeImprovement': 1 - model_mae / flat_mae,
        'predictedLogReturnStd': float(np.std(predicted_return)),
        'actualLogReturnStd': float(np.std(y)),
        'predictiveStdRatio': float(np.std(predicted_return) / np.std(y)),
        'priceCorrelation': corr(predicted, actual),
        'copyOriginClosePriceCorrelation': corr(origin, actual),
        'returnCorrelation': corr(predicted_return, y),
        'returnR2VsZero': float(1 - np.mean((predicted_return - y) ** 2) / np.mean(y ** 2)),
        'meanReturnDirection': direction(y, np.sign(predicted_return)),
        'classificationDirection': direction(y, np.where(probability >= .5, 1, -1)),
        'classificationBrier': float(np.mean((probability[nz] - (y[nz] > 0)) ** 2)),
    }


def payload(interval, candles):
    rows = []
    for row in candles:
        item = dict(zip(SERVICE.REQUIRED_FIELDS, row.tolist()))
        item['time'] = int(row[0] // 1000)
        item['trades'] = int(item['trades'])
        item['isClosed'] = True
        rows.append(item)
    return {'symbol': 'BTCUSDT', 'interval': interval, 'candles': rows}


def infer(bundle, raw, vol, close):
    return SERVICE.TREES.predict_features(bundle['tree'], raw, np.zeros(len(raw), dtype=np.int8), vol, close)


def audit_interval(engine, interval):
    started = time.perf_counter()
    bundle = engine.load(interval)  # Service verifies tree/magnitude contracts and real weight hashes.
    metadata = bundle['tree']['contract']['dataMetadata']
    source = metadata['sources']['BTCUSDT']
    path = ROOT / source['path']
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != source['sha256']:
        raise ValueError(f'Frozen raw data hash differs: {interval}')
    candles = b4_data.validate_rows(json.loads(path.read_text()), 'BTCUSDT', interval, metadata['cutoff'])
    raw, vol, names = b4_data.features(candles)
    assert names == bundle['featureNames']
    index = np.arange(len(candles))
    anchors = index[(index >= 99) & (index + 12 < len(candles)) & (candles[:, 0] >= bundle['cutoff'])][-240:]
    if not len(anchors):
        raise ValueError(f'No jointly observable origins: {interval}')
    close = candles[anchors, 4]
    full = infer(bundle, raw[anchors], vol[anchors], close)
    prefix_raw, prefix_vol = [], []
    for anchor in anchors:
        xx, vv, prefix_names = b4_data.features(candles[:anchor + 1])
        assert prefix_names == names
        prefix_raw.append(xx[-1])
        prefix_vol.append(vv[-1])
    prefix_raw, prefix_vol = np.asarray(prefix_raw), np.asarray(prefix_vol)
    prefix = infer(bundle, prefix_raw, prefix_vol, close)
    parity = {
        'originsChecked': len(anchors), 'returnHorizonsChecked': list(range(1, 13)),
        'directionHorizonsChecked': [1, 3, 6, 12],
        'featuresExactlyEqual': bool(np.array_equal(prefix_raw, raw[anchors])),
        'volatilityExactlyEqual': bool(np.array_equal(prefix_vol, vol[anchors])),
        'maxFeatureDifference': float(np.max(np.abs(prefix_raw - raw[anchors]))),
        'maxLogReturnDifference': float(np.max(np.abs(prefix['logReturns'] - full['logReturns']))),
        'maxPriceDifferenceUsd': float(np.max(np.abs(prefix['prices'] - full['prices']))),
        'maxProbabilityDifference': max(float(np.max(np.abs(prefix['directionProbabilities'][k] - v)))
                                       for k, v in full['directionProbabilities'].items()),
    }
    # Perturb every future OHLC/flow field, leaving the origin prefix intact.
    probe_anchors = sorted(set([int(anchors[0]), int(anchors[len(anchors)//2]), int(anchors[-1])]))
    changed_features, changed_returns = 0., 0.
    service_prices, service_probs = 0., 0.
    for anchor in probe_anchors:
        modified = candles.copy()
        modified[anchor + 1:, 1:5] *= 1.7
        modified[anchor + 1:, 5:] *= 1.3
        xx, vv, _ = b4_data.features(modified)
        changed_features = max(changed_features, float(np.max(np.abs(xx[anchor] - raw[anchor]))))
        altered = infer(bundle, xx[anchor:anchor+1], vv[anchor:anchor+1], candles[anchor:anchor+1, 4])
        original = infer(bundle, raw[anchor:anchor+1], vol[anchor:anchor+1], candles[anchor:anchor+1, 4])
        changed_returns = max(changed_returns, float(np.max(np.abs(altered['logReturns'] - original['logReturns']))))
        result = engine.forecast(payload(interval, candles[max(0, anchor-999):anchor+1]), now_seconds=metadata['cutoff']/1000)
        service_prices = max(service_prices, float(np.max(np.abs(np.asarray([p['mid'] for p in result['path'][1:]]) - original['prices'][0]))))
        for h in (1, 3, 6, 12):
            service_probs = max(service_probs, abs(result['path'][h]['upProbability'] - original['directionProbabilities'][f'h{h}'][0]))
        assert all(point['time'] == candles[anchor, 0]//1000 + h * b4_data.INTERVAL_MS[interval]//1000
                   for h, point in enumerate(result['path']))
    parity.update({'futurePerturbationOrigins': len(probe_anchors),
                   'futurePerturbationMaxFeatureDifference': changed_features,
                   'futurePerturbationMaxLogReturnDifference': changed_returns,
                   'servicePrefixOrigins': len(probe_anchors),
                   'servicePrefixMaxPriceDifferenceUsd': service_prices,
                   'servicePrefixMaxProbabilityDifference': service_probs})
    # Ending at the last paired origin's h1 target makes service history contain
    # exactly the same paired-origin window (no extra eleven h1-only origins).
    replay = engine.forecast(payload(interval, candles[int(anchors[0])-99:int(anchors[-1])+2]),
                             now_seconds=metadata['cutoff']/1000)['history']
    assert len(replay) == len(anchors)
    alignment = True
    history_diff = 0.
    for row, anchor, price in zip(replay, anchors, full['prices'][:, 0]):
        alignment &= (row['originTime'] == candles[anchor, 0]//1000
                      and row['time'] == candles[anchor+1, 0]//1000
                      and row['baseline'] == candles[anchor, 4]
                      and row['actual'] == candles[anchor+1, 4])
        history_diff = max(history_diff, abs(row['predicted'] - price))
    parity.update({'serviceHistoryOriginsChecked': len(replay), 'serviceHistoryAlignmentCorrect': bool(alignment),
                   'serviceHistoryMaxPriceDifferenceUsd': history_diff})
    parity['passed'] = bool(parity['featuresExactlyEqual'] and parity['volatilityExactlyEqual']
                            and parity['maxLogReturnDifference'] == 0 and parity['maxProbabilityDifference'] == 0
                            and changed_features == 0 and changed_returns == 0 and alignment
                            and service_prices <= 1e-6 and history_diff <= 1e-6 and service_probs <= 1e-9)
    results = {f'h{h}': metrics(close, candles[anchors+h, 4], full['logReturns'][:, h-1],
                              full['directionProbabilities'][f'h{h}']) for h in (1, 12)}
    # Fixed before evaluating: illustrate the latest jointly observable origin.
    anchor = int(anchors[-1])
    rolling = np.arange(anchor, anchor+12)
    stitched = infer(bundle, raw[rolling], vol[rolling], candles[rolling, 4])['prices'][:, 0]
    actual = candles[anchor+np.arange(1, 13), 4]
    fixed = full['prices'][-1]
    example = {'selectionRule': 'Latest jointly evaluable origin, never chosen by error',
               'origin': iso(candles[anchor, 0]), 'originClose': float(candles[anchor, 4]),
               'targetTimes': [iso(t) for t in candles[anchor+np.arange(1, 13), 0]],
               'actualCloses': actual.tolist(), 'fixedOriginTwelvePrices': fixed.tolist(),
               'rollingH1TwelvePrices': stitched.tolist(),
               'rollingCopyPreviousClose': candles[rolling, 4].tolist(),
               'fixedOriginPathMaePct': float(np.mean(abs(fixed/actual-1))*100),
               'rollingH1PathMaePct': float(np.mean(abs(stitched/actual-1))*100),
               'rollingCopyPreviousCloseMaePct': float(np.mean(abs(candles[rolling, 4]/actual-1))*100),
               'warning': 'Rolling h1 receives eleven newly observed actual closes. These paths use different information sets and are not a fair model-skill comparison.'}
    old_report = json.loads((ROOT / f'reports/experiments/b4-trees/{interval}/report.json').read_text())
    return {'interval': interval, 'symbol': 'BTCUSDT', 'samples': len(anchors), 'requestedSamples': 240,
            'dataSource': source['path'], 'rawDataSha256': actual_hash,
            'fitCalibrationCutoff': iso(bundle['cutoff']), 'artifactFrozenAt': old_report['selection']['frozenAt'],
            'originStart': iso(candles[anchors[0], 0]), 'originEnd': iso(candles[anchors[-1], 0]),
            'lastObservedTarget': iso(candles[anchors[-1]+12, 0]),
            'strictlyAfterArtifactFreezeOrigins': int(np.sum(candles[anchors, 0] > datetime.fromisoformat(old_report['selection']['frozenAt']).timestamp()*1000)),
            'metrics': results, 'causalParity': parity, 'sameOriginIllustration': example,
            'existingPooledFourSymbolHoldout': {'scope': 'All BTC/ETH/BNB/SOL historical final-partition samples, not this recent BTC window',
                'h1': old_report['results']['h1']['test'], 'h12': old_report['results']['h12']['test']},
            'seconds': time.perf_counter() - started}


def markdown(report):
    lines = ['# B4 历史回放与同起点未来预测审计', '',
             f"生成：{report['createdAt']}。当前真实服务使用的冻结 CatBoost B4 权重；未训练、未选模、未更改部署。", '',
             '最近 BTC 窗口：h1 和 h12 使用完全相同的起点。起点必须晚于拟合/校准截止，且后续 12 根均已收盘；最多 240 个，日线不足时如实报告。所有行情来自原冻结缓存并核对 SHA256。', '',
             '**这是冻结权重的历史重建，不是模型创建后实时留存的预测。** 缓存结束时间早于本轮模型创建时间，本次审计窗口中模型创建之后的起点为 0；本审计不推断其他实时归档是否存在。末段此前已被其他实验查看，也不能称为全新独立测试。', '',
             '价格 MAE(%) = mean(abs(预测价 / 实际价 − 1)) × 100；持平基线复制每个起点的收盘价。收益为 log(C[t+h]/C[t])。幅度比为预测收益标准差 / 真实收益标准差。方向准确率排除真实收益恰好为 0 的样本，均值头恰好预测 0 记为弃权且不计命中；分类头按概率 ≥ 0.5 判断上涨。', '',
             '| 周期 | 步长 | N | 模型 MAE% | 持平 MAE% | MAE改善% | 收益幅度比 | 价格相关 | 持平价格相关 | 收益相关 | 均值方向准确率% | 均值平衡准确率% | 分类准确率% | 分类平衡准确率% |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in report['intervals']:
        for horizon, m in item['metrics'].items():
            lines.append(f"| {item['interval']} | {horizon} | {m['samples']} | {m['priceMaePct']:.4f} | {m['copyOriginCloseMaePct']:.4f} | {m['relativePriceMaeImprovement']*100:+.2f} | {m['predictiveStdRatio']:.4f} | {m['priceCorrelation']:.5f} | {m['copyOriginClosePriceCorrelation']:.5f} | {m['returnCorrelation']:.5f} | {m['meanReturnDirection']['accuracy']*100:.2f} | {m['meanReturnDirection']['balancedAccuracy']*100:.2f} | {m['classificationDirection']['accuracy']*100:.2f} | {m['classificationDirection']['balancedAccuracy']*100:.2f} |")
    hourly = next(item for item in report['intervals'] if item['interval'] == '1h')['metrics']
    lines += ['', f"1h 的 h1 模型误差为 {hourly['h1']['priceMaePct']:.5f}%，复制上一根收盘价为 {hourly['h1']['copyOriginCloseMaePct']:.5f}%；两者价格相关都约 {hourly['h1']['priceCorrelation']:.3f}，但模型收益相关为 {hourly['h1']['returnCorrelation']:.4f}。h12 模型误差 {hourly['h12']['priceMaePct']:.5f}%，同起点持平基线 {hourly['h12']['copyOriginCloseMaePct']:.5f}%。本窗口没有支持 1h 平均价格路径优于持平基线的证据。", '',
              '不能只看方向普通准确率：例如本窗口 15m h12 回归头全部给负收益，命中 66.25% 只是样本下跌占比较高，其平衡准确率仍为 50%；正负预测比例、弃权比例和真实上涨比例详见 JSON。']
    lines += ['', '## 为什么回放会看起来特别准', '',
              '当前历史线是滚动单步预测：每根收盘后读取新的真实价格，再预测下一根。每个点为 C[t] × exp(r̂[t,1])。未来蓝线是一次固定起点的直接多步预测：未来第 h 点始终为 C[t] × exp(r̂[t,h])，并未获得未来真实收盘价。', '',
              '将滚动 h1 连成一条线，会把每根新观测到的实际价格不断带入曲线。只复制上一根收盘价的模型也能画出非常贴近价格的曲线和很高的价格相关系数；这不代表预测收益、转折或未来 12 根具备同等准确率。应看相同起点、相同步长相对持平基线的误差和收益相关。', '',
              '同样，分类头的上涨概率与回归头的有符号平均收益是分别训练的预测。分类命中率不能拿来解释平均价格线的幅度；预估绝对幅度的上下情景也不是平均价格预测。', '',
              '## 因果性与对齐实测', '',
              '| 周期 | 起点范围 UTC | 实测 N | 所有 12 步前缀/完整历史完全一致 | 后续扰动影响 | 服务历史对齐 | 服务最大价格差 USD |',
              '|---|---|---:|---|---:|---|---:|']
    for item in report['intervals']:
        p = item['causalParity']
        exact = p['featuresExactlyEqual'] and p['maxLogReturnDifference'] == 0 and p['maxProbabilityDifference'] == 0
        lines.append(f"| {item['interval']} | {item['originStart']} → {item['originEnd']} | {item['samples']} | {exact} | {p['futurePerturbationMaxLogReturnDifference']:.2g} | {p['serviceHistoryAlignmentCorrect']} | {max(p['serviceHistoryMaxPriceDifferenceUsd'], p['servicePrefixMaxPriceDifferenceUsd']):.3g} |")
    lines += ['', '对每个起点独立重算只截至该起点的原始前缀特征，与含后续行情的完整特征逐元素比较，再用真实权重比较全部 12 个收益输出及 4 个分类概率。每周期首/中/末起点将未来 OHLC × 1.7、成交相关字段 × 1.3，检查过去预测不变。额外经真实 ForecastService 调用，核对起点、目标时间、实际目标收盘价与上一根收盘基线，价格容差 1e-6 USD。', '',
              f"全部检查通过：{report['allCausalChecksPassed']}。这是对本次窗口、特征和真实推理链的实测，没有发现未来特征泄漏或 h1 目标错位；不等于统计证明模型有效。", '',
              '## 固定起点与滚动拼接的具体例子', '',
              '每周期固定选择最后一个有完整 h12 结果的起点，不按误差挑例子。JSON 保存 12 个目标、两种预测路径和滚动持平路径的每个数值。滚动线比一次性路径多获得 11 根真实收盘数据，下面误差仅用来说明视觉错觉，不能作为公平优劣比较。', '',
              '| 周期 | 固定起点 UTC | 同起点 12 点 MAE% | 滚动 h1 MAE% | 滚动复制上一收盘 MAE% |',
              '|---|---|---:|---:|---:|']
    for item in report['intervals']:
        ex = item['sameOriginIllustration']
        lines.append(f"| {item['interval']} | {ex['origin']} | {ex['fixedOriginPathMaePct']:.4f} | {ex['rollingH1PathMaePct']:.4f} | {ex['rollingCopyPreviousCloseMaePct']:.4f} |")
    lines += ['', '## 解释边界与展示建议', '',
              '本表是 BTC 最近一段已查看历史的描述性复评。既有四币留出评估另保存在 JSON 的 existingPooledFourSymbolHoldout 字段，不与本窗口混用。h12 目标相互重叠，240 点并非 240 个独立样本；不根据本窗口挑模型或声称稳定方向优势。', '',
              '建议将历史线明确标为“滚动单步回放”，同时画出复制上一收盘价的基线；新增选定历史起点的整条未来 12 根路径，对照该起点之后的真实走势。完整模型能力必须按同起点、同预测步长比较，并逐步保存模型创建后的真实预测，积累前向证据。', '']
    return '\n'.join(lines)


def main():
    started = time.perf_counter()
    engine = SERVICE.ForecastService()
    items = []
    for interval in ('15m', '1h', '4h', '1d'):
        item = audit_interval(engine, interval)
        items.append(item)
        print(json.dumps({'interval': interval, 'samples': item['samples'], 'metrics': item['metrics'],
                          'causalParity': item['causalParity'], 'seconds': item['seconds']}, ensure_ascii=False), flush=True)
    report = {'createdAt': datetime.now(timezone.utc).isoformat(), 'model': 'guanchao-b4 current CatBoost service',
              'protocol': 'Latest <=240 BTC origins after fit/calibration cutoff, same origins for h1 and h12, frozen data/weights, no fitting or selection',
              'limitations': ['Historical reconstruction, not live-issued forecast archive', 'Recent final-period history was already inspected in prior experiments', 'Overlapping h12 targets are serially dependent'],
              'allCausalChecksPassed': all(x['causalParity']['passed'] for x in items),
              'intervals': items, 'seconds': time.perf_counter()-started}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    OUT.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
    print(f'WROTE {OUT}.json and .md seconds={report["seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()

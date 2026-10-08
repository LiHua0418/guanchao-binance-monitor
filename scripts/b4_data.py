"""Frozen, causal multi-scale data shared by B4 experiment families.

Run ``python scripts/b4_data.py`` once before training. ``get_dataset(interval)``
loads frozen NumPy arrays without contacting the network. Partitions expose x,
y (12 cumulative log returns), y_norm, vol, time, symbol, close, row_index.
Sequence input of length L (at most 64) is:
    d['features'][p['row_index'][:, None] + np.arange(1-L, 1)]
No feature uses a future candle; every target stays within its calendar split.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/training-expanded'
OLD_CACHE = ROOT / '.cache/training'
REPORT = ROOT / 'reports/experiments/b4-expanded/protocol-data.json'
API = 'https://data-api.binance.vision/api/v3'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
JOBS = {'1h': 30000, '15m': 40000, '4h': 14000, '1d': 4000}
INTERVAL_MS = {'15m': 900000, '1h': 3600000, '4h': 14400000, '1d': 86400000}
PART_NAMES = ['train', 'tune', 'val1', 'val2', 'val3', 'calibration', 'test']
HORIZON = 12
MAX_SEQUENCE = 64
WARMUP = 99
ANCHOR_START = WARMUP + MAX_SEQUENCE - 1
_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat().replace('+00:00', 'Z')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')


def get_json(endpoint, params=None):
    global _LAST_REQUEST
    for attempt in range(6):
        with _LOCK:
            delay = max(0, .15 - (time.monotonic() - _LAST_REQUEST))
            if delay:
                time.sleep(delay)
            _LAST_REQUEST = time.monotonic()
        try:
            response = requests.get(f'{API}/{endpoint}', params=params, timeout=40)
            if response.status_code in (418, 429):
                time.sleep(min(60, float(response.headers.get('Retry-After', 20))))
            response.raise_for_status()
            data = response.json()
            if endpoint == 'klines' and not isinstance(data, list):
                raise ValueError('Unexpected kline response')
            return data
        except (requests.RequestException, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(20, 2 ** attempt))


def validate_rows(rows, symbol, interval, cutoff):
    if len(rows) < ANCHOR_START + HORIZON + 100:
        raise ValueError(f'Insufficient candles: {symbol} {interval}')
    if any(len(r) < 11 or int(r[6]) >= cutoff for r in rows):
        raise ValueError(f'Incomplete or unclosed candle: {symbol} {interval}')
    # time, OHLC, base volume, quote volume, trades, buy base, buy quote
    a = np.asarray([[int(r[0]), *map(float, r[1:6]), float(r[7]), float(r[8]),
                     float(r[9]), float(r[10])] for r in rows], dtype=np.float64)
    if (not np.isfinite(a).all() or np.any(a[:, 1:5] <= 0) or np.any(a[:, 5:] < 0)
            or np.any(a[:, 8] > a[:, 5] * (1 + 1e-9) + 1e-8)
            or np.any(a[:, 9] > a[:, 6] * (1 + 1e-9) + 1e-8)
            or np.any(a[:, 2] < np.maximum(a[:, 1], a[:, 4]))
            or np.any(a[:, 3] > np.minimum(a[:, 1], a[:, 4]))):
        raise ValueError(f'Invalid candle values: {symbol} {interval}')
    differences = np.diff(a[:, 0])
    if not np.all(differences == INTERVAL_MS[interval]):
        bad = np.where(differences != INTERVAL_MS[interval])[0]
        raise ValueError(f'Noncontinuous candles: {symbol} {interval}; indices {bad[:10].tolist()}')
    return a


def fetch_history(symbol, interval, bars, cutoff):
    step = INTERVAL_MS[interval]
    last_open = cutoff // step * step - step
    path = CACHE / f'{symbol}-{interval}-{bars}-{last_open}.json'
    requests_count, reused = 0, 0
    if path.exists():
        rows = json.loads(path.read_text(encoding='utf-8'))
        reused = len(rows)
    else:
        rows = []
        candidates = list(CACHE.glob(f'{symbol}-{interval}-*.json')) + list(OLD_CACHE.glob(f'{symbol}-{interval}-*.json'))
        for candidate in candidates:
            data = json.loads(candidate.read_text(encoding='utf-8'))
            rows.extend(r for r in data if int(r[6]) < cutoff and int(r[0]) <= last_open)
        rows = sorted({int(r[0]): r for r in rows}.values(), key=lambda r: int(r[0]))[-bars:]
        reused = len(rows)
        # Add recent bars first, then extend history backwards. Existing raw
        # kline payloads (including taker/trade/quote fields) remain intact.
        if rows and int(rows[-1][0]) < last_open:
            start = int(rows[-1][0]) + step
            while start <= last_open:
                batch = get_json('klines', {'symbol': symbol, 'interval': interval, 'limit': 1000,
                                           'startTime': start, 'endTime': last_open + step - 1})
                requests_count += 1
                if not batch:
                    break
                rows.extend(batch)
                start = int(batch[-1][0]) + step
        rows = sorted({int(r[0]): r for r in rows if int(r[6]) < cutoff}.values(), key=lambda r: int(r[0]))[-bars:]
        end = int(rows[0][0]) - 1 if rows else last_open + step - 1
        while len(rows) < bars:
            limit = min(1000, bars - len(rows))
            batch = get_json('klines', {'symbol': symbol, 'interval': interval, 'limit': limit, 'endTime': end})
            requests_count += 1
            if not batch:
                break
            rows = batch + rows
            end = int(batch[0][0]) - 1
            if len(batch) < limit:
                break
        rows = sorted({int(r[0]): r for r in rows if int(r[6]) < cutoff}.values(), key=lambda r: int(r[0]))[-bars:]
        validate_rows(rows, symbol, interval, cutoff)
        write_json(path, rows)
    candles = validate_rows(rows, symbol, interval, cutoff)
    info = {'bars': len(rows), 'first': iso(rows[0][0]), 'last': iso(rows[-1][0]),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'path': str(path.relative_to(ROOT)),
            'reusedBars': reused, 'networkRequests': requests_count, 'continuous': True, 'closed': True}
    print(f'DATA {symbol} {interval} bars={len(rows)} reused={reused} requests={requests_count}', flush=True)
    return candles, info


def rolling_mean(values, window):
    sums = np.cumsum(np.r_[0., values], dtype=np.float64)
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (sums[end] - sums[start]) / (end - start)


def lag(values, n):
    return values[np.maximum(np.arange(len(values)) - n, 0)]


def features(candles):
    """Exact portable feature specification: all rolling windows include t.

    Input columns: timestamp, open, high, low, close, baseVolume, quoteVolume,
    trades, takerBuyBase, takerBuyQuote. Return raw float64 features and sigma48.
    Prefixes shorter than 100 are warm-up and excluded from training.
    """
    t, o, h, l, c, volume, quote, trades, buy_base, buy_quote = candles.T
    values, names = [], []
    def add(name, value):
        names.append(name)
        values.append(value)
    logc = np.log(c)
    r1 = logc - lag(logc, 1)
    returns = {n: logc - lag(logc, n) for n in (1, 2, 3, 6, 12, 24, 48)}
    for n, r in returns.items():
        add(f'logReturn{n}', r)
    vols = {}
    for n in (12, 24, 48):
        vols[n] = np.sqrt(np.maximum(rolling_mean(r1 * r1, n) - rolling_mean(r1, n) ** 2, 0))
        add(f'volatility{n}', vols[n])
    vol = np.maximum(vols[48], 1e-5)
    add('rangeRatio', (h - l) / c)
    add('bodyRatio', (c - o) / o)
    add('upperWickRatio', (h - np.maximum(o, c)) / c)
    add('lowerWickRatio', (np.minimum(o, c) - l) / c)
    add('closeWithinCandle', np.divide(c - l, h - l, out=np.full(len(c), .5), where=h > l) - .5)
    for n in (5, 10, 20, 50, 100):
        add(f'sma{n}Distance', c / rolling_mean(c, n) - 1)
    for name, v in [('baseVolume', volume), ('quoteVolume', quote)]:
        for n in (20, 50):
            add(f'{name}Relative{n}', v / np.maximum(rolling_mean(v, n), 1e-8) - 1)
        add(f'{name}LogChange', np.log1p(v) - lag(np.log1p(v), 1))
    imbalance = np.divide(2 * buy_base, volume, out=np.ones(len(c)), where=volume > 0) - 1
    for n in (0, 1, 2, 3):
        add(f'takerImbalanceLag{n}', lag(imbalance, n))
    for n in (3, 6, 12):
        add(f'takerImbalanceMean{n}', rolling_mean(imbalance, n))
    add('tradeCountLogChange', np.log1p(trades) - lag(np.log1p(trades), 1))
    for n in (20, 50):
        add(f'tradeCountRelative{n}', trades / np.maximum(rolling_mean(trades, n), 1) - 1)
    avg_trade = volume / np.maximum(trades, 1)
    add('averageTradeLogChange', np.log1p(avg_trade) - lag(np.log1p(avg_trade), 1))
    day = t / 86400000
    for name, phase in [('utcTime', day % 1), ('utcWeek', ((day + 3) % 7) / 7)]:
        add(f'{name}Sin', np.sin(2 * np.pi * phase))
        add(f'{name}Cos', np.cos(2 * np.pi * phase))
    for n, r in returns.items():
        add(f'volNormalizedReturn{n}', r / (vol * np.sqrt(n)))
    gains = rolling_mean(np.maximum(r1, 0), 14)
    losses = rolling_mean(np.maximum(-r1, 0), 14)
    add('rsi14Centered', np.divide(gains, gains + losses, out=np.full(len(c), .5), where=gains + losses > 0) - .5)
    for n in (20, 50):
        mu = rolling_mean(logc, n)
        sigma = np.sqrt(np.maximum(rolling_mean(logc ** 2, n) - mu ** 2, 0))
        add(f'logPriceZScore{n}', (logc - mu) / np.maximum(sigma, 1e-5))
    for n in (12, 24):
        add(f'meanRange{n}', rolling_mean((h - l) / c, n))
    add('volatilityRatio12to48', vols[12] / vol - 1)
    add('logVolatility48', np.log(vol))
    add('vwapDistance', np.divide(quote, volume * c, out=np.ones(len(c)), where=volume > 0) - 1)
    add('bodyTakerInteraction', (c - o) / (o * vol) * imbalance)
    x = np.column_stack(values)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features')
    return x, vol, names


def build_dataset(interval, groups, source_info, cutoff):
    step = INTERVAL_MS[interval]
    common_start = max(c[ANCHOR_START, 0] for c in groups.values())
    common_end = min(c[-1, 0] for c in groups.values()) + step
    span_bars = int((common_end - common_start) / step)
    boundaries = np.asarray([int(common_start + int(span_bars * p) * step) for p in (.6, .7, .75, .8, .85, .9)], dtype=np.int64)
    limits = np.r_[int(common_start), boundaries, int(common_end)]
    raw, train_rows, records = [], [], []
    offset = 0
    for sid, symbol in enumerate(SYMBOLS):
        candles = groups[symbol]
        x, vol, names = features(candles)
        # This catches accidental centered windows and all future-value use.
        for index in (ANCHOR_START, len(candles) // 2, len(candles) - 1):
            prefix, prefix_vol, prefix_names = features(candles[:index + 1])
            if not np.allclose(prefix[-1], x[index], rtol=1e-10, atol=1e-10):
                raise AssertionError(f'Noncausal features {symbol} {interval} at {index}')
            if not np.isclose(prefix_vol[-1], vol[index]):
                raise AssertionError('Noncausal volatility')
            assert prefix_names == names
        raw.append(x)
        feature_train = (candles[:, 0] >= common_start) & (candles[:, 0] < boundaries[0])
        train_rows.append(x[feature_train])
        anchors = np.arange(ANCHOR_START, len(candles) - HORIZON)
        origins = candles[anchors, 0].astype(np.int64)
        targets_end = candles[anchors + HORIZON, 0].astype(np.int64)
        part_id = np.searchsorted(boundaries, origins, side='right')
        valid = (origins >= common_start) & (targets_end < limits[part_id + 1])
        anchors, origins, part_id = anchors[valid], origins[valid], part_id[valid]
        target = np.log(candles[anchors[:, None] + np.arange(1, HORIZON + 1), 4] / candles[anchors, 4, None])
        records.append({'y': target, 'vol': vol[anchors], 'time': origins,
                        'symbol': np.full(len(anchors), sid, dtype=np.int8), 'close': candles[anchors, 4],
                        'row_index': anchors + offset, 'partition': part_id.astype(np.int8)})
        offset += len(candles)
    raw = np.concatenate(raw)
    stats_rows = np.concatenate(train_rows)
    mean = stats_rows.mean(axis=0)
    scale = np.maximum(stats_rows.std(axis=0), 1e-6)
    standardized = np.clip((raw - mean) / scale, -12, 12).astype(np.float32)
    dataset = {key: np.concatenate([r[key] for r in records]) for key in records[0]}
    dataset['y_norm'] = dataset['y'] / (dataset['vol'][:, None] * np.sqrt(np.arange(1, HORIZON + 1)))
    dataset.update(raw_features=raw.astype(np.float32), features=standardized, mean=mean, scale=scale,
                   feature_names=np.asarray(names), boundaries=boundaries)
    partitions = {}
    for pid, name in enumerate(PART_NAMES):
        mask = dataset['partition'] == pid
        if not np.any(mask):
            raise ValueError(f'Empty partition {interval} {name}')
        times = dataset['time'][mask]
        partitions[name] = {'samples': int(mask.sum()), 'firstOrigin': iso(times.min()), 'lastOrigin': iso(times.max()),
                            'lastTarget': iso(times.max() + HORIZON * step)}
        assert times.max() + HORIZON * step < limits[pid + 1]
    metadata = {'interval': interval, 'sources': source_info, 'cutoff': cutoff, 'cutoffIso': iso(cutoff),
                'features': names, 'featureCount': len(names), 'boundaryMs': boundaries.tolist(),
                'boundaries': [iso(b) for b in boundaries], 'partitions': partitions,
                'commonStart': iso(common_start), 'commonEndExclusive': iso(common_end),
                'purgeBars': HORIZON, 'maxSequence': MAX_SEQUENCE, 'featureWarmup': WARMUP,
                'normalization': 'Mean/std fitted only on feature rows in train calendar partition; clip +/-12.',
                'target': 'Next 1..12 closed-bar cumulative log returns; y_norm=y/(causal sigma48*sqrt(horizon)).',
                'directionLabels': 'Consumer-selected flat threshold on y_norm; never selected on final partition.',
                'validation': 'tune selects checkpoints; val1/val2/val3 are successive walk-forward blocks; refit uses only earlier targets.',
                'knownLimitation': 'Recent history overlaps prior inspected experiments. The final partition is historical reevaluation, not an untouched future test.',
                'checks': {'closedCandles': True, 'continuousTime': True, 'prefixCausality': True, 'targetPurge': True}}
    dataset['metadata_json'] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    path = CACHE / f'{interval}-dataset.npz'
    np.savez_compressed(path, **dataset)
    metadata['datasetPath'] = str(path.relative_to(ROOT))
    metadata['datasetSha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(CACHE / f'{interval}-metadata.json', metadata)
    print(f'READY {interval}: features={len(names)} samples={len(dataset["y"])} splits={[(n,p["samples"]) for n,p in partitions.items()]}', flush=True)
    return metadata


def get_dataset(interval):
    """Load cached frozen arrays only. No network and no mutable fit step."""
    path = CACHE / f'{interval}-dataset.npz'
    if not path.exists():
        raise FileNotFoundError(f'{path} is not ready; run scripts/b4_data.py first')
    metadata_path = CACHE / f'{interval}-metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['datasetSha256']:
        raise ValueError(f'Dataset hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    result = {k: data[k] for k in ('features', 'raw_features', 'mean', 'scale', 'boundaries')}
    result['feature_names'] = data['feature_names'].tolist()
    result['metadata'] = metadata
    result['symbols'] = SYMBOLS.copy()
    fields = ('y', 'y_norm', 'vol', 'time', 'symbol', 'close', 'row_index')
    for pid, name in enumerate(PART_NAMES):
        mask = data['partition'] == pid
        p = {key: data[key][mask] for key in fields}
        p['x'] = result['features'][p['row_index']]
        result[name] = p
    result['validation'] = {key: np.concatenate([result[n][key] for n in ('val1', 'val2', 'val3')])
                            for key in (*fields, 'x')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals', nargs='+', choices=list(JOBS), default=list(JOBS))
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    cutoff_path = CACHE / 'cutoff.json'
    if cutoff_path.exists():
        cutoff = int(json.loads(cutoff_path.read_text())['serverTime'])
    else:
        cutoff = int(get_json('time')['serverTime'])
        write_json(cutoff_path, {'serverTime': cutoff, 'iso': iso(cutoff), 'source': API + '/time'})
    protocol = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {
        'version': 'b4-expanded-v1', 'cutoff': cutoff, 'cutoffIso': iso(cutoff),
        'symbols': SYMBOLS, 'requestedBarsPerSymbol': JOBS, 'workers': 4, 'requestSpacingSeconds': .15,
        'splitFractions': {'train': [0,.6], 'tune': [.6,.7], 'val1': [.7,.75], 'val2': [.75,.8],
                           'val3': [.8,.85], 'calibration': [.85,.9], 'test': [.9,1]},
        'purgeBars': HORIZON,
        'knownLimitation': 'Recent dates were inspected during older B3 experiments; expanded earlier history does not create an untouched future holdout.',
        'intervals': {}}
    # Individual intervals can be rebuilt from frozen raw caches independently.
    # Their manifests are authoritative when merging the shared protocol.
    for existing_interval in JOBS:
        manifest = CACHE / f'{existing_interval}-metadata.json'
        if manifest.exists():
            protocol['intervals'][existing_interval] = json.loads(manifest.read_text(encoding='utf-8'))
    write_json(REPORT, protocol)
    for interval in args.intervals:
        groups, info = {}, {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_history, symbol, interval, JOBS[interval], cutoff): symbol for symbol in SYMBOLS}
            for future in as_completed(futures):
                symbol = futures[future]
                groups[symbol], info[symbol] = future.result()
        protocol['intervals'][interval] = build_dataset(interval, groups, info, cutoff)
        write_json(REPORT, protocol)


if __name__ == '__main__':
    main()

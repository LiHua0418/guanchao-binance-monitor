"""Loopback-only inference for frozen B4 return, direction and magnitude models.

Run .venv-training/Scripts/python.exe scripts/serve-b4.py . This service never
trains, downloads market data, selects models or alters existing model files.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
import sqlite3
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import numpy as np

import b4_data
from b4_forward import ForwardLedger

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = '127.0.0.1', 8765
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MIN_CANDLES, MAX_CANDLES, MAX_HISTORY = 100, 2000, 240
REQUIRED_FIELDS = ('time', 'open', 'high', 'low', 'close', 'volume', 'quoteVolume',
                   'trades', 'takerBuyVolume', 'takerBuyQuoteVolume')


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TREES = import_script('b4_service_trees', ROOT / 'scripts/train-b4-trees.py')
MAGNITUDE = import_script('b4_service_magnitude', ROOT / 'scripts/train-b4-magnitude.py')


class InvalidRequest(ValueError):
    pass


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def iso_ms(value):
    return datetime.fromtimestamp(int(value) / 1000, timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_request(payload, now_seconds=None):
    """Require observed, fully closed and contiguous candles with real flow data."""
    if not isinstance(payload, dict):
        raise InvalidRequest('Expected a JSON object')
    symbol, interval = payload.get('symbol'), payload.get('interval')
    if symbol not in b4_data.SYMBOLS or interval not in b4_data.INTERVAL_MS:
        raise InvalidRequest('Supported symbols: BTCUSDT/ETHUSDT/BNBUSDT/SOLUSDT; intervals: 15m/1h/4h/1d')
    rows = payload.get('candles')
    if not isinstance(rows, list) or not MIN_CANDLES <= len(rows) <= MAX_CANDLES:
        raise InvalidRequest(f'Expected {MIN_CANDLES}..{MAX_CANDLES} closed candles')
    step = b4_data.INTERVAL_MS[interval] // 1000
    now_seconds = time.time() if now_seconds is None else now_seconds
    values = []
    for index, candle in enumerate(rows):
        if not isinstance(candle, dict) or candle.get('isClosed') is not True:
            raise InvalidRequest(f'Candle {index} must have isClosed=true')
        if any(key not in candle for key in REQUIRED_FIELDS):
            raise InvalidRequest(f'Candle {index} is missing OHLCV or quote/trades/taker fields; missing values are never filled with zero')
        row = []
        for name in REQUIRED_FIELDS:
            value = candle[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InvalidRequest(f'Candle {index} field {name} must be a finite number')
            row.append(float(value))
        t, o, h, low, c, volume, quote, trades, buy, buy_quote = row
        if t <= 0 or t != int(t) or int(t) % step or t + step > now_seconds:
            raise InvalidRequest(f'Candle {index} time must be interval-aligned UTC seconds for a completed bar')
        if min(o, h, low, c) <= 0 or h < max(o, c, low) or low > min(o, c):
            raise InvalidRequest(f'Candle {index} has invalid OHLC prices')
        if min(volume, quote, trades, buy, buy_quote) < 0 or trades != int(trades):
            raise InvalidRequest(f'Candle {index} has invalid volume or integer trade count')
        if buy > volume * (1 + 1e-9) + 1e-8 or buy_quote > quote * (1 + 1e-9) + 1e-8:
            raise InvalidRequest(f'Candle {index} taker volume exceeds total volume')
        row[0] *= 1000
        values.append(row)
    candles = np.asarray(values, dtype=np.float64)
    if not np.all(np.diff(candles[:, 0]) == b4_data.INTERVAL_MS[interval]):
        raise InvalidRequest('Candles must be strictly increasing without duplicates or missing bars')
    return symbol, interval, candles


class ForecastService:
    def __init__(self, root=ROOT, ledger_path=None):
        self.root = Path(root)
        self.models = {}
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.ledger = ForwardLedger(ledger_path) if ledger_path is not None else None

    def health(self):
        readiness = {}
        for interval in b4_data.INTERVAL_MS:
            readiness[interval] = all((self.root / 'reports/experiments' / family / interval / name).is_file()
                                      for family in ('b4-trees', 'b4-magnitude')
                                      for name in ('model-contract.json', 'report.json'))
        return {'ok': True, 'modelId': 'guanchao-b4', 'service': 'local-frozen-inference',
                'apiVersion': 2, 'forwardLedgerEnabled': self.ledger is not None,
                'host': HOST, 'port': PORT, 'intervals': readiness, 'loadedIntervals': list(self.models),
                'cachedRequests': len(self.cache), 'minimumClosedCandles': MIN_CANDLES,
                'directionStatus': 'experimental', 'historyMeaning': 'Frozen-model historical replay after training/calibration cutoff; not forecasts recorded live'}

    def load(self, interval):
        if interval in self.models:
            return self.models[interval]
        tree_root = self.root / 'reports/experiments/b4-trees'
        magnitude_root = self.root / 'reports/experiments/b4-magnitude'
        tree = TREES.load_models(interval, tree_root)
        magnitude = MAGNITUDE.load_models(interval, magnitude_root)
        metadata = tree['contract']['dataMetadata']
        tree_report = read_json(tree_root / interval / 'report.json')
        magnitude_report = read_json(magnitude_root / interval / 'report.json')
        magnitude_protocol = read_json(magnitude_root / interval / 'protocol.json')
        magnitude_metadata = magnitude_protocol['data']
        # The first completed tree run embedded the dataset metadata before the
        # NPZ digest was added. Its raw-source hashes, boundaries and feature
        # specification still fully identify the frozen data without mutating
        # or inventing provenance for those already trained artifacts.
        for field in ('cutoff', 'boundaryMs', 'features', 'maxSequence', 'normalization'):
            if metadata[field] != magnitude_metadata[field]:
                raise ValueError(f'Direction and magnitude frozen data mismatch: {field}')
        for symbol in b4_data.SYMBOLS:
            if metadata['sources'][symbol]['sha256'] != magnitude_metadata['sources'][symbol]['sha256']:
                raise ValueError('Direction and magnitude raw source hashes disagree')
        if ('datasetSha256' in metadata and 'datasetSha256' in magnitude_metadata
                and metadata['datasetSha256'] != magnitude_metadata['datasetSha256']):
            raise ValueError('Direction and magnitude dataset hashes disagree')
        if tree['contract']['rawFeatureOrder'] != magnitude['contract']['features'][:-4]:
            raise ValueError('Direction and magnitude feature contracts disagree')
        cutoff = max(metadata['boundaryMs'][-1], magnitude_metadata['boundaryMs'][-1])
        h1 = tree_report['results']['h1']
        mh1 = magnitude_report['results']['h1']
        strong = mh1['baselines']['rollingAbsReturn20']
        mse_skill = 1 - mh1['test']['mse'] / strong['mse']
        mae_skill = 1 - mh1['test']['maePct'] / strong['maePct']
        uncertainty = mh1.get('strongBaselineBootstrap', {}).get('rollingAbsReturn20', {})
        lower_mse = uncertainty.get('relativeMseImprovement95', [float('-inf')])[0]
        lower_mae = uncertainty.get('relativeMaeImprovement95', [float('-inf')])[0]
        evaluation = {
            'direction': {
                'balancedAccuracy': h1['test']['balancedAccuracy'],
                # The simple direction comparator flips the preceding candle's
                # persistence label, including its deterministic zero tie rule.
                'baselineBalancedAccuracy': 1 - h1['baselines']['persistence']['balancedAccuracy'],
                'brier': h1['test']['brier'], 'baselineBrier': h1['baselines']['trainMean']['brier'],
                'cvPassed': tree['contract']['selectionGate']['directionEligible'],
            },
            'magnitude': {
                'mseImprovement': mse_skill, 'maeImprovement': mae_skill,
                'cvPassed': magnitude['contract']['validationEligible'],
                # Descriptive only: these diagnostics never select, replace,
                # suppress, amplify or switch a model's output.
                'heldoutSkill': bool(mse_skill > 0 and mae_skill > 0 and lower_mse > 0 and lower_mae > 0),
            },
            'samples': h1['test']['samples'], 'dataCutoff': iso_ms(cutoff),
            'dataEnd': metadata['partitions']['test']['lastTarget'],
        }
        version_data = json.dumps([tree['contract'], magnitude['contract']], sort_keys=True,
                                  separators=(',', ':'), allow_nan=False).encode('utf-8')
        model_version = f'b4:{interval}:' + hashlib.sha256(version_data).hexdigest()
        bundle = {'tree': tree, 'magnitude': magnitude, 'evaluation': evaluation, 'cutoff': cutoff,
                  'modelVersion': model_version,
                  'featureNames': tree['contract']['rawFeatureOrder']}
        self.models[interval] = bundle
        return bundle

    def attach_forward(self, result, candles, now_seconds):
        if self.ledger is None:
            return result
        interval = result['interval']
        # Stamp actual publication time, not request start/candle label. Cached
        # inference still resolves new observations without replacing first issue.
        summary = self.ledger.observe(result, candles, b4_data.INTERVAL_MS[interval] // 1000,
                                      self.models[interval]['modelVersion'],
                                      time.time() if now_seconds is None else now_seconds)
        return {**result, 'forward': summary}

    def forecast(self, payload, now_seconds=None):
        symbol, interval, candles = parse_request(payload, now_seconds)
        key = (symbol, interval, hashlib.sha256(candles.tobytes()).hexdigest())
        # Only a single lightweight CPU inference per process runs at a time;
        # HTTP accepts concurrent requests, repeated closed inputs hit the LRU.
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.attach_forward(self.cache[key], candles, now_seconds)
            bundle = self.load(interval)
            if candles[-1, 0] < bundle['cutoff']:
                raise InvalidRequest('Forecast origin precedes the frozen model training/calibration cutoff')
            raw, vol, feature_names = b4_data.features(candles)
            if feature_names != bundle['featureNames']:
                raise ValueError('Live feature order differs from frozen model contract')
            anchors = np.arange(MIN_CANDLES - 1, len(candles) - 1)
            anchors = anchors[candles[anchors, 0] >= bundle['cutoff']][-MAX_HISTORY:]
            # A complete path keeps ONE origin for all twelve future bars. It
            # must never be assembled from twelve rolling next-bar predictions.
            path_anchors = np.arange(MIN_CANDLES - 1, len(candles) - 12)
            path_anchors = path_anchors[candles[path_anchors, 0] >= bundle['cutoff']][-MAX_HISTORY:]
            inference_indices = np.unique(np.r_[anchors, path_anchors, len(candles) - 1])
            prediction_rows = {int(anchor): row for row, anchor in enumerate(inference_indices)}
            symbols = np.full(len(inference_indices), b4_data.SYMBOLS.index(symbol), dtype=np.int8)
            closes = candles[inference_indices, 4]
            forecast = TREES.predict_features(bundle['tree'], raw[inference_indices], symbols, vol[inference_indices], closes)
            moves = MAGNITUDE.predict_features(bundle['magnitude'], raw[inference_indices], symbols, vol[inference_indices], closes)
            if any(not np.isfinite(forecast[k]).all() for k in ('prices', 'lowerPrices', 'upperPrices')):
                raise ValueError('Nonfinite return model output')
            if not np.isfinite(moves['expectedMovePct']).all():
                raise ValueError('Nonfinite magnitude model output')
            origin_time = int(candles[-1, 0] // 1000)
            origin_price = float(candles[-1, 4])
            step = b4_data.INTERVAL_MS[interval] // 1000
            path = [{'time': origin_time, 'mid': origin_price, 'lower': origin_price, 'upper': origin_price,
                     'upScenario': origin_price, 'downScenario': origin_price, 'expectedMovePct': 0.}]
            for h in range(1, 13):
                point = {'time': origin_time + h * step,
                         'mid': float(forecast['prices'][-1, h-1]),
                         'lower': float(forecast['lowerPrices'][-1, h-1]),
                         'upper': float(forecast['upperPrices'][-1, h-1]),
                         'upScenario': float(moves['upScenarioPrices'][-1, h-1]),
                         'downScenario': float(moves['downScenarioPrices'][-1, h-1]),
                         'expectedMovePct': float(moves['expectedMovePct'][-1, h-1])}
                if f'h{h}' in forecast['directionProbabilities']:
                    point['upProbability'] = float(forecast['directionProbabilities'][f'h{h}'][-1])
                path.append(point)
            history = []
            for anchor in anchors:
                row = prediction_rows[int(anchor)]
                previous, actual = float(candles[anchor, 4]), float(candles[anchor+1, 4])
                history.append({'time': int(candles[anchor+1, 0] // 1000),
                                'originTime': int(candles[anchor, 0] // 1000),
                                'predicted': float(forecast['prices'][row, 0]),
                                'actual': actual, 'baseline': previous,
                                'upProbability': float(forecast['directionProbabilities']['h1'][row]),
                                'expectedMovePct': float(moves['expectedMovePct'][row, 0]),
                                'actualMovePct': float(np.expm1(abs(np.log(actual / previous))) * 100)})
            replay_paths = []
            for anchor in path_anchors:
                row = prediction_rows[int(anchor)]
                start_time = int(candles[anchor, 0] // 1000)
                start_price = float(candles[anchor, 4])
                replay_paths.append({'originTime': start_time, 'originPrice': start_price,
                    'path': [{'time': start_time, 'horizon': 0, 'predicted': start_price, 'actual': start_price}]
                    + [{'time': start_time + h * step, 'horizon': h,
                        'predicted': float(forecast['prices'][row, h-1]),
                        'actual': float(candles[anchor+h, 4])} for h in range(1, 13)]})
            result = {'modelId': 'guanchao-b4', 'symbol': symbol, 'interval': interval,
                      'originTime': origin_time, 'originPrice': origin_price, 'path': path,
                      'history': history, 'replayPaths': replay_paths, 'evaluation': bundle['evaluation'],
                      'semantics': {
                          'mid': 'Unamplified conditional mean price from direct horizon return heads.',
                          'scenarios': 'Symmetric +/- expected absolute cumulative log-return magnitude; not confidence bounds or a directional price prediction.',
                          'interval': '10th/90th calibration-residual price quantiles from the return model; historical calibration does not guarantee future coverage.',
                          'direction': 'Experimental calibrated probability; no claim of stable directional advantage.',
                          'history': 'Frozen-model reconstruction after fit/calibration cutoff, using only each origin prefix; not predictions recorded in real time.',
                          'replayPaths': 'Each path uses ONE closed origin for h1..h12. Actuals never re-anchor the predicted path. Complete realized paths only; historical reconstruction, not live-issued records.',
                          'evaluation': 'Pooled four-symbol h1 historical reevaluation. Recent dates were already inspected in prior experiments; no untouched future evidence.',
                          'magnitudeBaseline': 'Mean absolute one-bar log return of previous 20 closed candles.',
                          'directionBaseline': 'Previous-candle direction reversed; Brier baseline uses pre-calibration training direction prior.',
                      }}
            # Ensure an HTTP response can always be strict finite JSON before
            # caching it, including all nested metrics and optional outputs.
            json.dumps(result, allow_nan=False)
            self.cache[key] = result
            while len(self.cache) > 64:
                self.cache.popitem(last=False)
            return self.attach_forward(result, candles, now_seconds)


class Handler(BaseHTTPRequestHandler):
    service = ForecastService()
    server_version = 'B4Local/1'

    def allowed(self):
        host = urlsplit('http://' + self.headers.get('Host', '')).hostname
        if host not in ('127.0.0.1', 'localhost'):
            self.respond(403, {'error': 'Loopback Host required'})
            return False
        origin = self.headers.get('Origin')
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('127.0.0.1', 'localhost'):
                self.respond(403, {'error': 'Only a local panel may access this service'})
                return False
        return True

    def respond(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        origin = self.headers.get('Origin')
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme in ('http', 'https') and parsed.hostname in ('127.0.0.1', 'localhost'):
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        if not self.allowed():
            return
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin', 'http://127.0.0.1:5173'))
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        if not self.allowed():
            return
        if self.path != '/api/b4/health':
            self.respond(404, {'error': 'Not found'})
            return
        self.respond(200, self.service.health())

    def do_POST(self):
        if not self.allowed():
            return
        if self.path != '/api/b4/forecast':
            self.respond(404, {'error': 'Not found'})
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            self.respond(415, {'error': 'Content-Type application/json required'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        if length <= 0 or self.headers.get('Transfer-Encoding'):
            self.respond(411, {'error': 'A positive Content-Length is required; chunked bodies are unsupported'})
            return
        if length > MAX_REQUEST_BYTES:
            self.respond(413, {'error': 'Request body exceeds 2 MiB'})
            return
        try:
            self.connection.settimeout(15)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise InvalidRequest('Incomplete request body')
            payload = json.loads(raw.decode('utf-8'), parse_constant=lambda value: (_ for _ in ()).throw(InvalidRequest(f'Nonfinite JSON value: {value}')))
            result = self.service.forecast(payload)
            self.respond(200, result)
        except (InvalidRequest, UnicodeDecodeError, json.JSONDecodeError) as error:
            self.respond(422, {'error': str(error)})
        except FileNotFoundError:
            self.respond(503, {'error': 'Frozen B4 model files are not yet available for this interval'})
        except (ValueError, OSError, KeyError, sqlite3.Error) as error:
            self.log_error('Inference refused: %s', error)
            self.respond(503, {'error': 'B4 artifact validation or inference failed; inspect local service log'})


def main():
    Handler.service = ForecastService(ledger_path=ROOT / '.cache/forward-evaluation.sqlite3')
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f'B4 local inference http://{HOST}:{PORT}; loopback only; Ctrl+C to stop', flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

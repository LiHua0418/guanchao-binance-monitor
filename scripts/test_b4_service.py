"""Integration checks against real frozen B4 artifacts and loopback HTTP."""
from __future__ import annotations

import copy
import http.client
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b4_service_test_target', ROOT / 'scripts/serve-b4.py')
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def fixture(interval, count=500, symbol='BTCUSDT'):
    meta = json.loads((ROOT / f'.cache/training-expanded/{interval}-metadata.json').read_text())
    rows = json.loads((ROOT / meta['sources'][symbol]['path']).read_text())[-count:]
    candles = [{'time': int(r[0]) // 1000, 'open': float(r[1]), 'high': float(r[2]), 'low': float(r[3]),
                'close': float(r[4]), 'volume': float(r[5]), 'quoteVolume': float(r[7]),
                'trades': int(r[8]), 'takerBuyVolume': float(r[9]), 'takerBuyQuoteVolume': float(r[10]),
                'isClosed': True} for r in rows]
    return {'symbol': symbol, 'interval': interval, 'candles': candles}


class B4ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = service.ForecastService()
        class QuietHandler(service.Handler):
            service = cls.engine
            def log_message(self, *args):
                pass
        cls.server = service.ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, payload=None, path='/api/b4/forecast', headers=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data=body,
                                         headers=headers or {'Content-Type': 'application/json'})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health(self):
        status, value = self.request(path='/api/b4/health')
        self.assertEqual(status, 200)
        self.assertEqual(value['host'], '127.0.0.1')
        self.assertTrue(all(value['intervals'].values()))

    def test_four_interval_real_http_predictions(self):
        for interval in ('15m', '1h', '4h', '1d'):
            with self.subTest(interval=interval):
                payload = fixture(interval)
                status, value = self.request(payload)
                self.assertEqual(status, 200, value)
                self.assertEqual(len(value['path']), 13)
                self.assertEqual(value['originPrice'], payload['candles'][-1]['close'])
                step = service.b4_data.INTERVAL_MS[interval] // 1000
                self.assertTrue(0 < len(value['history']) <= 240)
                for h, point in enumerate(value['path']):
                    self.assertEqual(point['time'], value['originTime'] + h*step)
                    self.assertLessEqual(point['lower'], point['upper'])
                    self.assertLessEqual(point['downScenario'], point['upScenario'])
                    self.assertGreaterEqual(point['expectedMovePct'], 0)
                    self.assertEqual('upProbability' in point, h in (1, 3, 6, 12))
                cutoff = service.datetime.fromisoformat(value['evaluation']['dataCutoff'].replace('Z', '+00:00')).timestamp()
                for point in value['history']:
                    self.assertGreaterEqual(point['originTime'], cutoff)
                    self.assertEqual(point['time'] - point['originTime'], step)
                    self.assertLessEqual(point['time'], value['originTime'])
                    self.assertGreaterEqual(point['actualMovePct'], 0)
                self.assertTrue(all(b['time'] - a['time'] == step for a,b in zip(value['history'], value['history'][1:])))

    def test_prefix_causal_predictions_all_intervals(self):
        for interval in ('15m', '1h', '4h', '1d'):
            with self.subTest(interval=interval):
                full = fixture(interval)
                prefix = copy.deepcopy(full)
                prefix['candles'] = prefix['candles'][:-20]
                a = self.engine.forecast(prefix)
                b = self.engine.forecast(full)
                replay = next(p for p in b['history'] if p['originTime'] == a['originTime'])
                self.assertEqual(a['path'][1]['mid'], replay['predicted'])
                self.assertEqual(a['path'][1]['upProbability'], replay['upProbability'])
                self.assertEqual(a['path'][1]['expectedMovePct'], replay['expectedMovePct'])
                changed = copy.deepcopy(full)
                for candle in changed['candles'][-20:]:
                    for key in ('open', 'high', 'low', 'close'):
                        candle[key] *= 1.1
                c = self.engine.forecast(changed)
                unchanged = next(p for p in c['history'] if p['originTime'] == a['originTime'])
                self.assertEqual(replay['predicted'], unchanged['predicted'])
                self.assertEqual(replay['upProbability'], unchanged['upProbability'])
                self.assertEqual(replay['expectedMovePct'], unchanged['expectedMovePct'])

    def test_complete_replay_paths_have_one_origin_and_observed_targets(self):
        for interval in ('15m', '1h', '4h', '1d'):
            with self.subTest(interval=interval):
                value = fixture(interval)
                result = self.engine.forecast(value)
                candles = {c['time']: c for c in value['candles']}
                history = {p['originTime']: p for p in result['history']}
                step = service.b4_data.INTERVAL_MS[interval] // 1000
                cutoff = self.engine.load(interval)['cutoff'] // 1000
                replays = result['replayPaths']
                self.assertTrue(0 < len(replays) <= 240)
                self.assertEqual(replays[-1]['originTime'], result['originTime'] - 12 * step)
                self.assertEqual(len({r['originTime'] for r in replays}), len(replays))
                for replay in replays:
                    origin = replay['originTime']
                    self.assertGreaterEqual(origin, cutoff)
                    self.assertLessEqual(origin + 12 * step, result['originTime'])
                    self.assertEqual(replay['originPrice'], candles[origin]['close'])
                    self.assertEqual(len(replay['path']), 13)
                    for h, point in enumerate(replay['path']):
                        self.assertEqual(point['horizon'], h)
                        self.assertEqual(point['time'], origin + h * step)
                        self.assertEqual(point['actual'], candles[origin + h * step]['close'])
                    self.assertEqual(replay['path'][0]['predicted'], replay['originPrice'])
                    if origin in history:
                        self.assertEqual(replay['path'][1]['predicted'], history[origin]['predicted'])
                        self.assertEqual(replay['path'][1]['actual'], history[origin]['actual'])

    def test_complete_replay_equals_original_prefix_and_ignores_future_actuals(self):
        for interval in ('15m', '1h', '4h', '1d'):
            with self.subTest(interval=interval):
                full = fixture(interval)
                prefix = copy.deepcopy(full)
                prefix['candles'] = prefix['candles'][:-20]
                original = self.engine.forecast(prefix)
                replay = next(p for p in self.engine.forecast(full)['replayPaths']
                              if p['originTime'] == original['originTime'])
                # The entire original twelve-step forecast is retained. A
                # sequence of re-anchored rolling h1 values cannot pass this.
                for h, point in enumerate(replay['path']):
                    self.assertEqual(point['time'], original['path'][h]['time'])
                    self.assertEqual(point['predicted'], original['path'][h]['mid'])
                modified = copy.deepcopy(full)
                for candle in modified['candles'][-20:]:
                    for key in ('open', 'high', 'low', 'close'):
                        candle[key] *= 1.1
                changed = self.engine.forecast(modified)
                revised = next(p for p in changed['replayPaths'] if p['originTime'] == original['originTime'])
                for h in range(13):
                    self.assertEqual(revised['path'][h]['predicted'], replay['path'][h]['predicted'])
                    self.assertEqual(revised['path'][h]['actual'], replay['path'][h]['actual'] * (1.1 if h else 1))
                rolling_h12 = next(p for p in changed['history'] if p['time'] == revised['path'][12]['time'])
                self.assertNotEqual(revised['path'][12]['predicted'], rolling_h12['predicted'])

    def test_replay_requires_twelve_targets_and_excludes_calibration_origins(self):
        for interval in ('15m', '1h', '4h', '1d'):
            with self.subTest(interval=interval):
                metadata = json.loads((ROOT / f'.cache/training-expanded/{interval}-metadata.json').read_text())
                source = json.loads((ROOT / metadata['sources']['BTCUSDT']['path']).read_text())
                all_candles = fixture(interval, count=len(source))
                cutoff = metadata['boundaryMs'][-1] // 1000
                stop = next(i for i, c in enumerate(all_candles['candles']) if c['time'] >= cutoff)
                incomplete = copy.deepcopy(all_candles)
                incomplete['candles'] = all_candles['candles'][stop-99:stop+12]
                self.assertEqual(self.engine.forecast(incomplete)['replayPaths'], [])
                complete = copy.deepcopy(all_candles)
                complete['candles'] = all_candles['candles'][stop-99:stop+13]
                paths = self.engine.forecast(complete)['replayPaths']
                self.assertEqual(len(paths), 1)
                self.assertEqual(paths[0]['originTime'], cutoff)
                self.assertEqual(paths[0]['path'][12]['actual'], complete['candles'][-1]['close'])

    def test_missing_flow_is_rejected(self):
        for key in ('quoteVolume', 'trades', 'takerBuyVolume', 'takerBuyQuoteVolume'):
            value = fixture('1h')
            del value['candles'][-1][key]
            status, body = self.request(value)
            self.assertEqual(status, 422, body)

    def test_unclosed_candle_is_rejected(self):
        value = fixture('1h')
        value['candles'][-1]['isClosed'] = False
        self.assertEqual(self.request(value)[0], 422)

    def test_malformed_candles_are_rejected(self):
        changes = [lambda p: p['candles'][-1].update(high=1),
                   lambda p: p['candles'][-1].update(trades=1.5),
                   lambda p: p['candles'][-1].update(close=float('nan')),
                   lambda p: p['candles'][-1].update(takerBuyVolume=1e30),
                   lambda p: p['candles'][-1].update(time=p['candles'][-2]['time']),
                   lambda p: p.update(symbol='FAKEUSDT'),
                   lambda p: p.update(interval='5m'),
                   lambda p: p.update(candles=p['candles'][:99])]
        for change in changes:
            value = fixture('1h')
            change(value)
            self.assertEqual(self.request(value)[0], 422)

    def test_future_candle_is_rejected(self):
        value = fixture('1h')
        value['candles'][-1]['time'] = (int(service.time.time()) // 3600 + 1) * 3600
        self.assertEqual(self.request(value)[0], 422)

    def test_fit_and_calibration_origins_are_rejected(self):
        metadata = json.loads((ROOT / '.cache/training-expanded/1h-metadata.json').read_text())
        cutoff = metadata['boundaryMs'][-1] // 1000
        rows = json.loads((ROOT / metadata['sources']['BTCUSDT']['path']).read_text())
        stop = next(i for i,r in enumerate(rows) if int(r[0]) // 1000 >= cutoff)
        value = fixture('1h', count=len(rows))
        value['candles'] = value['candles'][stop-500:stop]
        self.assertEqual(self.request(value)[0], 422)

    def test_cache_includes_past_input_corrections(self):
        value = fixture('1h')
        first = self.engine.forecast(value)
        self.assertIs(first, self.engine.forecast(value))
        changed = copy.deepcopy(value)
        changed['candles'][-10]['trades'] += 1
        self.assertIsNot(first, self.engine.forecast(changed))

    def test_remote_browser_origin_is_rejected(self):
        status, value = self.request(fixture('1h'), headers={'Content-Type':'application/json', 'Origin':'https://example.com'})
        self.assertEqual(status, 403, value)

    def test_request_size_limit(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        connection.putrequest('POST', '/api/b4/forecast')
        connection.putheader('Content-Type', 'application/json')
        connection.putheader('Content-Length', str(service.MAX_REQUEST_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        response.read()
        connection.close()


class B4ForwardServiceTests(unittest.TestCase):
    """Exercise real model/service/cache integration with an isolated ledger.

    Historical fixtures use an explicit simulated server clock solely inside
    tests. The public HTTP API does not accept a caller-supplied issue time.
    """
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ledger_path = Path(self.directory.name) / 'forward.sqlite3'
        self.engine = service.ForecastService(ledger_path=self.ledger_path)
        self.full = fixture('1h')
        self.first = copy.deepcopy(self.full)
        self.first['candles'] = self.first['candles'][:-12]
        self.following = copy.deepcopy(self.full)
        self.following['candles'] = self.following['candles'][:-11]
        self.origin = self.first['candles'][-1]['time']
        self.step = 3600
        self.issued_at = self.origin + self.step + 5

    def test_cache_and_input_revision_preserve_first_issue_without_importing_replay(self):
        first = self.engine.forecast(self.first, now_seconds=self.issued_at)
        self.assertGreater(len(first['history']), 1)
        self.assertGreater(len(first['replayPaths']), 1)
        forward = first['forward']
        self.assertEqual(forward['recordedOrigins'], 1)
        self.assertEqual(forward['eligibleOrigins'], 1)
        self.assertEqual(forward['resolvedNextBarOrigins'], 0)
        self.assertEqual(forward['recent'][0]['issuedAt'], self.issued_at)
        self.assertEqual(len(forward['recent'][0]['path']), 12)
        for h, point in enumerate(forward['recent'][0]['path'], 1):
            self.assertEqual(point['predicted'], first['path'][h]['mid'])
            self.assertNotIn('actual', point)
        cached = self.engine.forecast(self.first, now_seconds=self.issued_at + 20)
        self.assertEqual(cached['forward'], forward)
        corrected = copy.deepcopy(self.first)
        for key in ('open', 'high', 'low', 'close'):
            corrected['candles'][-1][key] *= 1.01
        recomputed = self.engine.forecast(corrected, now_seconds=self.issued_at + 25)
        self.assertNotEqual(recomputed['path'][1]['mid'], first['path'][1]['mid'])
        self.assertEqual(recomputed['forward'], forward)
        with sqlite3.connect(self.ledger_path) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forward_predictions').fetchone()[0], 12)

    def test_new_closed_bar_resolves_h1_and_cached_response_refreshes_summary(self):
        first = self.engine.forecast(self.first, now_seconds=self.issued_at)
        # Merely waiting for the target does not invent its unobserved close.
        waiting = self.engine.forecast(self.first, now_seconds=self.origin + 2*self.step - 1)
        self.assertEqual(waiting['forward']['resolvedNextBarOrigins'], 0)
        resolved = self.engine.forecast(self.following, now_seconds=self.issued_at + self.step)
        summary = resolved['forward']
        self.assertEqual(summary['recordedOrigins'], 2)
        self.assertEqual(summary['resolvedNextBarOrigins'], 1)
        self.assertEqual(summary['resolvedOrigins'], 0)
        self.assertEqual(summary['byHorizon'][1]['samples'], 0)
        actual = self.following['candles'][-1]['close']
        scored = summary['byHorizon'][0]
        self.assertEqual(scored['samples'], 1)
        self.assertAlmostEqual(scored['maePct'], abs(first['path'][1]['mid']/actual-1)*100)
        self.assertAlmostEqual(scored['baselineMaePct'], abs(first['originPrice']/actual-1)*100)
        issued = next(row for row in summary['recent'] if row['originTime'] == self.origin)
        self.assertEqual(issued['issuedAt'], self.issued_at)
        self.assertEqual(issued['path'][0]['actual'], actual)
        self.assertEqual(issued['path'][0]['resolvedAt'], self.issued_at + self.step)
        # Reusing an older input hits the inference cache, but the returned
        # ledger snapshot must reflect the resolution above, without mutation
        # of a response previously returned to another caller.
        refreshed = self.engine.forecast(self.first, now_seconds=self.issued_at + self.step + 1)
        self.assertEqual(refreshed['forward'], summary)
        self.assertEqual(first['forward']['recordedOrigins'], 1)
        self.assertEqual(first['forward']['resolvedNextBarOrigins'], 0)
        correction = copy.deepcopy(self.following)
        for key in ('open', 'high', 'low', 'close'):
            correction['candles'][-1][key] *= 1.1
        corrected = self.engine.forecast(correction, now_seconds=self.issued_at + self.step + 2)
        old = next(row for row in corrected['forward']['recent'] if row['originTime'] == self.origin)
        self.assertEqual(old['path'][0]['actual'], actual)
        self.assertEqual(old['path'][0]['resolvedAt'], self.issued_at + self.step)

    def test_late_forecasts_resolve_but_never_enter_scores(self):
        late = self.engine.forecast(self.first, now_seconds=self.origin + self.step + 61)['forward']
        self.assertEqual(late['lateOrigins'], 1)
        self.assertEqual(late['eligibleOrigins'], 0)
        resolved = self.engine.forecast(self.following, now_seconds=self.issued_at + self.step)['forward']
        original = next(row for row in resolved['recent'] if row['originTime'] == self.origin)
        self.assertEqual(original['eligibility'], 'late')
        self.assertEqual(original['path'][0]['actual'], self.following['candles'][-1]['close'])
        self.assertEqual(resolved['resolvedNextBarOrigins'], 0)
        for score in resolved['byHorizon']:
            self.assertEqual(score['samples'], 0)
            self.assertIsNone(score['maePct'])
            self.assertIsNone(score['directionAccuracy'])

    def test_complete_fixed_origin_resolves_all_twelve_real_targets(self):
        original = self.engine.forecast(self.first, now_seconds=self.issued_at)
        done = self.engine.forecast(self.full, now_seconds=self.issued_at + 12*self.step)['forward']
        self.assertEqual(done['recordedOrigins'], 2)
        self.assertEqual(done['resolvedOrigins'], 1)
        self.assertEqual(done['resolvedNextBarOrigins'], 1)
        completed = next(row for row in done['recent'] if row['originTime'] == self.origin)
        self.assertEqual(completed['originPrice'], original['originPrice'])
        for h, point in enumerate(completed['path'], 1):
            actual = self.full['candles'][-13+h]['close']
            self.assertEqual(point['time'], self.origin+h*self.step)
            self.assertEqual(point['predicted'], original['path'][h]['mid'])
            self.assertEqual(point['actual'], actual)
            score = done['byHorizon'][h-1]
            self.assertEqual(score['samples'], 1)
            self.assertAlmostEqual(score['baselineMaePct'], abs(original['originPrice']/actual-1)*100)

    def test_version_fingerprint_is_stable_on_reload_and_changes_with_calibration(self):
        original = self.engine.forecast(self.first, now_seconds=self.issued_at)['forward']
        self.assertRegex(original['modelVersion'], r'^b4:1h:[0-9a-f]{64}$')
        reloaded = service.ForecastService(ledger_path=self.ledger_path)
        same = reloaded.forecast(self.first, now_seconds=self.issued_at + 1)['forward']
        self.assertEqual(same, original)
        loaded = reloaded.load('1h')['tree']
        changed = {'contract': copy.deepcopy(loaded['contract']), 'models': loaded['models']}
        # A changed effective calibration is an actual model contract change;
        # patch the loader in memory only, leaving all real artifacts intact.
        changed['contract']['calibration']['h1']['intercept'] += .1
        with patch.object(service.TREES, 'load_models', return_value=changed):
            upgraded = service.ForecastService(ledger_path=self.ledger_path)
            new = upgraded.forecast(self.first, now_seconds=self.issued_at + 2)['forward']
        self.assertNotEqual(new['modelVersion'], original['modelVersion'])
        self.assertEqual(new['recordedOrigins'], 1)
        self.assertEqual(new['recent'][0]['issuedAt'], self.issued_at + 2)
        self.assertNotEqual(new['recent'][0]['path'][0]['upProbability'], original['recent'][0]['path'][0]['upProbability'])
        with sqlite3.connect(self.ledger_path) as connection:
            self.assertEqual(connection.execute('SELECT count(DISTINCT model_version) FROM forward_predictions').fetchone()[0], 2)
            self.assertEqual(connection.execute('SELECT count(*) FROM forward_predictions').fetchone()[0], 24)


if __name__ == '__main__':
    unittest.main(verbosity=2)

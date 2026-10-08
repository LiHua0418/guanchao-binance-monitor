"""Deterministic clock/immutability tests for actual forward observations."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from b4_forward import ForwardLedger, INTERVAL_SECONDS


ORIGIN = 1728000000  # UTC midnight, aligned to all supported periods.


def candle(timestamp, close=100.):
    return {'time': timestamp, 'open': close, 'high': close, 'low': close, 'close': close,
            'volume': 10., 'quoteVolume': close*10, 'trades': 20,
            'takerBuyVolume': 5., 'takerBuyQuoteVolume': close*5, 'isClosed': True}


def forecast(origin=ORIGIN, close=100., interval='1h', prediction=101., symbol='BTCUSDT'):
    step = INTERVAL_SECONDS[interval]
    path = [{'time': origin, 'mid': close}]
    for h in range(1, 13):
        point = {'time': origin+h*step, 'mid': prediction}
        if h in (1,3,6,12):
            point['upProbability'] = .7
        path.append(point)
    # Bogus replay must never be imported as live issuance.
    history = [{'time': origin-10*step, 'originTime': origin-11*step,
                'predicted': 500, 'actual': 500}]
    return {'modelId':'guanchao-b4','symbol':symbol,'interval':interval,
            'originTime':origin,'originPrice':close,'path':path,'history':history}


class ForwardLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'forward.sqlite3'
        self.ledger = ForwardLedger(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def observe(self, result=None, candles=None, now=None, version='version-a'):
        result = forecast() if result is None else result
        step = INTERVAL_SECONDS[result['interval']]
        candles = [candle(result['originTime'], result['originPrice'])] if candles is None else candles
        now = result['originTime'] + step + 5 if now is None else now
        return self.ledger.observe(result, candles, step, version, now)

    def test_first_issued_clock_and_forecasts_are_immutable(self):
        a = self.observe()
        revised = forecast(prediction=999.)
        revised['path'][1]['upProbability'] = .1
        b = self.observe(revised, now=ORIGIN+3600+25)
        self.assertEqual(a, b)
        self.assertEqual(b['recordedOrigins'], 1)
        self.assertEqual(len(b['recent'][0]['path']), 12)
        self.assertEqual(b['recent'][0]['issuedAt'], ORIGIN+3600+5)
        self.assertEqual(b['recent'][0]['path'][0]['predicted'], 101.)
        self.assertEqual(b['recent'][0]['path'][0]['upProbability'], .7)
        self.assertEqual(b['byHorizon'][0]['samples'], 0)
        self.assertIsNone(b['byHorizon'][0]['maePct'])

    def test_late_origin_never_enters_scoring(self):
        late = self.observe(now=ORIGIN+3600+61)
        self.assertEqual(late['eligibleOrigins'], 0)
        self.assertEqual(late['lateOrigins'], 1)
        self.assertEqual(late['recent'][0]['issuedAt'], ORIGIN+3600+61)
        now = ORIGIN+2*3600+10
        later = self.observe(forecast(ORIGIN+3600, 102.),
                             [candle(ORIGIN),candle(ORIGIN+3600,102.)], now)
        self.assertEqual(later['byHorizon'][0]['samples'], 0)
        previous = next(r for r in later['recent'] if r['originTime'] == ORIGIN)
        self.assertEqual(previous['path'][0]['actual'], 102.)
        self.assertEqual(previous['eligibility'], 'late')

    def test_no_backdating_and_no_history_import(self):
        now = ORIGIN+40*3600
        result = self.observe(now=now)
        self.assertEqual(result['recordedOrigins'], 1)
        self.assertEqual(result['recent'][0]['issuedAt'], now)
        self.assertEqual(result['recent'][0]['latencySeconds'], 39*3600)
        self.assertEqual(result['recent'][0]['eligibility'], 'late')
        self.assertEqual(result['byHorizon'][0]['samples'], 0)

    def test_target_open_does_not_resolve_until_closed(self):
        self.observe()
        repeated = self.observe(now=ORIGIN+2*3600-1)
        self.assertEqual(repeated['resolvedNextBarOrigins'], 0)
        with self.assertRaises(ValueError):
            self.observe(forecast(ORIGIN+3600,102.), [candle(ORIGIN),candle(ORIGIN+3600,102.)], ORIGIN+2*3600-1)
        finished = self.observe(forecast(ORIGIN+3600,102.),
                                [candle(ORIGIN),candle(ORIGIN+3600,102.)], ORIGIN+2*3600)
        h1 = finished['byHorizon'][0]
        self.assertEqual(h1['samples'], 1)
        self.assertAlmostEqual(h1['maePct'], abs(101/102-1)*100)
        self.assertAlmostEqual(h1['baselineMaePct'], abs(100/102-1)*100)
        self.assertAlmostEqual(h1['skillPct'], 50.)
        self.assertEqual(h1['directionAccuracy'], 1.)
        self.assertAlmostEqual(h1['brier'], .09)
        self.assertEqual(finished['resolvedOrigins'], 0)

    def test_actual_first_seen_and_resolved_clock_are_immutable(self):
        self.observe()
        first_clock = ORIGIN+2*3600+7
        self.observe(forecast(ORIGIN+3600,102.), [candle(ORIGIN),candle(ORIGIN+3600,102.)], first_clock)
        changed = self.observe(forecast(ORIGIN+3600,150.), [candle(ORIGIN),candle(ORIGIN+3600,150.)], first_clock+10)
        previous = next(r for r in changed['recent'] if r['originTime'] == ORIGIN)
        self.assertEqual(previous['path'][0]['actual'], 102.)
        self.assertEqual(previous['path'][0]['resolvedAt'], first_clock)
        with sqlite3.connect(self.path) as connection:
            row = connection.execute('SELECT close,observed_at FROM forward_observations WHERE target_time=?', (ORIGIN+3600,)).fetchone()
        self.assertEqual(row, (102., first_clock))

    def test_all_supported_intervals_have_disclosed_latency_window(self):
        for interval, step in INTERVAL_SECONDS.items():
            with self.subTest(interval=interval):
                a = self.observe(forecast(interval=interval), now=ORIGIN+step+60)
                self.assertEqual(a['eligibilityWindowSeconds'], 60.)
                self.assertEqual(a['eligibleOrigins'], 1)
                self.assertEqual(a['recent'][0]['latencySeconds'], 60.)
                b = self.observe(forecast(interval=interval), now=ORIGIN+step+60.001, version='late-version')
                self.assertEqual(b['lateOrigins'], 1)

    def test_model_versions_symbols_and_intervals_are_separate(self):
        self.observe()
        for result, version in [(forecast(), 'version-b'),
                                (forecast(symbol='ETHUSDT'), 'version-a'),
                                (forecast(interval='15m'), 'version-a')]:
            summary = self.observe(result, version=version)
            self.assertEqual(summary['recordedOrigins'], 1)
            self.assertEqual(summary['scope'], {'symbol':result['symbol'],'interval':result['interval']})
            self.assertEqual(summary['modelVersion'], version)

    def test_real_predictions_survive_process_reload(self):
        self.observe()
        self.ledger = ForwardLedger(self.path)
        again = self.observe(forecast(prediction=999), now=ORIGIN+3600+61)
        self.assertEqual(again['eligibleOrigins'], 1)
        self.assertEqual(again['recent'][0]['path'][0]['predicted'],101.)
        self.assertEqual(again['recent'][0]['issuedAt'],ORIGIN+3600+5)

    def test_twelve_horizons_resolve_and_recent_is_capped(self):
        prices = [100.+i for i in range(16)]
        summary = None
        for i in range(16):
            rows = [candle(ORIGIN+j*3600, prices[j]) for j in range(i+1)]
            summary = self.observe(forecast(ORIGIN+i*3600, prices[i]), rows)
        self.assertEqual(summary['recordedOrigins'],16)
        self.assertEqual(summary['resolvedOrigins'],4)
        self.assertEqual(summary['pendingOrigins'],12)
        self.assertEqual(summary['resolvedNextBarOrigins'],15)
        self.assertEqual(summary['byHorizon'][11]['samples'],4)
        self.assertEqual(len(summary['recent']),12)
        self.assertEqual(summary['recent'][0]['originTime'],ORIGIN+15*3600)

    def test_service_matrix_input_uses_milliseconds(self):
        row = [ORIGIN*1000,100.,100.,100.,100.,10.,1000.,20,5.,500.]
        result = self.observe(candles=[row])
        self.assertEqual(result['recordedOrigins'],1)
        self.assertEqual(result['recent'][0]['originTime'],ORIGIN)

    def test_invalid_or_unclosed_input_is_rejected_without_writes(self):
        cases = []
        bad = [candle(ORIGIN)]; bad[0]['isClosed'] = False; cases.append((forecast(),bad,ORIGIN+3605))
        cases.append((forecast(),[candle(ORIGIN)],ORIGIN+3599))
        cases.append((forecast(),[candle(ORIGIN),candle(ORIGIN+3600)],ORIGIN+7205))
        cases.append((forecast(),[candle(ORIGIN),candle(ORIGIN)],ORIGIN+3605))
        bad_result = forecast(); bad_result['path'][2]['time'] += 1
        cases.append((bad_result,[candle(ORIGIN)],ORIGIN+3605))
        bad_result = forecast(); bad_result['path'][1]['mid'] = float('nan')
        cases.append((bad_result,[candle(ORIGIN)],ORIGIN+3605))
        for result, rows, now in cases:
            with self.assertRaises(ValueError):
                self.observe(result,rows,now)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forward_predictions').fetchone()[0],0)

    def test_sql_triggers_reject_overwriting_forecast_and_actual(self):
        self.observe()
        self.observe(forecast(ORIGIN+3600,102.),[candle(ORIGIN),candle(ORIGIN+3600,102.)])
        for statement in ["UPDATE forward_predictions SET predicted=999 WHERE horizon=1",
                          "UPDATE forward_predictions SET issued_at=1 WHERE horizon=1",
                          "UPDATE forward_predictions SET actual=999 WHERE actual IS NOT NULL",
                          "UPDATE forward_observations SET close=999"]:
            with sqlite3.connect(self.path) as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(statement)

    def test_flat_actual_excluded_from_direction_metric(self):
        self.observe()
        summary = self.observe(forecast(ORIGIN+3600,100.),[candle(ORIGIN),candle(ORIGIN+3600,100.)])
        self.assertEqual(summary['byHorizon'][0]['samples'],1)
        self.assertEqual(summary['byHorizon'][0]['directionSamples'],0)
        self.assertIsNone(summary['byHorizon'][0]['directionAccuracy'])
        self.assertIsNone(summary['byHorizon'][0]['skillPct'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

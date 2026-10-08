"""Immutable, local, forward-only forecast ledger using Python's sqlite3.

The caller supplies the real server clock to ``observe``; injecting a clock is
supported for deterministic tests, never to backdate a historical forecast.
Only result.path at the current request's final closed origin is recorded.
result.history is deliberately ignored. No networking or scheduler is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import math
from numbers import Real
from pathlib import Path
import re
import sqlite3
import threading


INTERVAL_SECONDS = {'15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
SCHEMA_VERSION = 1


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def integer_time(value, name):
    number = finite(value, name, positive=True)
    if number != int(number):
        raise ValueError(f'{name} must be an integer timestamp in seconds')
    return int(number)


def closed_observations(candles, step, now):
    """Accept validated service Nx10 rows (ms) or request mappings (seconds)."""
    observations = []
    for row in candles:
        if isinstance(row, Mapping):
            if row.get('isClosed') is not True:
                raise ValueError('Forward ledger accepts only explicitly closed candles')
            timestamp = integer_time(row.get('time'), 'candle.time')
            close = finite(row.get('close'), 'candle.close', positive=True)
        else:
            if len(row) != 10:
                raise ValueError('Service matrix rows must have ten validated candle fields')
            timestamp_ms = finite(row[0], 'candle.timeMs', positive=True)
            timestamp = integer_time(timestamp_ms / 1000, 'candle.time')
            close = finite(row[4], 'candle.close', positive=True)
        if timestamp % step or timestamp + step > now:
            raise ValueError('A candle must be aligned and fully closed by the observation clock')
        if observations and timestamp != observations[-1][0] + step:
            raise ValueError('Closed observations must be ordered, unique and continuous')
        observations.append((timestamp, close))
    if not observations:
        raise ValueError('At least one closed origin candle is required')
    return observations


class ForwardLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self._connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS forward_metadata (
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forward_observations (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    target_time INTEGER NOT NULL,
                    close REAL NOT NULL CHECK (close > 0),
                    observed_at REAL NOT NULL,
                    PRIMARY KEY (symbol, interval, target_time)
                );
                CREATE TABLE IF NOT EXISTS forward_predictions (
                    model_version TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    origin_time INTEGER NOT NULL,
                    horizon INTEGER NOT NULL CHECK (horizon BETWEEN 1 AND 12),
                    target_time INTEGER NOT NULL,
                    issued_at REAL NOT NULL,
                    origin_price REAL NOT NULL CHECK (origin_price > 0),
                    predicted REAL NOT NULL CHECK (predicted > 0),
                    up_probability REAL,
                    eligibility TEXT NOT NULL CHECK (eligibility IN ('eligible', 'late')),
                    latency_seconds REAL NOT NULL CHECK (latency_seconds >= 0),
                    actual REAL CHECK (actual > 0),
                    resolved_at REAL,
                    PRIMARY KEY (model_version, symbol, interval, origin_time, horizon),
                    CHECK (up_probability IS NULL OR up_probability BETWEEN 0 AND 1),
                    CHECK ((actual IS NULL AND resolved_at IS NULL) OR
                           (actual IS NOT NULL AND resolved_at IS NOT NULL))
                );
                CREATE INDEX IF NOT EXISTS forward_pending_targets
                    ON forward_predictions (symbol, interval, target_time)
                    WHERE actual IS NULL;
                CREATE TRIGGER IF NOT EXISTS forward_prediction_immutable
                    BEFORE UPDATE OF model_version, symbol, interval, origin_time,
                        horizon, target_time, issued_at, origin_price, predicted,
                        up_probability, eligibility, latency_seconds
                    ON forward_predictions
                    BEGIN SELECT RAISE(ABORT, 'Issued forecasts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS forward_actual_immutable
                    BEFORE UPDATE OF actual, resolved_at ON forward_predictions
                    WHEN OLD.actual IS NOT NULL
                    BEGIN SELECT RAISE(ABORT, 'Resolved actuals are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS forward_observation_immutable
                    BEFORE UPDATE ON forward_observations
                    BEGIN SELECT RAISE(ABORT, 'Observed closes are immutable'); END;
            ''')
            row = connection.execute('SELECT schema_version FROM forward_metadata').fetchone()
            if row is None:
                connection.execute('INSERT INTO forward_metadata VALUES (?)', (SCHEMA_VERSION,))
            elif row[0] != SCHEMA_VERSION:
                raise ValueError(f'Unsupported forward-ledger schema {row[0]}')

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA busy_timeout=10000')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def observe(self, result, candles, intervalSeconds, modelVersion, nowSeconds):
        """Record once, resolve from currently supplied closed bars, summarize.

        ``issuedAt`` is exactly nowSeconds. Eligible means issuance at or after
        the origin closes, with latency <= min(10% of a bar, 60 seconds). For
        h1 this is a forecast issued up to 60 seconds *after* the target opens,
        not a promise of having forecast an entire untouched bar. Every score
        additionally requires issuance strictly before that target closes.

        Results are scoped to this exact model version, symbol and interval.
        ``resolvedOrigins`` counts origins with all 12 targets resolved;
        ``resolvedNextBarOrigins`` counts eligible origins with h1 resolved.
        ``byHorizon`` scores only eligible forecasts; late rows remain visible
        in recent records and totals but never enter any accuracy metric.
        """
        if not isinstance(result, Mapping):
            raise ValueError('Expected a forecast response object')
        symbol, interval = result.get('symbol'), result.get('interval')
        if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9]{5,24}', symbol):
            raise ValueError('Invalid forecast symbol')
        if interval not in INTERVAL_SECONDS or intervalSeconds != INTERVAL_SECONDS[interval]:
            raise ValueError('Forecast interval and intervalSeconds disagree')
        if not isinstance(modelVersion, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}', modelVersion):
            raise ValueError('A stable, nonempty model artifact version is required')
        step = INTERVAL_SECONDS[interval]
        now = finite(nowSeconds, 'nowSeconds', positive=True)
        origin = integer_time(result.get('originTime'), 'originTime')
        origin_price = finite(result.get('originPrice'), 'originPrice', positive=True)
        observations = closed_observations(candles, step, now)
        if origin != observations[-1][0] or not math.isclose(origin_price, observations[-1][1], rel_tol=1e-12):
            raise ValueError('Forecast must match the final closed candle of this request')
        path = result.get('path')
        if not isinstance(path, list) or len(path) != 13:
            raise ValueError('Expected origin plus twelve forecast horizons')
        if not isinstance(path[0], Mapping):
            raise ValueError('Path origin must be an object')
        if integer_time(path[0].get('time'), 'path origin time') != origin:
            raise ValueError('Path origin and forecast origin disagree')
        if not math.isclose(finite(path[0].get('mid'), 'path origin price', positive=True), origin_price, rel_tol=1e-12):
            raise ValueError('Path origin price and candle close disagree')
        latency = now - (origin + step)
        if latency < 0:
            raise ValueError('Cannot issue a forecast before its input origin closes')
        window = min(.1 * step, 60.)
        eligibility = 'eligible' if latency <= window else 'late'
        forecast_rows = []
        for horizon, point in enumerate(path[1:], 1):
            if not isinstance(point, Mapping):
                raise ValueError('Each path point must be an object')
            target = integer_time(point.get('time'), 'target time')
            if target != origin + horizon * step:
                raise ValueError('Targets must be consecutive per-interval horizons')
            predicted = finite(point.get('mid'), 'predicted price', positive=True)
            probability = point.get('upProbability')
            if probability is not None:
                probability = finite(probability, 'upProbability')
                if not 0 <= probability <= 1:
                    raise ValueError('upProbability must be in [0,1]')
            forecast_rows.append((modelVersion, symbol, interval, origin, horizon, target,
                                  now, origin_price, predicted, probability, eligibility, latency))
        with self.lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.executemany('''
                INSERT OR IGNORE INTO forward_predictions
                (model_version,symbol,interval,origin_time,horizon,target_time,issued_at,
                 origin_price,predicted,up_probability,eligibility,latency_seconds)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ''', forecast_rows)
            connection.executemany('''
                INSERT OR IGNORE INTO forward_observations
                (symbol,interval,target_time,close,observed_at) VALUES (?,?,?,?,?)
            ''', [(symbol, interval, timestamp, close, now) for timestamp, close in observations])
            # Resolve only targets present in this current closed-bar request.
            # The first observed close stays authoritative if a later exchange
            # backfill revises the same timestamp's market data.
            connection.executemany('''
                UPDATE forward_predictions
                   SET actual=(SELECT close FROM forward_observations
                               WHERE symbol=? AND interval=? AND target_time=?),
                       resolved_at=?
                 WHERE symbol=? AND interval=? AND target_time=? AND actual IS NULL
                   AND target_time + ? <= ?
            ''', [(symbol, interval, timestamp, now, symbol, interval, timestamp, step, now)
                  for timestamp, _ in observations])
            rows = connection.execute('''
                SELECT * FROM forward_predictions
                 WHERE model_version=? AND symbol=? AND interval=?
                 ORDER BY origin_time, horizon
            ''', (modelVersion, symbol, interval)).fetchall()
            summary = self._summary(rows, modelVersion, symbol, interval, step, window)
        return summary

    @staticmethod
    def _summary(rows, version, symbol, interval, step, window):
        origins = {}
        for row in rows:
            origins.setdefault(row['origin_time'], []).append(row)
        resolved_origins = sum(all(r['actual'] is not None for r in group) and len(group) == 12
                               for group in origins.values())
        by_horizon = []
        for horizon in range(1, 13):
            measured = [r for r in rows if r['horizon'] == horizon and r['eligibility'] == 'eligible'
                        and r['actual'] is not None and r['issued_at'] < r['target_time'] + step]
            n = len(measured)
            mae = sum(abs(r['predicted'] / r['actual'] - 1) * 100 for r in measured) / n if n else None
            baseline = sum(abs(r['origin_price'] / r['actual'] - 1) * 100 for r in measured) / n if n else None
            directional = [r for r in measured if r['actual'] != r['origin_price']]
            probabilities = [r for r in directional if r['up_probability'] is not None]
            direction_hits = sum((r['predicted'] > r['origin_price']) == (r['actual'] > r['origin_price'])
                                 and r['predicted'] != r['origin_price'] for r in directional)
            probability_hits = sum((r['up_probability'] >= .5) == (r['actual'] > r['origin_price'])
                                   for r in probabilities)
            by_horizon.append({
                'horizon': horizon, 'samples': n, 'maePct': mae, 'baselineMaePct': baseline,
                'skillPct': 100 * (1 - mae / baseline) if n and baseline > 0 else None,
                'directionAccuracy': direction_hits / len(directional) if directional else None,
                'directionSamples': len(directional),
                'probabilityDirectionAccuracy': probability_hits / len(probabilities) if probabilities else None,
                'probabilitySamples': len(probabilities),
                'brier': sum((r['up_probability'] - (r['actual'] > r['origin_price'])) ** 2
                             for r in probabilities) / len(probabilities) if probabilities else None,
            })
        recent = []
        for origin in sorted(origins, reverse=True)[:12]:
            group = origins[origin]
            first = group[0]
            path = []
            for row in group:
                point = {'time': row['target_time'], 'horizon': row['horizon'], 'predicted': row['predicted']}
                if row['up_probability'] is not None:
                    point['upProbability'] = row['up_probability']
                if row['actual'] is not None:
                    point.update(actual=row['actual'], resolvedAt=row['resolved_at'])
                path.append(point)
            recent.append({'originTime': origin, 'originPrice': first['origin_price'],
                           'issuedAt': first['issued_at'], 'eligibility': first['eligibility'],
                           'latencySeconds': first['latency_seconds'], 'path': path})
        return {
            'modelVersion': version, 'scope': {'symbol': symbol, 'interval': interval},
            'recordedOrigins': len(origins),
            'eligibleOrigins': sum(group[0]['eligibility'] == 'eligible' for group in origins.values()),
            'resolvedOrigins': resolved_origins, 'pendingOrigins': len(origins) - resolved_origins,
            'resolvedNextBarOrigins': by_horizon[0]['samples'],
            'lateOrigins': sum(group[0]['eligibility'] == 'late' for group in origins.values()),
            'eligibilityWindowSeconds': window,
            'eligibilityMeaning': 'Issued after the origin closed, no later than min(10% of a bar, 60 seconds) after the next bar opened; h1 is a forecast with this disclosed delay, not a whole-bar advance forecast.',
            'recordMeaning': 'First issued forecast and first observed actual close are immutable. Late origins are recorded but excluded from all score metrics. Historical replay is never imported.',
            'resolvedMeaning': 'resolvedOrigins means all 12 horizons resolved; resolvedNextBarOrigins means eligible h1 forecasts resolved.',
            'byHorizon': by_horizon, 'recent': recent,
        }

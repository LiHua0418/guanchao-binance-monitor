# Local B4 inference service

Start from the repository root in PowerShell:

```powershell
.venv-training/Scripts/python.exe scripts/serve-b4.py
```

The service binds only `127.0.0.1:8765`. It loads the frozen CatBoost return,
direction and unsigned-magnitude artifacts from `reports/experiments/b4-trees`
and `reports/experiments/b4-magnitude`. Model SHA256 values and feature contracts
are checked before first inference. It does not train, download candles, switch
models based on final-period results, or write deployed model weights. Stop with
Ctrl+C. Restart the process after intentionally replacing model artifacts.

`GET /api/b4/health` reports artifact availability and the loaded intervals.
`POST /api/b4/forecast` accepts JSON:

```json
{
  "symbol": "BTCUSDT",
  "interval": "1h",
  "candles": [
    {
      "time": 1791457200,
      "open": 82000,
      "high": 82100,
      "low": 81900,
      "close": 82050,
      "volume": 100,
      "quoteVolume": 8205000,
      "trades": 1000,
      "takerBuyVolume": 52,
      "takerBuyQuoteVolume": 4266600,
      "isClosed": true
    }
  ]
}
```

The example shows one candle's format; an actual request needs 100–2,000
continuous closed candles. Times are the candle **open** time in UTC seconds.
Supported symbols are BTCUSDT, ETHUSDT, BNBUSDT and SOLUSDT; intervals are 15m,
1h, 4h and 1d. Every quote-volume, trade-count and taker-volume field is required.
Missing fields are rejected rather than imputed. Nonfinite values, bad OHLC,
unclosed/future bars, gaps, duplicates, invalid volumes and a forecast origin
before the fit/calibration cutoff are rejected. Maximum request size is 2 MiB.

The response follows `src/lib/b4.ts`: a 13-point `path` (origin plus 12 separately
trained future heads), up to 240 consecutive next-bar `history` points, up to
240 complete fixed-origin `replayPaths`, evaluation diagnostics, and a `forward`
summary when the production ledger is enabled. Each replay path contains h0 and
all 12 predictions made using that origin's prefix only, plus their subsequently
observed actual closes. Consecutive origins remain one candle apart. For all 240
complete paths supply at least 351 closed candles, with enough history after the
model's calibration cutoff. A daily model may return fewer eligible origins.

- `mid` is the unamplified conditional mean price. It may remain close to the
  current price when direction is not predictable.
- `upScenario` and `downScenario` are symmetric price scenarios at plus/minus the
  separately predicted **unsigned absolute log-return magnitude**. They are not
  confidence bounds or a claim that either path will occur.
- `lower` and `upper` are the return model's 10th/90th calibration-residual
  price quantiles. The historical calibration is not a future coverage guarantee.
- `upProbability` appears only at horizons 1, 3, 6 and 12, where independent
  classification heads exist. Direction remains experimental.
- `expectedMovePct` is `100 * expm1(expected absolute cumulative log return)`.
  Historical `actualMovePct` uses the same nonnegative display transformation.
  These percentages are not expected signed returns.
- History uses each origin's inputs only and excludes all origins before the
  training/calibration cutoff. It reconstructs frozen-model predictions; it is
  **not a record of forecasts issued live**.
- The panel compares fixed-origin `replayPaths` with actual closes and a flat
  baseline that retains that origin's close for all 12 steps. Rolling next-bar
  `history` receives new actual inputs every bar and is not a comparable 12-step
  forecast; it is retained for diagnostics, not displayed as B4's default replay.

Production startup enables `.cache/forward-evaluation.sqlite3`. Its first-issued
forecasts are append-only and isolated by model contract hash, symbol, interval,
origin and horizon. No historical reconstruction is backfilled into the ledger.
`issuedAt` is the real server time after inference. Only forecasts issued within
60 seconds of the origin's close (also capped at 10% of the interval) are eligible;
late records remain visible but never enter scores. This disclosed delay means
h1 is not a guarantee of an entire candle's advance notice. Targets settle only
after their close and receipt of actual closed candles. Both model and flat
baseline use identical eligible samples and actual target prices as denominators.
Empty metrics remain null. Local callers supply the candles, so this is a local
reproducibility ledger, not an independent market-data attestation.

The panel must be open and requesting inference to record an origin; the service
does not continuously collect market data. Missing first issues are not invented
later. Repeated requests and cache hits can settle existing records but cannot
rewrite first-issued predictions or their issue times. Cache entries omit ledger
summaries, which are recomputed per request. Importing `ForecastService` does not
create a ledger; integration tests opt into a temporary SQLite path.

Evaluation refers to the pooled four-symbol **next-bar** historical period.
Magnitude MSE/MAE improvements are fractions versus a causal rolling 20-bar
mean absolute return baseline. `heldoutSkill` describes positive improvements
with positive block-bootstrap lower bounds against that baseline; it never
selects or changes a model. Direction balanced accuracy is compared with the
opposite of the prior candle's direction label, while Brier score is compared
with the earlier training direction prior. The recent evaluation dates were
already inspected in previous experiments and are not untouched future evidence.

The server caches loaded models and up to 64 complete closed-input requests.
Every candle field participates in the request fingerprint, so corrections to
old input bars invalidate the cache. Open-bar updates must be filtered by the
caller; the server rejects requests containing them. Only loopback Host headers
and localhost/127.0.0.1 browser origins are accepted.

Run real-artifact and HTTP tests:

```powershell
.venv-training/Scripts/python.exe scripts/test_b4_service.py
.venv-training/Scripts/python.exe scripts/test_b4_forward.py
```

The tests verify all four intervals, causal prefix parity, rejection of malformed
and unclosed inputs, cutoff exclusion, request limits, and cache invalidation.
They also check fixed-origin 12-step prefix parity, future perturbations, exact
target alignment, first-issue immutability, late exclusion, version isolation,
settlement timing and fixed-origin baseline scores.
They start a temporary loopback server on an automatically assigned port and
shut it down after the run. The production service always uses port 8765.

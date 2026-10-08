# B4 unsigned absolute-move prediction

All four next-bar magnitude models beat the prespecified volatility-scaled baseline and both rolling-absolute-return baselines on this historical reevaluation. This is improvement in forecasting **how much price may move**, not a validated prediction of the signed future price path.

Each interval has 12 independently trained heads. Input is the last fully closed candle and earlier candles. The h1 selection gate does not validate all later horizons. In particular, h12 on 4h has worse MAE and h12 on 1d has worse MSE and MAE than the volatility baseline.

All model choices were frozen from three expanding temporal validation folds; calibration and final evaluation occur later in time. The final dates overlap previously inspected experiments and are not fresh untouched evidence. The tables below do not choose or replace any model.

## Next-bar results

| Interval | Observations / distinct times | CV passed | MSE reduction vs vol baseline | MAE reduction vs vol baseline | Absolute-move IC | Model magnitude MAE, pp |
|---|---:|---|---:|---:|---:|---:|
| 15m | 15888 / 3972 | True | 14.06% | 9.23% | 0.3819 | 0.10501 |
| 1h | 11888 / 2972 | True | 12.71% | 8.58% | 0.3862 | 0.22348 |
| 4h | 5288 / 1322 | True | 6.74% | 4.29% | 0.3495 | 0.48551 |
| 1d | 788 / 197 | True | 7.83% | 4.18% | 0.2543 | 1.22189 |

The volatility baseline is origin sigma48 multiplied by the training-only mean normalized absolute target. MAE is error in absolute log-return magnitude, expressed in percentage points; it must not be reported as an improvement in signed-price MAE.

## Stronger and simpler comparator audit

Every cell is relative MSE reduction / relative MAE reduction. Positive means lower model error. No comparator is selected to alter predictions.

| Interval | Training mean | Volatility Gaussian factor | Rolling abs return20 | Rolling abs return48 |
|---|---:|---:|---:|---:|
| 15m | 19.18% / 21.44% | 14.72% / 10.36% | 9.45% / 4.03% | 10.17% / 4.67% |
| 1h | 22.76% / 25.24% | 13.97% / 10.58% | 9.74% / 3.45% | 9.91% / 4.23% |
| 4h | 38.35% / 36.12% | 8.09% / 7.00% | 8.02% / 3.07% | 6.49% / 2.51% |
| 1d | 34.99% / 32.53% | 9.14% / 6.06% | 7.59% / 1.47% | 7.55% / 3.05% |

## Longer-horizon boundary

| Interval | Horizon (bars) | MSE reduction vs vol baseline | MAE reduction vs vol baseline | Absolute-move IC |
|---|---:|---:|---:|---:|
| 15m | 3 | 14.20% | 9.15% | 0.3572 |
| 15m | 6 | 15.86% | 9.55% | 0.3463 |
| 15m | 12 | 17.17% | 11.12% | 0.3403 |
| 1h | 3 | 11.92% | 7.63% | 0.3606 |
| 1h | 6 | 11.44% | 6.57% | 0.3261 |
| 1h | 12 | 10.86% | 4.38% | 0.3043 |
| 4h | 3 | 3.84% | 0.53% | 0.3147 |
| 4h | 6 | 1.86% | -1.42% | 0.2941 |
| 4h | 12 | 3.30% | -4.23% | 0.2791 |
| 1d | 3 | 4.91% | 1.32% | 0.1515 |
| 1d | 6 | 2.63% | -2.59% | -0.0257 |
| 1d | 12 | -3.73% | -6.57% | -0.0733 |

## Uncertainty and consistency

Block bootstrap keeps all contemporaneous symbols together and resamples 48-bar time blocks, 500 draws. These are descriptive intervals for an inspected period. The daily evaluation contains only 197 distinct origin times (five 48-bar blocks); its interval is especially weak evidence about future market regimes.

| Interval | h1 MSE reduction vs vol: bootstrap 95% | h1 MSE reduction vs rolling abs20: bootstrap 95% | Coins improving MSE / MAE vs rolling abs20 |
|---|---:|---:|---:|
| 15m | [10.48%, 17.35%] | [6.63%, 12.04%] | 4/4 / 4/4 |
| 1h | [8.62%, 17.62%] | [6.95%, 12.12%] | 4/4 / 4/4 |
| 4h | [4.37%, 9.13%] | [5.45%, 10.58%] | 4/4 / 4/4 |
| 1d | [4.94%, 10.20%] | [0.68%, 10.47%] | 4/4 / 3/4 |

## Runtime contract and UI interpretation

- Module: `scripts/train-b4-magnitude.py`; load once with `load_models(interval)` and call `predict_features(loaded, raw_features, symbols, origin_volatility, last_close=None)`.
- Inputs: raw causal features `[N,57]` in the frozen contract order; symbol indices `[N]` BTC=0, ETH=1, BNB=2, SOL=3; positive causal sigma48 `[N]`; optional last closed prices `[N]`.
- `absLogReturns[N,12]` estimates unsigned absolute cumulative log return independently at every future bar. `expectedMovePct=100*expm1(absLogReturns)` is a display conversion, not the expected signed return or exactly the expected absolute simple return.
- `cvPassed` and `heldoutSkill` concern **h1 only**. `skillScope` explicitly states this; `historicalDiagnosticsByHorizon` carries later-horizon results without selecting models from them.
- Optional `upScenarioPrices` and `downScenarioPrices` equal `close*exp(+/-absLogReturn)`. Label them “±预估变动幅度情景”; they are neither confidence bounds nor conditional price paths. The up/down probability model is a separate model.
- Suggested release: expose h1 expected magnitude as an experimental trained signal at every selected bar interval. Display later horizons as experimental magnitude scenarios with their evidence limitations. Keep signed mean-price forecasts separately identified as unverified; do not multiply direction probability and magnitude to manufacture a blue price line.

## Artifact verification

Verification reloads all 48 magnitude heads, checks CBM SHA256 values, reproduces every stored test prediction, and checks nonnegative finite magnitudes. For each of 16 source series it also checks source SHA256 and three origin prefixes; changing every future candle leaves the origin features, volatility and model prediction unchanged. See `verification.json` and per-interval reports.

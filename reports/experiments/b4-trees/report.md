# B4 CatBoost direction and amplitude experiment

Causal features, three expanding temporal validation folds, separate probability calibration. No changes to deployed model weights.

Final historical reevaluation can overlap previous experiments. It is not fresh untouched evidence; no candidate selection uses its results.

| Interval | CV return eligible | CV direction eligible | Test BA | Uncalibrated BA | Return IC | R² vs zero | Pred/actual std | Model price MAE % | Flat price MAE % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 15m | False | True | 0.5206 | 0.5225 | 0.0166 | 0.00002 | 0.0025 | 0.15462 | 0.15462 |
| 1h | False | True | 0.5276 | 0.5254 | 0.0156 | -0.00006 | 0.0335 | 0.31574 | 0.31541 |
| 4h | True | True | 0.5236 | 0.5187 | -0.0068 | -0.00379 | 0.0592 | 0.67953 | 0.67588 |
| 1d | False | False | 0.5000 | 0.5016 | 0.0203 | 0.00186 | 0.0414 | 1.76456 | 1.76067 |

No amplitude floor or artificial curve amplification is applied. Conditional mean returns are allowed to be small; classification and uncertainty remain separate quantities.

Each interval directory contains the preregistered protocol, frozen validation selection, portable JSON and CBM models, model contract, frozen calibration, per-symbol diagnostics, block-bootstrap intervals and final predictions.

## Direction comparator audit

The main direction head is h1. The simple reversal rule is the opposite sign of the latest completed candle; it is a descriptive comparator added after the frozen predictions, not a test-selected replacement. Daily h1 direction did not pass the validation gate; later daily direction heads have no independent validation gate.

| Interval | h1 direction BA | Reversal BA | Brier model | Brier training prior | BA bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| 15m | 52.06% | 51.14% | 0.248947 | 0.250002 | [51.19%, 53.01%] |
| 1h | 52.76% | 52.35% | 0.249073 | 0.249877 | [51.54%, 53.93%] |
| 4h | 52.36% | 53.35% | 0.249404 | 0.250006 | [50.84%, 53.80%] |
| 1d | 50.00% | 50.63% | 0.253539 | 0.249785 | [50.00%, 50.00%] |

15m and 1h show modest historical direction information; 4h classification is weaker than the simple reversal comparator despite beating the training-prior Brier score. Daily direction remains unvalidated. Direction evidence is insufficient to claim a useful signed-price path, and mean-return regression remains weak on all intervals. Later classification horizons must not inherit the h1 validation label.

For a product integration, label h1 probabilities experimental and expose calibrated up probability separately from any magnitude/scenario display. The separately trained `b4-magnitude` family estimates unsigned movement; multiplying it by the sign/probability head does not produce a validated point forecast.

Artifact verification checks all source/model hashes, reproduces all stored predictions and compares origin-prefix features/predictions with the same origin after changing future candles. No deployed B3 model was overwritten by this experiment.

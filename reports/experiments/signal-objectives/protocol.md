# B3 signal objective experiment protocol

Recorded before training. Model selection uses validation only.

- **createdAt**: 2026-10-08T12:09:34.338645+00:00
- **cutoffMs**: 1791458027830
- **objectives**: ['huber', 'mse', 'mse-direction']
- **seeds**: [20261008, 20261009, 20261010]
- **maxEpochs**: 24
- **cpuThreads**: 6
- **architecture**: unchanged CNN-LSTM-Attention 5045 parameters
- **trainingLoss**: normalized returns; h1 weight .6, later .4/11; Huber beta=.5 or MSE; direction candidate adds .1 BCE h1 with fixed temperature1
- **checkpointSelection**: validation normalized return MSE: .8*h1+.2*mean(h2..h12); early stop6; no shrinkage or amplitude floor
- **candidates**: 3 individual seeds plus equal-weight ensemble per objective; flat and train-mean baselines
- **promotionGate**: validation h1 R2vsZero>0; MSE beats train-mean; PearsonIC>.02; balancedAccuracy>.5; priceMAE<=flat*1.005; >=2 of3 time blocks both positive R2vsZero and IC
- **selection**: among eligible choose minimum validation score; if none, best candidate is diagnostic only and must not be promoted
- **evaluationStatus**: same historical holdout reevaluation, not fresh unseen data; no test data used in candidate, checkpoint or promotion selection
- **deployment**: No deployed model file modified. Ensembles require additional JS runtime support; constituent models use existing weight schema.
- **deploymentHashesBefore**: {'b3-15m.json': '432902c5300a568e154abf46991081efcb6790cc05f97008207a0fee87ff27b9', 'b3-1d.json': '45f81d91ae3015f3fd993bf31529ba7b643df5dd090d0c69485b192025d8d30d', 'b3-1h.json': '01ee2a33b42d9434babe15a725a70ca916a2c7150ad18be011ac24e7a13ede5e', 'b3-4h.json': 'cb6b1b3c52067deafd2e76060351150d19ced805b3c1ffded5d219cbaa236acb'}

import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { validateB4Forecast, type B4Forecast, type B4ForwardSummary } from '../src/lib/b4.ts'
import { INTERVAL_MS, type Candle, type Interval } from '../src/types.ts'

const ORIGIN = 1728000000

// Contract fixture reproduced from the real ForwardLedger with fractional
// clocks: issue an eligible forecast at origin+step+5.25, then observe h1 while
// issuing a late next-origin forecast at origin+2*step+65.5. The ledger returns
// 12 targets per recent record (there is no h0 point in recent.path).
function fixture(interval: Interval = '1h'): { model: B4Forecast; origin: Candle } {
  const step = INTERVAL_MS[interval] / 1000
  const origin = { time: ORIGIN + step, open: 102, high: 103, low: 101, close: 102, volume: 10 }
  const metric = (horizon: number) => ({ horizon, samples: 0, maePct: null, baselineMaePct: null,
    skillPct: null, directionAccuracy: null, directionSamples: 0,
    probabilityDirectionAccuracy: null, probabilitySamples: 0, brier: null })
  const record = (offset: number): B4ForwardSummary['recent'][number] => ({ originTime: ORIGIN + offset * step,
    originPrice: offset ? 102 : 100, issuedAt: ORIGIN + (offset + 1) * step + (offset ? 65.5 : 5.25),
    eligibility: offset ? 'late' : 'eligible', latencySeconds: offset ? 65.5 : 5.25,
    path: Array.from({ length: 12 }, (_, i) => ({ time: ORIGIN + (offset + i + 1) * step,
      horizon: i + 1, predicted: 101, ...([1, 3, 6, 12].includes(i + 1) ? { upProbability: .7 } : {}),
      ...(offset === 0 && i === 0 ? { actual: 102, resolvedAt: ORIGIN + 2 * step + 65.5 } : {}) })) })
  const forward: B4ForwardSummary = { modelVersion: 'fixture-v1', scope: { symbol: 'BTCUSDT', interval },
    eligibilityWindowSeconds: 60, recordedOrigins: 2, eligibleOrigins: 1, resolvedOrigins: 0,
    resolvedNextBarOrigins: 1, pendingOrigins: 2, lateOrigins: 1,
    byHorizon: Array.from({ length: 12 }, (_, i) => metric(i + 1)), recent: [record(1), record(0)] }
  forward.byHorizon[0] = { horizon: 1, samples: 1, maePct: .9803921568627416,
    baselineMaePct: 1.9607843137254943, skillPct: 50.00000000000029,
    directionAccuracy: 1, directionSamples: 1, probabilityDirectionAccuracy: 1, probabilitySamples: 1, brier: .09000000000000002 }
  const model: B4Forecast = { modelId: 'guanchao-b4', symbol: 'BTCUSDT', interval,
    originTime: origin.time, originPrice: origin.close, history: [], forward,
    path: Array.from({ length: 13 }, (_, h) => ({ time: origin.time + h * step, mid: 102, lower: 102, upper: 102,
      upScenario: 102, downScenario: 102, expectedMovePct: 0, ...([1, 3, 6, 12].includes(h) ? { upProbability: .7 } : {}) })),
    evaluation: { direction: { balancedAccuracy: .52, baselineBalancedAccuracy: .52, brier: .249, baselineBrier: .25, cvPassed: true },
      magnitude: { mseImprovement: .1, maeImprovement: .04, cvPassed: true, heldoutSkill: true }, samples: 100,
      dataCutoff: new Date((ORIGIN - step) * 1000).toISOString(), dataEnd: new Date((ORIGIN + 30 * step) * 1000).toISOString() } }
  return { model, origin }
}
function check(model: B4Forecast, origin: Candle) {
  return validateB4Forecast(model, model.symbol, model.interval, origin)
}

describe('optional B4 forward ledger response contract', () => {
  for (const interval of ['15m', '1h', '4h', '1d'] as const) test(`${interval} accepts real-ledger clocks, 12 targets and eligible/late records without changing results`, () => {
    const { model, origin } = fixture(interval)
    const before = structuredClone(model)
    assert.equal(check(model, origin), model)
    assert.deepEqual(model, before)
    assert.equal(model.forward!.recent[0].path.length, 12)
  })

  test('legacy service may omit the entire field; empty ledgers keep all errors null', () => {
    const { model, origin } = fixture()
    delete model.forward
    assert.equal(check(model, origin), model)
    const empty = fixture().model.forward!
    for (const key of ['recordedOrigins', 'eligibleOrigins', 'resolvedOrigins', 'resolvedNextBarOrigins', 'pendingOrigins', 'lateOrigins'] as const) empty[key] = 0
    empty.recent = []
    empty.byHorizon[0] = { ...empty.byHorizon[1], horizon: 1 }
    model.forward = empty
    assert.equal(check(model, origin), model)
    assert.ok(model.forward.byHorizon.every((row) => row.maePct === null && row.baselineMaePct === null))
  })

  test('bad summary shapes, sample hierarchies, nullable metrics and impossible probabilities are rejected', () => {
    const mutations: Array<(forward: B4ForwardSummary) => void> = [
      (f) => { f.modelVersion = '' }, (f) => { f.modelVersion = 'bad version /' },
      (f) => { f.scope!.symbol = 'ETHUSDT' }, (f) => { f.scope!.interval = '4h' },
      (f) => { f.eligibilityWindowSeconds = 3600 },
      (f) => { f.recordedOrigins = -1 }, (f) => { f.eligibleOrigins = .5 },
      (f) => { f.pendingOrigins = 0 }, (f) => { f.resolvedNextBarOrigins = 0 },
      (f) => { f.byHorizon.pop() }, (f) => { f.byHorizon[1].horizon = 1 },
      (f) => { f.byHorizon[0].samples = 2 }, (f) => { f.byHorizon[0].directionSamples = 2 },
      (f) => { f.byHorizon[0].probabilitySamples = 2 },
      (f) => { f.byHorizon[0].maePct = NaN }, (f) => { f.byHorizon[0].maePct = null },
      (f) => { f.byHorizon[0].directionAccuracy = 1.01 }, (f) => { f.byHorizon[0].brier = -.1 },
      (f) => { f.byHorizon[0].probabilityDirectionAccuracy = Infinity },
      (f) => { f.byHorizon[0].skillPct = 100 },
      (f) => { f.byHorizon[1].maePct = 0 }, (f) => { f.byHorizon[1].directionAccuracy = 0 },
      (f) => { f.byHorizon[1].brier = 0 }, (f) => { f.recent = [] },
    ]
    for (const mutate of mutations) {
      const { model, origin } = fixture()
      mutate(model.forward!)
      assert.throws(() => check(model, origin), /实时首发存档校验失败/)
    }
    for (const bad of [null, {}, [], 'forward']) {
      const { model, origin } = fixture()
      model.forward = bad as unknown as B4ForwardSummary
      assert.throws(() => check(model, origin), /实时首发存档校验失败/)
    }
  })

  test('recent records cannot misstate origin, issuance, eligibility, target spacing or settlement time', () => {
    const mutations: Array<(f: B4ForwardSummary) => void> = [
      (f) => { f.recent[0].originTime += 3600 },
      (f) => { f.recent[0].originTime += 1 },
      (f) => { f.recent[0].issuedAt = ORIGIN },
      (f) => { f.recent[0].issuedAt = 1e100 },
      (f) => { f.recent[0].eligibility = 'eligible' },
      (f) => { f.recent[0].latencySeconds = 5 },
      (f) => { f.recent[0].originPrice = 0 },
      (f) => { f.recent.reverse() },
      (f) => { f.recent[0].path.push({ ...f.recent[0].path[0] }) },
      (f) => { f.recent[0].path[0].horizon = 0 },
      (f) => { f.recent[0].path[0].time += 3600 },
      (f) => { f.recent[0].path[0].predicted = NaN },
      (f) => { f.recent[0].path[0].upProbability = 2 },
      (f) => { f.recent[0].path[0].actual = 103; f.recent[0].path[0].resolvedAt = ORIGIN + 100000 },
      (f) => { f.recent[0].path[0].resolvedAt = ORIGIN + 100000 },
      (f) => { delete f.recent[1].path[0].resolvedAt },
      (f) => { f.recent[1].path[0].resolvedAt = ORIGIN + 3600 },
      (f) => { f.recent[1].path[0].actual = -1 },
    ]
    for (const mutate of mutations) {
      const { model, origin } = fixture()
      mutate(model.forward!)
      assert.throws(() => check(model, origin), /实时首发存档校验失败/)
    }
  })

  test('legitimate very late records and immutable observations are not rewritten using current candles', () => {
    const { model, origin } = fixture()
    model.forward!.recent[0].issuedAt = origin.time + 40 * 3600 + .25
    model.forward!.recent[0].latencySeconds = 39 * 3600 + .25
    // A later exchange correction may change current prices. The first issued
    // forecast and the first observed actual in the ledger must remain intact.
    origin.close = 150
    model.originPrice = 150
    model.path = model.path.map((point) => ({ ...point, mid: 150, lower: 150, upper: 150, upScenario: 150, downScenario: 150 }))
    const before = structuredClone(model.forward)
    assert.equal(check(model, origin), model)
    assert.deepEqual(model.forward, before)
    assert.equal(model.forward!.recent[1].path[0].actual, 102)
  })

  test('zero baseline uses null relative skill, and old optional extra metadata can be absent', () => {
    const { model, origin } = fixture()
    const forward = model.forward!
    delete forward.scope
    for (const record of forward.recent) { delete record.originPrice; delete record.latencySeconds }
    forward.byHorizon[0].baselineMaePct = 0
    forward.byHorizon[0].skillPct = null
    assert.equal(check(model, origin), model)
    forward.byHorizon[0].skillPct = 0
    assert.throws(() => check(model, origin), /实时首发存档校验失败/)
  })
})

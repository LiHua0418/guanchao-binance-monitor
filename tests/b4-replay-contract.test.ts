import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { validateB4Forecast, type B4Forecast, type B4ReplayPath } from '../src/lib/b4.ts'
import { INTERVAL_MS, type Candle, type Interval } from '../src/types.ts'

const START = Date.UTC(2026, 8, 1) / 1000
const closeAt = (index: number) => 100 + index * .3 + (index % 3) * .02
const predictAt = (index: number, horizon: number) => closeAt(index) + .04 * horizon + .005 * index

function fixture(interval: Interval = '1h'): { origin: Candle; model: B4Forecast } {
  const step = INTERVAL_MS[interval] / 1000
  const origin: Candle = { time: START + 40 * step, open: closeAt(40), high: closeAt(40) + 1,
    low: closeAt(40) - 1, close: closeAt(40), volume: 100, isClosed: true }
  const replayPaths: B4ReplayPath[] = Array.from({ length: 29 }, (_, index) => ({
    originTime: START + index * step, originPrice: closeAt(index),
    path: Array.from({ length: 13 }, (_, horizon) => ({
      time: START + (index + horizon) * step, horizon,
      predicted: horizon === 0 ? closeAt(index) : predictAt(index, horizon),
      actual: closeAt(index + horizon),
    })),
  }))
  const model: B4Forecast = { modelId: 'guanchao-b4', symbol: 'BTCUSDT', interval,
    originTime: origin.time, originPrice: origin.close,
    path: Array.from({ length: 13 }, (_, h) => {
      const expectedMovePct = h * .2
      return { time: origin.time + h * step, mid: origin.close + .01 * h,
        lower: origin.close - .1 * h, upper: origin.close + .1 * h,
        upScenario: origin.close * (1 + expectedMovePct / 100), downScenario: origin.close / (1 + expectedMovePct / 100),
        expectedMovePct, ...([1, 3, 6, 12].includes(h) ? { upProbability: .53 } : {}),
      }
    }),
    history: Array.from({ length: 40 }, (_, index) => ({
      originTime: START + index * step, time: START + (index + 1) * step,
      baseline: closeAt(index), predicted: predictAt(index, 1), actual: closeAt(index + 1),
      upProbability: .53, expectedMovePct: .2,
      actualMovePct: Math.expm1(Math.abs(Math.log(closeAt(index + 1) / closeAt(index)))) * 100,
    })), replayPaths,
    evaluation: { direction: { balancedAccuracy: .52, baselineBalancedAccuracy: .52, brier: .249, baselineBrier: .25, cvPassed: true },
      magnitude: { mseImprovement: .09, maeImprovement: .03, cvPassed: true, heldoutSkill: true },
      samples: 1000, dataCutoff: new Date(START * 1000).toISOString(), dataEnd: new Date(origin.time * 1000).toISOString() },
  }
  return { origin, model }
}

const validate = ({ model, origin }: ReturnType<typeof fixture>) => validateB4Forecast(model, 'BTCUSDT', model.interval, origin)

describe('B4 complete fixed-origin replay response contract', () => {
  for (const interval of ['15m', '1h', '4h', '1d'] as const) test(`${interval} accepts each actual interval step and anchors every replay to one original close`, () => {
    const value = fixture(interval)
    assert.equal(validate(value), value.model)
    const newest = value.model.replayPaths!.at(-1)!
    assert.equal(newest.path.length, 13)
    assert.equal(newest.path[12].time, value.origin.time)
    assert.equal(newest.path[12].actual, value.origin.close)
    for (const replay of value.model.replayPaths!) {
      const oneStep = value.model.history.find((point) => point.originTime === replay.originTime)!
      assert.equal(replay.originPrice, oneStep.baseline)
      assert.equal(replay.path[1].predicted, oneStep.predicted)
      assert.equal(replay.path[1].actual, oneStep.actual)
      assert.deepEqual(replay.path.map((p) => p.horizon), Array.from({ length: 13 }, (_, h) => h))
    }
  })

  test('replay remains optional and an explicitly empty replay is allowed before twelve future targets close', () => {
    const value = fixture()
    delete value.model.replayPaths
    assert.equal(validate(value), value.model)
    value.model.replayPaths = []
    assert.equal(validate(value), value.model)
  })

  test('incomplete, extra or misnumbered horizons are rejected as whole replay records', () => {
    const mutations: Array<(replay: B4ReplayPath) => void> = [
      (r) => { r.path.pop() },
      (r) => { r.path.splice(6, 1) },
      (r) => { r.path.push({ ...r.path[12], horizon: 13, time: r.path[12].time + 3600 }) },
      (r) => { r.path[6].horizon = 7 },
      (r) => { r.path[1].horizon = .5 },
      (r) => { r.path.reverse() },
    ]
    for (const mutate of mutations) {
      const value = fixture()
      mutate(value.model.replayPaths![0])
      assert.throws(() => validate(value))
    }
  })

  test('off-grid, nonfinite, duplicate and noncontiguous replay times cannot masquerade as per-bar forecasts', () => {
    const mutations: Array<(model: B4Forecast) => void> = [
      (m) => { m.replayPaths![0].path[5].time += 1 },
      (m) => { m.replayPaths![0].path[5].time += 11 * 3600 },
      (m) => { m.replayPaths![0].path[5].time = NaN },
      (m) => { m.replayPaths![0].originTime = Infinity },
      (m) => { m.replayPaths![0].originTime += .5 },
      (m) => { m.replayPaths![1] = structuredClone(m.replayPaths![0]) },
      (m) => { m.replayPaths!.splice(1, 1) },
      (m) => { m.replayPaths!.reverse() },
    ]
    for (const mutate of mutations) {
      const value = fixture()
      mutate(value.model)
      assert.throws(() => validate(value))
    }
  })

  test('pre-calibration origins and paths whose twelfth actual candle is still in the future are rejected', () => {
    const value = fixture()
    value.model.history = []
    value.model.replayPaths = [value.model.replayPaths![0]]
    value.model.replayPaths[0].originTime -= 3600
    for (const point of value.model.replayPaths[0].path) point.time -= 3600
    assert.throws(() => validate(value))
    const future = fixture()
    future.model.history = []
    future.model.replayPaths = [future.model.replayPaths!.at(-1)!]
    future.model.replayPaths[0].originTime += 3600
    for (const point of future.model.replayPaths[0].path) point.time += 3600
    assert.throws(() => validate(future))
  })

  test('a whole internally consistent replay cannot move one second off the real candle grid', () => {
    const value = fixture()
    value.model.history = []
    value.model.replayPaths = [value.model.replayPaths![0]]
    value.model.replayPaths[0].originTime += 1
    for (const point of value.model.replayPaths[0].path) point.time += 1
    assert.throws(() => validate(value), 'all replay origins must share the actual request candle time grid')
  })

  test('origin prediction and actual must both equal the retained origin close', () => {
    for (const field of ['predicted', 'actual'] as const) {
      const value = fixture()
      value.model.replayPaths![0].path[0][field] += 1
      assert.throws(() => validate(value))
    }
    const wrongOrigin = fixture()
    const replay = wrongOrigin.model.replayPaths![0]
    replay.originPrice += 50
    replay.path[0].predicted = replay.originPrice
    replay.path[0].actual = replay.originPrice
    assert.throws(() => validate(wrongOrigin), 'a self-consistent fake anchor must still disagree with the historical origin baseline')
  })

  test('the complete replay h1 must match the existing rolling h1 estimate and actual at that same origin', () => {
    for (const field of ['predicted', 'actual'] as const) {
      const value = fixture()
      value.model.replayPaths![10].path[1][field] += 1
      assert.throws(() => validate(value))
    }
    const tolerance = fixture()
    tolerance.model.replayPaths![10].path[1].predicted += 1e-10
    assert.equal(validate(tolerance), tolerance.model, 'normal floating-point serialization differences remain acceptable')
  })

  test('nonpositive or nonfinite prices and a last actual inconsistent with the current candle are rejected', () => {
    for (const bad of [0, -1, Infinity, NaN]) {
      for (const field of ['predicted', 'actual'] as const) {
        const value = fixture()
        value.model.replayPaths![3].path[6][field] = bad
        assert.throws(() => validate(value))
      }
    }
    const wrongCurrentClose = fixture()
    wrongCurrentClose.model.replayPaths!.at(-1)!.path[12].actual += 1
    assert.throws(() => validate(wrongCurrentClose))
  })

  test('oversized or non-array replay containers are rejected', () => {
    const value = fixture()
    value.model.replayPaths = Array.from({ length: 241 }, () => structuredClone(value.model.replayPaths![0]))
    assert.throws(() => validate(value))
    value.model.replayPaths = {} as unknown as B4ReplayPath[]
    assert.throws(() => validate(value))
  })
})

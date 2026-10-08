import type { ForecastPoint } from './analysis'
import type { ReplayPoint } from './b3'
import { INTERVAL_MS, type Candle, type Interval } from '../types'

export type B4Point = ForecastPoint & {
  upScenario: number
  downScenario: number
  expectedMovePct: number
  upProbability?: number
}
export type B4HistoryPoint = ReplayPoint & {
  baseline: number
  upProbability: number
  expectedMovePct: number
  actualMovePct: number
}
export type B4ReplayPath = {
  originTime: number
  originPrice: number
  path: { time: number; horizon: number; predicted: number; actual: number }[]
}
export type B4ForwardSummary = {
  modelVersion: string
  scope?: { symbol: string; interval: Interval }
  eligibilityWindowSeconds: number
  recordedOrigins: number
  eligibleOrigins: number
  resolvedOrigins: number
  resolvedNextBarOrigins: number
  pendingOrigins: number
  lateOrigins: number
  byHorizon: {
    horizon: number
    samples: number
    maePct: number | null
    baselineMaePct: number | null
    skillPct: number | null
    directionAccuracy: number | null
    directionSamples: number
    probabilityDirectionAccuracy: number | null
    probabilitySamples: number
    brier: number | null
  }[]
  recent: {
    originTime: number
    originPrice?: number
    issuedAt: number
    eligibility: 'eligible' | 'late'
    latencySeconds?: number
    path: { time: number; horizon: number; predicted: number; actual?: number; resolvedAt?: number; upProbability?: number }[]
  }[]
}
export type B4Forecast = {
  modelId: 'guanchao-b4'
  symbol: string
  interval: Interval
  originTime: number
  originPrice: number
  path: B4Point[]
  history: B4HistoryPoint[]
  replayPaths?: B4ReplayPath[]
  forward?: B4ForwardSummary
  evaluation: {
    direction: { balancedAccuracy: number; baselineBalancedAccuracy: number; brier: number; baselineBrier: number; cvPassed: boolean }
    magnitude: { mseImprovement: number; maeImprovement: number; cvPassed: boolean; heldoutSkill: boolean }
    samples: number
    dataCutoff: string
    dataEnd: string
  }
}
export type B4Status = 'loading' | 'ready' | 'insufficient' | 'unsupported' | 'error'
export const B4_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

// Include all observed closed-bar inputs: a correction to an older bar must
// invalidate inference, while live trades in the open bar must not repaint it.
export function b4RequestKey(symbol: string, interval: Interval, candles: Candle[]): string {
  return JSON.stringify([symbol, interval, candles.map((c) => [c.time, c.open, c.high, c.low, c.close,
    c.volume, c.quoteVolume, c.trades, c.takerBuyVolume, c.takerBuyQuoteVolume])])
}

export function validateB4Forecast(value: unknown, symbol: string, interval: Interval, origin: Candle): B4Forecast {
  const model = value as B4Forecast
  const positive = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0
  const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
  const probability = (v: unknown) => finite(v) && v >= 0 && v <= 1
  const timestamp = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v > 0
  const near = (a: number, b: number) => finite(a) && finite(b) && Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(b))
  const step = INTERVAL_MS[interval] / 1000
  if (!model || !B4_SYMBOLS.includes(symbol) || !timestamp(origin.time) || !positive(origin.close)
    || !Number.isFinite(step) || model.modelId !== 'guanchao-b4' || model.symbol !== symbol || model.interval !== interval
    || model.originTime !== origin.time || model.originPrice !== origin.close
    || !Array.isArray(model.path) || model.path.length !== 13 || !Array.isArray(model.history)) {
    throw new Error('B4 返回的数据与当前交易对、周期或收盘起点不一致')
  }
  for (const [i, point] of model.path.entries()) {
    const directional = [1, 3, 6, 12].includes(i)
    if (!point || !timestamp(point.time) || point.time !== origin.time + i * step || ![point.mid, point.lower, point.upper, point.upScenario, point.downScenario].every(positive)
      || point.lower > point.upper || point.downScenario > point.upScenario
      || !finite(point.expectedMovePct) || point.expectedMovePct < 0
      || (directional ? !probability(point.upProbability) : point.upProbability !== undefined)
      || !near(point.upScenario, origin.close * (1 + point.expectedMovePct / 100))
      || !near(point.downScenario, origin.close / (1 + point.expectedMovePct / 100))
      || (i === 0 && (point.expectedMovePct !== 0 || ![point.mid, point.lower, point.upper, point.upScenario, point.downScenario].every((v) => v === origin.close)))) throw new Error('B4 预测路径校验失败')
  }
  const evaluation = model.evaluation
  if (!evaluation || !evaluation.direction || !evaluation.magnitude
    || ![evaluation.direction.balancedAccuracy, evaluation.direction.baselineBalancedAccuracy, evaluation.direction.brier, evaluation.direction.baselineBrier].every(probability)
    || ![evaluation.direction.cvPassed, evaluation.magnitude.cvPassed, evaluation.magnitude.heldoutSkill].every((v) => typeof v === 'boolean')
    || ![evaluation.magnitude.mseImprovement, evaluation.magnitude.maeImprovement].every((v) => finite(v) && v <= 1)
    || !Number.isSafeInteger(evaluation.samples) || evaluation.samples < 1
    || typeof evaluation.dataCutoff !== 'string' || typeof evaluation.dataEnd !== 'string'
    || !Number.isFinite(Date.parse(evaluation.dataCutoff)) || !Number.isFinite(Date.parse(evaluation.dataEnd))
    || Date.parse(evaluation.dataCutoff) > Date.parse(evaluation.dataEnd)
    || Date.parse(evaluation.dataCutoff) > origin.time * 1000) throw new Error('B4 评估信息校验失败')
  for (const [i, point] of model.history.entries()) {
    if (!point || !timestamp(point.time) || !timestamp(point.originTime) || point.time !== point.originTime + step || point.time > origin.time
      || point.originTime * 1000 < Date.parse(evaluation.dataCutoff)
      || (i > 0 && point.time !== model.history[i - 1].time + step)
      || (i > 0 && point.baseline !== model.history[i - 1].actual)
      || ![point.predicted, point.actual, point.baseline].every(positive)
      || !probability(point.upProbability) || !finite(point.expectedMovePct) || point.expectedMovePct < 0
      || !finite(point.actualMovePct) || point.actualMovePct < 0
      || !near(point.actualMovePct, Math.expm1(Math.abs(Math.log(point.actual / point.baseline))) * 100)
      || (point.time === origin.time && point.actual !== origin.close)) throw new Error('B4 逐根回放校验失败')
  }
  if (model.replayPaths !== undefined) {
    if (!Array.isArray(model.replayPaths) || model.replayPaths.length > 240) throw new Error('B4 固定起点回放格式错误')
    for (const [i, replay] of model.replayPaths.entries()) {
      if (!replay || !timestamp(replay.originTime) || !positive(replay.originPrice)
        || (model.originTime - replay.originTime) % step !== 0
        || replay.originTime * 1000 < Date.parse(evaluation.dataCutoff)
        || replay.originTime + 12 * step > model.originTime
        || (i > 0 && replay.originTime !== model.replayPaths[i - 1].originTime + step)
        || !Array.isArray(replay.path) || replay.path.length !== 13) throw new Error('B4 固定起点回放范围错误')
      for (const [h, point] of replay.path.entries()) {
        if (!point || point.horizon !== h || point.time !== replay.originTime + h * step
          || !positive(point.predicted) || !positive(point.actual)
          || (h === 0 && (point.predicted !== replay.originPrice || point.actual !== replay.originPrice))
          || (point.time === model.originTime && point.actual !== model.originPrice)) throw new Error('B4 固定起点回放路径错误')
      }
      const rolling = model.history.find((point) => point.originTime === replay.originTime)
      if (rolling && (replay.originPrice !== rolling.baseline || !near(replay.path[1].predicted, rolling.predicted) || replay.path[1].actual !== rolling.actual)) throw new Error('B4 回放与原始一步预测不一致')
    }
  }
  if (model.forward !== undefined) {
    const forward = model.forward
    const count = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0
    const nonnegative = (v: unknown): v is number => finite(v) && v >= 0
    // Issuance/settlement use the real clock and can contain fractional seconds.
    // Keep them representable by Date because the UI formats these timestamps.
    const clockTime = (v: unknown): v is number => positive(v) && Number.isFinite(new Date(v * 1000).getTime())
    const fail = () => { throw new Error('B4 实时首发存档校验失败') }
    if (!forward || typeof forward !== 'object'
      || typeof forward.modelVersion !== 'string' || !/^[A-Za-z0-9_.:-]{1,160}$/.test(forward.modelVersion)
      || (forward.scope !== undefined && (!forward.scope || forward.scope.symbol !== symbol || forward.scope.interval !== interval))
      || !positive(forward.eligibilityWindowSeconds) || !near(forward.eligibilityWindowSeconds, Math.min(step * .1, 60))
      || ![forward.recordedOrigins, forward.eligibleOrigins, forward.resolvedOrigins, forward.resolvedNextBarOrigins,
        forward.pendingOrigins, forward.lateOrigins].every(count)
      || forward.eligibleOrigins + forward.lateOrigins !== forward.recordedOrigins
      || forward.resolvedOrigins + forward.pendingOrigins !== forward.recordedOrigins
      || forward.resolvedNextBarOrigins > forward.eligibleOrigins
      || !Array.isArray(forward.byHorizon) || forward.byHorizon.length !== 12
      || !Array.isArray(forward.recent) || forward.recent.length !== Math.min(12, forward.recordedOrigins)) fail()
    for (const [i, row] of forward.byHorizon.entries()) {
      if (!row || row.horizon !== i + 1 || !count(row.samples) || row.samples > forward.eligibleOrigins
        || !count(row.directionSamples) || row.directionSamples > row.samples
        || !count(row.probabilitySamples) || row.probabilitySamples > row.directionSamples
        || (row.samples === 0 ? row.maePct !== null || row.baselineMaePct !== null || row.skillPct !== null
          : !nonnegative(row.maePct) || !nonnegative(row.baselineMaePct))
        || (row.directionSamples === 0 ? row.directionAccuracy !== null : !probability(row.directionAccuracy))
        || (row.probabilitySamples === 0 ? row.probabilityDirectionAccuracy !== null || row.brier !== null
          : !probability(row.probabilityDirectionAccuracy) || !probability(row.brier))) fail()
      if (row.samples > 0) {
        if (row.baselineMaePct === 0 ? row.skillPct !== null
          : !finite(row.skillPct) || !near(row.skillPct, 100 * (1 - row.maePct! / row.baselineMaePct!))) fail()
      }
    }
    if (forward.byHorizon[0].samples !== forward.resolvedNextBarOrigins) fail()
    const seen = new Set<number>()
    let recentEligible = 0
    let recentResolved = 0
    for (const [i, record] of forward.recent.entries()) {
      if (!record || !timestamp(record.originTime) || record.originTime % step !== 0
        || record.originTime > model.originTime || record.originTime * 1000 < Date.parse(evaluation.dataCutoff)
        || seen.has(record.originTime) || (i > 0 && record.originTime >= forward.recent[i - 1].originTime)
        || (record.originPrice !== undefined && !positive(record.originPrice))
        || !clockTime(record.issuedAt) || record.issuedAt < record.originTime + step
        || (record.eligibility !== 'eligible' && record.eligibility !== 'late')
        || !Array.isArray(record.path) || record.path.length !== 12) fail()
      seen.add(record.originTime)
      const latency = record.issuedAt - (record.originTime + step)
      if (record.eligibility !== (latency <= forward.eligibilityWindowSeconds ? 'eligible' : 'late')
        || (record.latencySeconds !== undefined && (!nonnegative(record.latencySeconds) || !near(record.latencySeconds, latency)))) fail()
      if (record.eligibility === 'eligible') recentEligible++
      let resolved = 0
      for (const [j, point] of record.path.entries()) {
        if (!point || point.horizon !== j + 1 || !timestamp(point.time)
          || point.time !== record.originTime + (j + 1) * step || !positive(point.predicted)
          || (point.upProbability !== undefined && !probability(point.upProbability))) fail()
        if (point.actual === undefined) {
          if (point.resolvedAt !== undefined) fail()
        } else {
          if (!positive(point.actual) || point.time > model.originTime || !clockTime(point.resolvedAt)
            || point.resolvedAt < point.time + step || point.resolvedAt < record.issuedAt) fail()
          resolved++
        }
      }
      if (resolved === 12) recentResolved++
    }
    if (recentEligible > forward.eligibleOrigins || forward.recent.length - recentEligible > forward.lateOrigins
      || recentResolved > forward.resolvedOrigins || forward.recent.length - recentResolved > forward.pendingOrigins
      || (forward.recordedOrigins <= 12 && (recentEligible !== forward.eligibleOrigins || recentResolved !== forward.resolvedOrigins))) fail()
  }
  return model
}

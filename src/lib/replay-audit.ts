import type { B4Forecast, B4ReplayPath } from './b4'
import { INTERVAL_MS } from '../types'

export type ReplayHorizonMetric = {
  horizon: number
  samples: number
  maePct: number | null
  baselineMaePct: number | null
  differencePct: number | null
  skillPct: number | null
}

export type ReplayAuditMetrics = {
  samples: number
  rejectedRecords: number
  byHorizon: ReplayHorizonMetric[]
  meanMaePct: number | null
  meanBaselineMaePct: number | null
}

const positive = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0

// All horizons deliberately share one set of complete, unique origins. An
// incomplete target may not silently reduce one horizon's sample set and make
// it look better than another. Every baseline is the same origin's close.
export function replayAuditMetrics(model: Pick<B4Forecast, 'replayPaths' | 'interval' | 'originTime' | 'evaluation'>): ReplayAuditMetrics {
  const records = Array.isArray(model.replayPaths) ? model.replayPaths : []
  const step = INTERVAL_MS[model.interval] / 1000
  const cutoff = Date.parse(model.evaluation.dataCutoff) / 1000
  const originCounts = new Map<number, number>()
  for (const row of records) if (row && Number.isSafeInteger(row.originTime)) {
    originCounts.set(row.originTime, (originCounts.get(row.originTime) ?? 0) + 1)
  }
  const complete = records.filter((row): row is B4ReplayPath => !!row
    && Number.isFinite(step) && step > 0 && Number.isFinite(cutoff)
    && Number.isSafeInteger(row.originTime) && row.originTime >= cutoff
    && originCounts.get(row.originTime) === 1 && positive(row.originPrice)
    && Array.isArray(row.path) && row.path.length === 13
    && row.path.every((point, horizon) => !!point && point.horizon === horizon
      && point.time === row.originTime + horizon * step && point.time <= model.originTime
      && positive(point.predicted) && positive(point.actual)
      && (horizon !== 0 || (point.predicted === row.originPrice && point.actual === row.originPrice))))
  const byHorizon = Array.from({ length: 12 }, (_, index): ReplayHorizonMetric => {
    const horizon = index + 1
    if (!complete.length) return { horizon, samples: 0, maePct: null, baselineMaePct: null, differencePct: null, skillPct: null }
    let error = 0
    let baselineError = 0
    for (const row of complete) {
      const point = row.path[horizon]
      error += Math.abs(point.predicted / point.actual - 1) * 100
      baselineError += Math.abs(row.originPrice / point.actual - 1) * 100
    }
    const maePct = error / complete.length
    const baselineMaePct = baselineError / complete.length
    return { horizon, samples: complete.length, maePct, baselineMaePct,
      differencePct: maePct - baselineMaePct,
      skillPct: baselineMaePct > 0 ? (1 - maePct / baselineMaePct) * 100 : null }
  })
  return { samples: complete.length, rejectedRecords: records.length - complete.length, byHorizon,
    meanMaePct: complete.length ? byHorizon.reduce((sum, row) => sum + row.maePct!, 0) / 12 : null,
    meanBaselineMaePct: complete.length ? byHorizon.reduce((sum, row) => sum + row.baselineMaePct!, 0) / 12 : null }
}

import type { Candle } from '../types'
import type { ForecastPoint } from './analysis'

export const B3_FEATURES = ['logReturn1', 'logReturn3', 'logReturn12', 'rangeRatio', 'bodyRatio', 'logVolumeChange', 'sma20Distance', 'sma50Distance']
export interface B3Bundle {
  schemaVersion: 1
  name: string
  architecture: string
  interval: string
  sequence: number
  horizon: number
  features: string[]
  mean: number[]
  scale: number[]
  targetScale: number[]
  blend: number
  blendByHorizon?: number[]
  trainingObjective?: string
  weights: {
    convWeight: number[][][]; convBias: number[]
    lstmWeightIH: number[][]; lstmWeightHH: number[][]
    lstmBiasIH: number[]; lstmBiasHH: number[]
    attentionWeight: number[]; attentionBias: number
    outputWeight: number[][]; outputBias: number[]
  }
  residualLower: number[]
  residualUpper: number[]
  trainedAt: string
  dataCutoff: string
  dataEnd: string
  trainedSymbols: string[]
  trainedBars: number
  epochs: number
  metrics: {
    testSamples: number; modelMaePct: number; baselineMaePct: number
    directionHitRate: number; coverage80: number
    byHorizon: { horizon: number; maePct: number; baselineMaePct: number; hitRate: number; coverage80: number }[]
  }
}
export interface ReplayPoint { time: number; predicted: number; actual: number; originTime: number; baseline?: number }
export interface B3Forecast {
  path: ForecastPoint[]
  endPct: number
  bundle: B3Bundle
}
function tensor(value: unknown, dimensions: number[]): boolean {
  if (!dimensions.length) return typeof value === 'number' && Number.isFinite(value)
  return Array.isArray(value) && value.length === dimensions[0] && value.every((v) => tensor(v, dimensions.slice(1)))
}
export function validateB3Bundle(value: unknown, expectedInterval?: string): B3Bundle {
  if (!value || typeof value !== 'object') throw new Error('模型文件格式不正确')
  const b = value as B3Bundle
  const w = b.weights
  if (b.schemaVersion !== 1 || b.architecture !== 'CNN-LSTM-Attention' || b.name !== 'guanchao-b3'
    || b.sequence !== 48 || b.horizon !== 12 || !['15m', '1h', '4h', '1d'].includes(b.interval)
    || (expectedInterval != null && b.interval !== expectedInterval)
    || JSON.stringify(b.features) !== JSON.stringify(B3_FEATURES)
    || !tensor(b.mean, [8]) || !tensor(b.scale, [8]) || b.scale.some((v) => v <= 0)
    || !tensor(b.targetScale, [12]) || b.targetScale.some((v) => v <= 0)
    || !Number.isFinite(b.blend) || b.blend < 0 || b.blend > 1
    || (b.trainingObjective === 'next-bar-priority' && b.blendByHorizon === undefined)
    || (b.blendByHorizon !== undefined && (!tensor(b.blendByHorizon, [12])
      || b.blendByHorizon.some((v) => v < 0 || v > 1) || b.blendByHorizon[0] !== b.blend))
    || !tensor(b.residualLower, [12]) || !tensor(b.residualUpper, [12])
    || b.residualLower.some((v, i) => v > b.residualUpper[i])
    || !Number.isFinite(Date.parse(b.dataCutoff)) || !Number.isFinite(Date.parse(b.trainedAt))
    || !Number.isFinite(Date.parse(b.dataEnd)) || Date.parse(b.dataEnd) < Date.parse(b.dataCutoff)
    || !Array.isArray(b.trainedSymbols) || !b.trainedSymbols.length || b.trainedSymbols.some((s) => typeof s !== 'string')
    || !Number.isFinite(b.trainedBars) || b.trainedBars <= 0 || !Number.isFinite(b.epochs)
    || !w || !tensor(w.convWeight, [16, 8, 3]) || !tensor(w.convBias, [16])
    || !tensor(w.lstmWeightIH, [96, 16]) || !tensor(w.lstmWeightHH, [96, 24])
    || !tensor(w.lstmBiasIH, [96]) || !tensor(w.lstmBiasHH, [96])
    || !tensor(w.attentionWeight, [24]) || !tensor(w.attentionBias, [])
    || !tensor(w.outputWeight, [12, 48]) || !tensor(w.outputBias, [12])) throw new Error('模型参数不完整或版本不兼容')
  const m = b.metrics
  if (!m || !Number.isInteger(m.testSamples) || m.testSamples <= 0
    || ![m.modelMaePct, m.baselineMaePct].every((v) => Number.isFinite(v) && v >= 0)
    || ![m.directionHitRate, m.coverage80].every((v) => Number.isFinite(v) && v >= 0 && v <= 1)
    || !Array.isArray(m.byHorizon) || m.byHorizon.length !== 12
    || m.byHorizon.some((h, i) => h.horizon !== i + 1
      || ![h.maePct, h.baselineMaePct].every((v) => Number.isFinite(v) && v >= 0)
      || ![h.hitRate, h.coverage80].every((v) => Number.isFinite(v) && v >= 0 && v <= 1))) throw new Error('模型评估记录不完整')
  return b
}
function validCandles(candles: Candle[], intervalMs?: number): boolean {
  return candles.every((c, i) => Number.isFinite(c.time) && c.time >= 0
    && [c.open, c.high, c.low, c.close].every((v) => Number.isFinite(v) && v > 0)
    && Number.isFinite(c.volume) && c.volume >= 0 && c.high >= Math.max(c.open, c.close)
    && c.low <= Math.min(c.open, c.close)
    && (!i || (intervalMs ? c.time - candles[i - 1].time === intervalMs / 1000 : c.time > candles[i - 1].time)))
}
export function closedCandles(candles: Candle[], intervalMs: number, nowMs: number): Candle[] {
  if (!Number.isFinite(intervalMs) || intervalMs <= 0 || !Number.isFinite(nowMs)) return []
  return candles.filter((c) => c.isClosed !== false && c.time * 1000 + intervalMs <= nowMs)
}
export function buildB3Features(candles: Candle[], sequence = 48): number[][] | null {
  if (!Number.isInteger(sequence) || sequence <= 0 || candles.length < sequence + 49 || !validCandles(candles)) return null
  const sums = [0]
  for (const c of candles) sums.push(sums[sums.length - 1] + c.close)
  return candles.slice(-sequence).map((c, offset) => {
    const i = candles.length - sequence + offset
    return [Math.log(c.close / candles[i - 1].close), Math.log(c.close / candles[i - 3].close),
      Math.log(c.close / candles[i - 12].close), (c.high - c.low) / c.close,
      (c.close - c.open) / c.open, Math.log1p(c.volume) - Math.log1p(candles[i - 1].volume),
      c.close / ((sums[i + 1] - sums[i - 19]) / 20) - 1,
      c.close / ((sums[i + 1] - sums[i - 49]) / 50) - 1]
  })
}
export function standardizeB3(features: number[][], bundle: B3Bundle): number[][] {
  return features.map((row) => row.map((v, i) => Math.max(-8, Math.min(8, (v - bundle.mean[i]) / bundle.scale[i]))))
}
function dot(a: number[], b: number[]): number {
  let value = 0
  for (let i = 0; i < a.length; i++) value += a[i] * b[i]
  return value
}
export function forwardB3(input: number[][], bundle: B3Bundle): number[] {
  const w = bundle.weights
  const states: number[][] = []
  let hidden = Array<number>(24).fill(0)
  let cell = Array<number>(24).fill(0)
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
  for (let t = 0; t < input.length; t++) {
    const conv = w.convWeight.map((kernel, channel) => {
      let value = w.convBias[channel]
      for (let f = 0; f < 8; f++) {
        for (let k = 0; k < 3; k++) if (t + k >= 2) value += kernel[f][k] * input[t + k - 2][f]
      }
      return Math.max(0, value)
    })
    const gates = w.lstmWeightIH.map((row, i) => dot(row, conv) + dot(w.lstmWeightHH[i], hidden) + w.lstmBiasIH[i] + w.lstmBiasHH[i])
    cell = cell.map((c, i) => sigmoid(gates[i + 24]) * c + sigmoid(gates[i]) * Math.tanh(gates[i + 48]))
    hidden = cell.map((c, i) => sigmoid(gates[i + 72]) * Math.tanh(c))
    states.push(hidden)
  }
  const scores = states.map((state) => dot(w.attentionWeight, state) + w.attentionBias)
  const maximum = Math.max(...scores)
  const exps = scores.map((s) => Math.exp(s - maximum))
  const denominator = exps.reduce((a, b) => a + b, 0)
  const context = Array<number>(24).fill(0)
  states.forEach((state, t) => state.forEach((v, i) => { context[i] += v * exps[t] / denominator }))
  const joined = [...hidden, ...context]
  return w.outputWeight.map((row, i) => dot(row, joined) + w.outputBias[i])
}
export function predictB3(candles: Candle[], bundle: B3Bundle): { logReturns: number[]; networkOutput: number[] } | null {
  const features = buildB3Features(candles, bundle.sequence)
  if (!features) return null
  const networkOutput = forwardB3(standardizeB3(features, bundle), bundle)
  const logReturns = networkOutput.map((v, i) => v * bundle.targetScale[i] * (bundle.blendByHorizon?.[i] ?? bundle.blend))
  if (![...networkOutput, ...logReturns].every(Number.isFinite)) return null
  return { logReturns, networkOutput }
}
export function forecastB3(candles: Candle[], intervalMs: number, bundle: B3Bundle): B3Forecast | null {
  if (!Number.isFinite(intervalMs) || intervalMs <= 0 || !validCandles(candles, intervalMs)) return null
  const predicted = predictB3(candles, bundle)
  const last = candles.at(-1)
  if (!predicted || !last) return null
  const path = [{ time: last.time, mid: last.close, lower: last.close, upper: last.close }]
  predicted.logReturns.forEach((value, i) => path.push({
    time: last.time + (i + 1) * intervalMs / 1000, mid: last.close * Math.exp(value),
    lower: last.close * Math.exp(value + bundle.residualLower[i]), upper: last.close * Math.exp(value + bundle.residualUpper[i]),
  }))
  if (path.some((p) => ![p.mid, p.lower, p.upper].every((v) => Number.isFinite(v) && v > 0))) return null
  const endPct = (path.at(-1)!.mid / last.close - 1) * 100
  return { path, endPct, bundle }
}
export function replayB3(candles: Candle[], intervalMs: number, bundle: B3Bundle, limit = 240): ReplayPoint[] {
  if (!Number.isFinite(intervalMs) || intervalMs <= 0 || !validCandles(candles, intervalMs)) return []
  const cutoff = Date.parse(bundle.dataCutoff) / 1000
  const result: ReplayPoint[] = []
  // Each target is predicted at the preceding close, without seeing that target.
  // One prediction per bar: a 15m chart is evaluated every 15m, a 1h chart hourly.
  for (let i = candles.length - 2; i >= bundle.sequence + 48 && result.length < limit; i--) {
    const origin = candles[i]
    if (origin.time <= cutoff) break
    const prediction = predictB3(candles.slice(i - bundle.sequence - 48, i + 1), bundle)
    if (!prediction) continue
    result.push({ originTime: origin.time, time: candles[i + 1].time,
      predicted: origin.close * Math.exp(prediction.logReturns[0]), actual: candles[i + 1].close,
      baseline: origin.close })
  }
  return result.reverse()
}

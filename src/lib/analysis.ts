import type { Candle } from '../types'

export const HORIZON = 12

export type Bias = '偏多' | '中性' | '偏空'

export type Factor = {
  id: string
  label: string
  detail: string
  score: number
}

export type ForecastPoint = {
  time: number
  mid: number
  upper: number
  lower: number
}

export type LinePoint = {
  time: number
  value: number
}

export type Analysis = {
  bias: Bias
  score: number
  cap: number
  bulls: number
  bears: number
  factors: Factor[]
  rsi: number
  rsiSeries: number[]
  macd: { dif: number; dea: number; hist: number }
  bollPb: number
  ema20: number
  ema50: number
  atr: number
  forecast: ForecastPoint[]
  endPct: number
  note: string
}

export function ema(values: number[], period: number): (number | null)[] {
  const out: (number | null)[] = Array(values.length).fill(null)
  if (values.length < period) return out
  const k = 2 / (period + 1)
  let sum = 0
  for (let i = 0; i < period; i += 1) sum += values[i]
  let prev = sum / period
  out[period - 1] = prev
  for (let i = period; i < values.length; i += 1) {
    prev = values[i] * k + prev * (1 - k)
    out[i] = prev
  }
  return out
}

export function rsi(values: number[], period = 14): (number | null)[] {
  const out: (number | null)[] = Array(values.length).fill(null)
  if (values.length <= period) return out
  let gain = 0
  let loss = 0
  for (let i = 1; i <= period; i += 1) {
    const delta = values[i] - values[i - 1]
    if (delta >= 0) gain += delta
    else loss -= delta
  }
  let avgGain = gain / period
  let avgLoss = loss / period
  const calc = (g: number, l: number) => (l === 0 ? 100 : 100 - 100 / (1 + g / l))
  out[period] = calc(avgGain, avgLoss)
  for (let i = period + 1; i < values.length; i += 1) {
    const delta = values[i] - values[i - 1]
    avgGain = (avgGain * (period - 1) + (delta > 0 ? delta : 0)) / period
    avgLoss = (avgLoss * (period - 1) + (delta < 0 ? -delta : 0)) / period
    out[i] = calc(avgGain, avgLoss)
  }
  return out
}

export function macd(values: number[]): {
  dif: (number | null)[]
  dea: (number | null)[]
  hist: (number | null)[]
} {
  const fast = ema(values, 12)
  const slow = ema(values, 26)
  const dif: (number | null)[] = values.map((_, i) =>
    fast[i] != null && slow[i] != null ? (fast[i] as number) - (slow[i] as number) : null,
  )
  const start = dif.findIndex((value) => value != null)
  const dea: (number | null)[] = values.map(() => null)
  const hist: (number | null)[] = values.map(() => null)
  if (start < 0) return { dif, dea, hist }
  const valid = dif.slice(start).map((value) => value as number)
  const signal = ema(valid, 9)
  for (let i = 0; i < signal.length; i += 1) {
    if (signal[i] == null) continue
    dea[start + i] = signal[i]
    hist[start + i] = valid[i] - (signal[i] as number)
  }
  return { dif, dea, hist }
}

export function bollinger(values: number[], period = 20, mult = 2): {
  mid: (number | null)[]
  upper: (number | null)[]
  lower: (number | null)[]
  pb: (number | null)[]
} {
  const mid: (number | null)[] = Array(values.length).fill(null)
  const upper: (number | null)[] = Array(values.length).fill(null)
  const lower: (number | null)[] = Array(values.length).fill(null)
  const pb: (number | null)[] = Array(values.length).fill(null)
  for (let i = period - 1; i < values.length; i += 1) {
    let sum = 0
    for (let j = i - period + 1; j <= i; j += 1) sum += values[j]
    const mean = sum / period
    let variance = 0
    for (let j = i - period + 1; j <= i; j += 1) {
      const delta = values[j] - mean
      variance += delta * delta
    }
    const sd = Math.sqrt(variance / period)
    const up = mean + mult * sd
    const lo = mean - mult * sd
    const width = up - lo
    mid[i] = mean
    upper[i] = up
    lower[i] = lo
    pb[i] = width === 0 ? 0.5 : (values[i] - lo) / width
  }
  return { mid, upper, lower, pb }
}

export function wilderAtr(candles: Candle[], period = 14): number | null {
  if (candles.length < period + 1) return null
  const trs: number[] = []
  for (let i = 1; i < candles.length; i += 1) {
    const current = candles[i]
    const prevClose = candles[i - 1].close
    trs.push(Math.max(
      current.high - current.low,
      Math.abs(current.high - prevClose),
      Math.abs(current.low - prevClose),
    ))
  }
  let value = trs.slice(0, period).reduce((sum, item) => sum + item, 0) / period
  for (let i = period; i < trs.length; i += 1) {
    value = (value * (period - 1) + trs[i]) / period
  }
  return value
}

export function linregSlope(values: number[]): number | null {
  const n = values.length
  if (n < 5) return null
  let sumX = 0
  let sumY = 0
  let sumXY = 0
  let sumXX = 0
  for (let i = 0; i < n; i += 1) {
    sumX += i
    sumY += values[i]
    sumXY += i * values[i]
    sumXX += i * i
  }
  const denom = n * sumXX - sumX * sumX
  if (denom === 0) return null
  return (n * sumXY - sumX * sumY) / denom
}

export function linePoints(candles: Candle[], values: (number | null)[]): LinePoint[] {
  const points: LinePoint[] = []
  for (let i = 0; i < candles.length; i += 1) {
    const value = values[i]
    if (value == null || !Number.isFinite(value)) continue
    points.push({ time: candles[i].time, value })
  }
  return points
}

function lastOf(values: (number | null)[]): number | null {
  for (let i = values.length - 1; i >= 0; i -= 1) {
    if (values[i] != null && Number.isFinite(values[i])) return values[i]
  }
  return null
}

function rsiDetail(value: number): { detail: string; score: number } {
  if (value < 30) return { detail: `RSI ${value.toFixed(1)} · 超卖，只有回归倾向`, score: 0.4 }
  if (value < 42) return { detail: `RSI ${value.toFixed(1)} · 偏弱`, score: -0.6 }
  if (value <= 58) return { detail: `RSI ${value.toFixed(1)} · 中性区`, score: 0 }
  if (value <= 72) return { detail: `RSI ${value.toFixed(1)} · 偏强`, score: 0.6 }
  return { detail: `RSI ${value.toFixed(1)} · 超买，上涨拥挤`, score: -0.45 }
}

export function analyze(candles: Candle[], intervalMs: number): Analysis | null {
  if (candles.length < 60) return null
  const closes = candles.map((candle) => candle.close)
  const ema20Series = ema(closes, 20)
  const ema50Series = ema(closes, 50)
  const rsiSeries = rsi(closes, 14)
  const macdSeries = macd(closes)
  const bands = bollinger(closes, 20, 2)
  const ema20 = lastOf(ema20Series)
  const ema50 = lastOf(ema50Series)
  const rsiNow = lastOf(rsiSeries)
  const dif = lastOf(macdSeries.dif)
  const dea = lastOf(macdSeries.dea)
  const hist = lastOf(macdSeries.hist)
  const prevHist = macdSeries.hist.length > 1 ? macdSeries.hist[macdSeries.hist.length - 2] : null
  const bollPb = lastOf(bands.pb)
  const atr = wilderAtr(candles, 14)
  const slope = linregSlope(closes.slice(-30))
  if (
    ema20 == null || ema50 == null || rsiNow == null || dif == null || dea == null
    || hist == null || prevHist == null || bollPb == null || atr == null || slope == null || atr === 0
  ) {
    return null
  }

  const close = closes[closes.length - 1]
  const rsiVote = rsiDetail(rsiNow)
  let macdScore = 0
  let macdDetail = 'MACD 柱接近零轴'
  if (hist > 0 && hist > prevHist) {
    macdScore = 1
    macdDetail = 'MACD 柱在零轴上，并且继续放大'
  } else if (hist > 0) {
    macdScore = 0.45
    macdDetail = 'MACD 柱仍为正，但在缩短'
  } else if (hist < 0 && hist < prevHist) {
    macdScore = -1
    macdDetail = 'MACD 柱在零轴下，并且继续放大'
  } else if (hist < 0) {
    macdScore = -0.45
    macdDetail = 'MACD 柱仍为负，但在收敛'
  }

  const ratio = slope / atr
  const slopeScore = ratio > 0.08 ? 1 : ratio < -0.08 ? -1 : 0
  const slopeDetail = slopeScore > 0
    ? '近 30 根回归斜率向上'
    : slopeScore < 0
      ? '近 30 根回归斜率向下'
      : '近 30 根斜率几乎走平'

  const factors: Factor[] = [
    {
      id: 'location',
      label: '价格位置',
      detail: close >= ema20 ? '收在 EMA20 之上' : '收在 EMA20 之下',
      score: close >= ema20 ? 1 : -1,
    },
    {
      id: 'stack',
      label: '均线结构',
      detail: ema20 >= ema50 ? 'EMA20 高于 EMA50' : 'EMA20 低于 EMA50',
      score: ema20 >= ema50 ? 1 : -1,
    },
    { id: 'macd', label: 'MACD', detail: macdDetail, score: macdScore },
    { id: 'rsi', label: 'RSI', detail: rsiVote.detail, score: rsiVote.score },
    { id: 'slope', label: '短线斜率', detail: slopeDetail, score: slopeScore },
  ]

  const score = factors.reduce((sum, factor) => sum + factor.score, 0)
  const cap = 4.6
  const bias: Bias = score >= 2.2 ? '偏多' : score <= -2.2 ? '偏空' : '中性'
  const strong = factors.filter((factor) => Math.abs(factor.score) >= 0.45)
  const bulls = strong.filter((factor) => factor.score > 0).length
  const bears = strong.filter((factor) => factor.score < 0).length

  const forecast: ForecastPoint[] = []
  const step = intervalMs / 1000
  const lastTime = candles[candles.length - 1].time
  for (let i = 0; i <= HORIZON; i += 1) {
    const mid = close + slope * i
    const band = 1.15 * atr * Math.sqrt(i)
    forecast.push({
      time: lastTime + step * i,
      mid,
      upper: mid + band,
      lower: mid - band,
    })
  }
  const end = forecast[forecast.length - 1]
  const endPct = close === 0 ? 0 : ((end.mid - close) / close) * 100
  const move = Math.abs(endPct) < 0.05
    ? '中枢几乎走平'
    : `中枢${endPct > 0 ? '上移' : '下移'}约 ${Math.abs(endPct).toFixed(2)}%`
  const head = bias === '偏多'
    ? '规则合计偏向多头。'
    : bias === '偏空'
      ? '规则合计偏向空头。'
      : '规则互相抵消，先看作震荡。'
  const heat = rsiNow >= 72
    ? ' RSI 偏热，上涨比较拥挤。'
    : rsiNow <= 30
      ? ' RSI 超卖，下跌比较拥挤。'
      : ''
  const note = `${head}按近 30 根斜率外推，未来 ${HORIZON} 根的${move}。虚线随波动率张开，不是目标价。${heat}`

  return {
    bias,
    score,
    cap,
    bulls,
    bears,
    factors,
    rsi: rsiNow,
    rsiSeries: rsiSeries.filter((value): value is number => value != null).slice(-48),
    macd: { dif, dea, hist },
    bollPb,
    ema20,
    ema50,
    atr,
    forecast,
    endPct,
    note,
  }
}

export function overlaySeries(candles: Candle[]): {
  ema20: LinePoint[]
  ema50: LinePoint[]
  upper: LinePoint[]
  lower: LinePoint[]
} {
  const closes = candles.map((candle) => candle.close)
  const bands = bollinger(closes, 20, 2)
  return {
    ema20: linePoints(candles, ema(closes, 20)),
    ema50: linePoints(candles, ema(closes, 50)),
    upper: linePoints(candles, bands.upper),
    lower: linePoints(candles, bands.lower),
  }
}

import { ema, type ForecastPoint } from './analysis'
import type { Candle } from '../types'

export const LSTM_HIDDEN = 16
export const LSTM_SEQUENCE = 16
export const LSTM_FEATURES = 4
const HOLDOUT = 48
const VAL = 48
const BATCH = 32
const CLIP = 1
const BAND_Z = 1.28
const HORIZON = 12

export type TrainProgress = {
  epoch: number
  epochs: number
  trainLoss: number
}

export type NeuralForecast = {
  path: ForecastPoint[]
  endPct: number
  hitRate: number
  holdout: number
  maePct: number
  bias: '看多' | '看空' | '走平'
  epochs: number
  trainLoss: number
  valLoss: number
  params: number
  trainedHitRate: number
  trainedBars: number
  trainedSymbols: string[]
}

type Net = {
  h: number
  x: number
  w: Float64Array
  u: Float64Array
  b: Float64Array
  wy: Float64Array
  by: number
}

type Grads = {
  w: Float64Array
  u: Float64Array
  b: Float64Array
  wy: Float64Array
  by: number
}

type Adam = {
  mw: Float64Array
  vw: Float64Array
  mu: Float64Array
  vu: Float64Array
  mb: Float64Array
  vb: Float64Array
  mwy: Float64Array
  vwy: Float64Array
  mby: number
  vby: number
  t: number
  lr: number
}

type StepCache = {
  x: Float64Array
  hPrev: Float64Array
  cPrev: Float64Array
  i: Float64Array
  f: Float64Array
  g: Float64Array
  o: Float64Array
  c: Float64Array
  h: Float64Array
  tanhC: Float64Array
}

type Sample = {
  x: Float64Array[]
  y: number
}

function randn(): number {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function sigmoid(value: number): number {
  if (value > 18) return 1
  if (value < -18) return 0
  return 1 / (1 + Math.exp(-value))
}

function zeros(length: number): Float64Array {
  return new Float64Array(length)
}

function createNet(xSize = LSTM_FEATURES, hSize = LSTM_HIDDEN): Net {
  const gates = 4 * hSize
  const scaleIn = Math.sqrt(2 / (xSize + hSize))
  const scaleRec = Math.sqrt(1 / hSize)
  const w = zeros(gates * xSize)
  const u = zeros(gates * hSize)
  const b = zeros(gates)
  for (let i = 0; i < w.length; i += 1) w[i] = randn() * scaleIn
  for (let i = 0; i < u.length; i += 1) u[i] = randn() * scaleRec
  for (let i = 0; i < hSize; i += 1) b[hSize + i] = 1
  const wy = zeros(hSize)
  for (let i = 0; i < hSize; i += 1) wy[i] = randn() * Math.sqrt(1 / hSize)
  return { h: hSize, x: xSize, w, u, b, wy, by: 0 }
}

function copyNet(net: Net): Net {
  return {
    h: net.h,
    x: net.x,
    w: net.w.slice(),
    u: net.u.slice(),
    b: net.b.slice(),
    wy: net.wy.slice(),
    by: net.by,
  }
}

function emptyGrads(net: Net): Grads {
  return {
    w: zeros(net.w.length),
    u: zeros(net.u.length),
    b: zeros(net.b.length),
    wy: zeros(net.wy.length),
    by: 0,
  }
}

function createAdam(net: Net): Adam {
  return {
    mw: zeros(net.w.length),
    vw: zeros(net.w.length),
    mu: zeros(net.u.length),
    vu: zeros(net.u.length),
    mb: zeros(net.b.length),
    vb: zeros(net.b.length),
    mwy: zeros(net.wy.length),
    vwy: zeros(net.wy.length),
    mby: 0,
    vby: 0,
    t: 0,
    lr: 0.003,
  }
}

function forward(seq: Float64Array[], net: Net): { y: number; caches: StepCache[] } {
  const { h: hSize, x: xSize } = net
  let h = zeros(hSize)
  let c = zeros(hSize)
  const caches: StepCache[] = []
  for (const x of seq) {
    const iGate = zeros(hSize)
    const fGate = zeros(hSize)
    const gGate = zeros(hSize)
    const oGate = zeros(hSize)
    const cNext = zeros(hSize)
    const hNext = zeros(hSize)
    const tanhC = zeros(hSize)
    for (let unit = 0; unit < hSize; unit += 1) {
      let zi = net.b[unit]
      let zf = net.b[hSize + unit]
      let zg = net.b[2 * hSize + unit]
      let zo = net.b[3 * hSize + unit]
      for (let input = 0; input < xSize; input += 1) {
        const value = x[input]
        zi += net.w[unit * xSize + input] * value
        zf += net.w[(hSize + unit) * xSize + input] * value
        zg += net.w[(2 * hSize + unit) * xSize + input] * value
        zo += net.w[(3 * hSize + unit) * xSize + input] * value
      }
      for (let prev = 0; prev < hSize; prev += 1) {
        const value = h[prev]
        zi += net.u[unit * hSize + prev] * value
        zf += net.u[(hSize + unit) * hSize + prev] * value
        zg += net.u[(2 * hSize + unit) * hSize + prev] * value
        zo += net.u[(3 * hSize + unit) * hSize + prev] * value
      }
      iGate[unit] = sigmoid(zi)
      fGate[unit] = sigmoid(zf)
      gGate[unit] = Math.tanh(zg)
      oGate[unit] = sigmoid(zo)
      cNext[unit] = fGate[unit] * c[unit] + iGate[unit] * gGate[unit]
      tanhC[unit] = Math.tanh(cNext[unit])
      hNext[unit] = oGate[unit] * tanhC[unit]
    }
    caches.push({
      x,
      hPrev: h,
      cPrev: c,
      i: iGate,
      f: fGate,
      g: gGate,
      o: oGate,
      c: cNext,
      h: hNext,
      tanhC,
    })
    h = hNext
    c = cNext
  }
  let y = net.by
  for (let unit = 0; unit < hSize; unit += 1) y += net.wy[unit] * h[unit]
  return { y, caches }
}

function backward(y: number, target: number, caches: StepCache[], net: Net, grads: Grads, scale: number): void {
  const { h: hSize, x: xSize } = net
  const dy = 2 * (y - target) * scale
  const last = caches[caches.length - 1]
  for (let unit = 0; unit < hSize; unit += 1) grads.wy[unit] += last.h[unit] * dy
  grads.by += dy
  let dh = zeros(hSize)
  for (let unit = 0; unit < hSize; unit += 1) dh[unit] = net.wy[unit] * dy
  let dcNext = zeros(hSize)
  for (let time = caches.length - 1; time >= 0; time -= 1) {
    const cache = caches[time]
    const dhPrev = zeros(hSize)
    const dcPrev = zeros(hSize)
    for (let unit = 0; unit < hSize; unit += 1) {
      const dc = dh[unit] * cache.o[unit] * (1 - cache.tanhC[unit] * cache.tanhC[unit]) + dcNext[unit]
      const dO = dh[unit] * cache.tanhC[unit] * cache.o[unit] * (1 - cache.o[unit])
      const dI = dc * cache.g[unit] * cache.i[unit] * (1 - cache.i[unit])
      const dF = dc * cache.cPrev[unit] * cache.f[unit] * (1 - cache.f[unit])
      const dG = dc * cache.i[unit] * (1 - cache.g[unit] * cache.g[unit])
      const packed = [dI, dF, dG, dO]
      for (let gate = 0; gate < 4; gate += 1) {
        const row = gate * hSize + unit
        const grad = packed[gate]
        grads.b[row] += grad
        for (let input = 0; input < xSize; input += 1) grads.w[row * xSize + input] += grad * cache.x[input]
        for (let prev = 0; prev < hSize; prev += 1) {
          grads.u[row * hSize + prev] += grad * cache.hPrev[prev]
          dhPrev[prev] += net.u[row * hSize + prev] * grad
        }
      }
      dcPrev[unit] = dc * cache.f[unit]
    }
    dh = dhPrev
    dcNext = dcPrev
  }
}

function clip(grads: Grads): void {
  let norm = grads.by * grads.by
  for (const value of grads.w) norm += value * value
  for (const value of grads.u) norm += value * value
  for (const value of grads.b) norm += value * value
  for (const value of grads.wy) norm += value * value
  norm = Math.sqrt(norm)
  if (norm <= CLIP || norm === 0) return
  const scale = CLIP / norm
  grads.by *= scale
  grads.w.forEach((_, index) => { grads.w[index] *= scale })
  grads.u.forEach((_, index) => { grads.u[index] *= scale })
  grads.b.forEach((_, index) => { grads.b[index] *= scale })
  grads.wy.forEach((_, index) => { grads.wy[index] *= scale })
}

function adamStep(param: Float64Array, grad: Float64Array, moment: Float64Array, velocity: Float64Array, t: number, lr: number): void {
  const b1 = 0.9
  const b2 = 0.999
  const fix1 = 1 - b1 ** t
  const fix2 = 1 - b2 ** t
  for (let index = 0; index < param.length; index += 1) {
    moment[index] = b1 * moment[index] + (1 - b1) * grad[index]
    velocity[index] = b2 * velocity[index] + (1 - b2) * grad[index] * grad[index]
    param[index] -= lr * (moment[index] / fix1) / (Math.sqrt(velocity[index] / fix2) + 1e-8)
  }
}

function applyAdam(net: Net, grads: Grads, adam: Adam): void {
  adam.t += 1
  adamStep(net.w, grads.w, adam.mw, adam.vw, adam.t, adam.lr)
  adamStep(net.u, grads.u, adam.mu, adam.vu, adam.t, adam.lr)
  adamStep(net.b, grads.b, adam.mb, adam.vb, adam.t, adam.lr)
  adamStep(net.wy, grads.wy, adam.mwy, adam.vwy, adam.t, adam.lr)
  const b1 = 0.9
  const b2 = 0.999
  adam.mby = b1 * adam.mby + (1 - b1) * grads.by
  adam.vby = b2 * adam.vby + (1 - b2) * grads.by * grads.by
  net.by -= adam.lr * (adam.mby / (1 - b1 ** adam.t)) / (Math.sqrt(adam.vby / (1 - b2 ** adam.t)) + 1e-8)
}

function predict(seq: Float64Array[], net: Net): number {
  return Math.max(-0.08, Math.min(0.08, forward(seq, net).y))
}

function featureAt(
  closes: number[],
  highs: number[],
  lows: number[],
  volumes: number[],
  ema20: (number | null)[],
  index: number,
): number[] | null {
  if (index < 1 || closes[index] <= 0 || closes[index - 1] <= 0) return null
  const fast = ema20[index]
  if (fast == null) return null
  const volumeChange = volumes[index] > 0 && volumes[index - 1] > 0
    ? Math.log(volumes[index] / volumes[index - 1])
    : 0
  return [
    Math.log(closes[index] / closes[index - 1]),
    (highs[index] - lows[index]) / closes[index],
    volumeChange,
    (closes[index] - fast) / closes[index],
  ]
}

function buildSamples(candles: Candle[]): { samples: Sample[]; closes: number[]; highs: number[]; lows: number[]; volumes: number[] } | null {
  const closes = candles.map((candle) => candle.close)
  const highs = candles.map((candle) => candle.high)
  const lows = candles.map((candle) => candle.low)
  const volumes = candles.map((candle) => candle.volume)
  const ema20 = ema(closes, 20)
  const samples: Sample[] = []
  for (let index = LSTM_SEQUENCE; index < closes.length - 1; index += 1) {
    const rows: Float64Array[] = []
    let valid = true
    for (let step = LSTM_SEQUENCE - 1; step >= 0; step -= 1) {
      const raw = featureAt(closes, highs, lows, volumes, ema20, index - step)
      if (!raw) {
        valid = false
        break
      }
      rows.push(Float64Array.from(raw))
    }
    if (!valid || closes[index + 1] <= 0) continue
    samples.push({ x: rows, y: Math.log(closes[index + 1] / closes[index]) })
  }
  if (samples.length < HOLDOUT + VAL + 40) return null
  return { samples, closes, highs, lows, volumes }
}

function standardize(samples: Sample[], end: number): { mean: Float64Array; scale: Float64Array } {
  const mean = zeros(LSTM_FEATURES)
  const scale = zeros(LSTM_FEATURES)
  let count = 0
  for (let index = 0; index < end; index += 1) {
    for (const row of samples[index].x) {
      for (let feature = 0; feature < LSTM_FEATURES; feature += 1) mean[feature] += row[feature]
      count += 1
    }
  }
  for (let feature = 0; feature < LSTM_FEATURES; feature += 1) mean[feature] /= count
  for (let index = 0; index < end; index += 1) {
    for (const row of samples[index].x) {
      for (let feature = 0; feature < LSTM_FEATURES; feature += 1) {
        const delta = row[feature] - mean[feature]
        scale[feature] += delta * delta
      }
    }
  }
  for (let feature = 0; feature < LSTM_FEATURES; feature += 1) {
    scale[feature] = Math.sqrt(scale[feature] / count)
    if (scale[feature] < 1e-8) scale[feature] = 1
  }
  return { mean, scale }
}

function applyScale(sample: Sample, mean: Float64Array, scale: Float64Array): Sample {
  return {
    y: sample.y,
    x: sample.x.map((row) => {
      const next = zeros(row.length)
      for (let feature = 0; feature < row.length; feature += 1) next[feature] = (row[feature] - mean[feature]) / scale[feature]
      return next
    }),
  }
}

function lossOf(samples: Sample[], net: Net): number {
  if (!samples.length) return 0
  let loss = 0
  for (const sample of samples) loss += (forward(sample.x, net).y - sample.y) ** 2
  return loss / samples.length
}

function evaluate(samples: Sample[], net: Net): { hitRate: number; maePct: number; residual: number } {
  let hits = 0
  let counted = 0
  let absError = 0
  const residuals: number[] = []
  for (const sample of samples) {
    const predicted = predict(sample.x, net)
    residuals.push(sample.y - predicted)
    absError += Math.abs(Math.exp(predicted) - Math.exp(sample.y))
    counted += 1
    if (predicted !== 0 && sample.y !== 0 && Math.sign(predicted) === Math.sign(sample.y)) hits += 1
  }
  const mean = residuals.reduce((sum, value) => sum + value, 0) / residuals.length
  const variance = residuals.reduce((sum, value) => sum + (value - mean) ** 2, 0) / residuals.length
  return {
    hitRate: counted ? hits / counted : 0,
    maePct: counted ? (absError / counted) * 100 : 0,
    residual: Math.sqrt(variance),
  }
}

function yieldTick(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0)
  })
}

export async function trainNeuralForecast(
  candles: Candle[],
  intervalMs: number,
  onProgress: (progress: TrainProgress) => void,
  cancelled: () => boolean,
): Promise<NeuralForecast | null> {
  const built = buildSamples(candles)
  if (!built) return null
  const { samples } = built
  const testStart = samples.length - HOLDOUT
  const valStart = testStart - VAL
  const scaler = standardize(samples, valStart)
  const scaled = samples.map((sample) => applyScale(sample, scaler.mean, scaler.scale))
  const train = scaled.slice(0, valStart)
  const validation = scaled.slice(valStart, testStart)
  const test = scaled.slice(testStart)
  const net = createNet()
  const adam = createAdam(net)
  let best = copyNet(net)
  let bestVal = Number.POSITIVE_INFINITY
  let bestEpoch = 0
  let wait = 0
  const epochs = 24
  const order = train.map((_, index) => index)

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    if (cancelled()) return null
    for (let index = order.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(Math.random() * (index + 1))
      ;[order[index], order[swap]] = [order[swap], order[index]]
    }
    let trainLoss = 0
    for (let start = 0; start < order.length; start += BATCH) {
      const grads = emptyGrads(net)
      const batch = order.slice(start, start + BATCH)
      for (const index of batch) {
        const sample = train[index]
        const passed = forward(sample.x, net)
        trainLoss += (passed.y - sample.y) ** 2
        backward(passed.y, sample.y, passed.caches, net, grads, 1 / batch.length)
      }
      clip(grads)
      applyAdam(net, grads, adam)
    }
    trainLoss /= order.length
    const valLoss = lossOf(validation, net)
    onProgress({ epoch, epochs, trainLoss })
    if (valLoss < bestVal - 1e-8) {
      bestVal = valLoss
      best = copyNet(net)
      bestEpoch = epoch
      wait = 0
    } else if (++wait >= 6) {
      break
    }
    await yieldTick()
  }

  const score = evaluate(test, best)
  const sigma = Math.max(score.residual, 1e-6)
  const closes = candles.map((candle) => candle.close)
  const highs = candles.map((candle) => candle.high)
  const lows = candles.map((candle) => candle.low)
  const volumes = candles.map((candle) => candle.volume)
  const last = closes[closes.length - 1]
  const lastTime = candles[candles.length - 1].time
  const stepMs = intervalMs / 1000
  const extendedClose = closes.slice()
  const extendedHigh = highs.slice()
  const extendedLow = lows.slice()
  const extendedVolume = volumes.slice()
  const path: ForecastPoint[] = [{ time: lastTime, mid: last, upper: last, lower: last }]
  for (let horizon = 1; horizon <= HORIZON; horizon += 1) {
    const ema20 = ema(extendedClose, 20)
    const index = extendedClose.length - 1
    const rows: Float64Array[] = []
    for (let back = LSTM_SEQUENCE - 1; back >= 0; back -= 1) {
      const raw = featureAt(extendedClose, extendedHigh, extendedLow, extendedVolume, ema20, index - back)
      if (!raw) return null
      const scaledRow = zeros(LSTM_FEATURES)
      for (let feature = 0; feature < LSTM_FEATURES; feature += 1) {
        scaledRow[feature] = (raw[feature] - scaler.mean[feature]) / scaler.scale[feature]
      }
      rows.push(scaledRow)
    }
    const predicted = predict(rows, best)
    const mid = extendedClose[index] * Math.exp(predicted)
    const band = BAND_Z * sigma * Math.sqrt(horizon)
    extendedClose.push(mid)
    extendedHigh.push(mid)
    extendedLow.push(mid)
    extendedVolume.push(extendedVolume[extendedVolume.length - 1] || 1)
    path.push({
      time: lastTime + stepMs * horizon,
      mid,
      upper: mid * Math.exp(band),
      lower: mid * Math.exp(-band),
    })
  }
  const endPct = last === 0 ? 0 : ((path[path.length - 1].mid - last) / last) * 100
  return {
    path,
    endPct,
    hitRate: score.hitRate,
    holdout: test.length,
    maePct: score.maePct,
    bias: Math.abs(endPct) < 0.2 ? '走平' : endPct > 0 ? '看多' : '看空',
    epochs: bestEpoch,
    trainLoss: lossOf(train, best),
    valLoss: bestVal,
    params: best.w.length + best.u.length + best.b.length + best.wy.length + 1,
    trainedHitRate: score.hitRate,
    trainedBars: candles.length,
    trainedSymbols: [],
  }
}

export type LstmBundle = {
  interval: string
  mean: number[]
  scale: number[]
  w: number[]
  u: number[]
  b: number[]
  wy: number[]
  by: number
  hidden: number
  sequence: number
  epochs: number
  trainLoss: number
  valLoss: number
  testHitRate: number
  testMaePct: number
  residual: number
  symbols: string[]
  bars: number
}

function samplesOf(candles: Candle[]): Sample[] {
  return buildSamples(candles)?.samples ?? []
}

export async function trainSeriesBundle(
  groups: { symbol: string; candles: Candle[] }[],
  interval: string,
  onEpoch?: (info: { epoch: number; maxEpochs: number; trainLoss: number; valLoss: number; lr: number; wait: number }) => void,
): Promise<LstmBundle | null> {
  const parts = groups
    .map((group) => ({ symbol: group.symbol, samples: samplesOf(group.candles) }))
    .filter((part) => part.samples.length >= HOLDOUT + VAL + 40)
  if (!parts.length) return null
  const trainRaw: Sample[] = []
  const valRaw: Sample[] = []
  const testRaw: Sample[] = []
  for (const part of parts) {
    const { samples } = part
    trainRaw.push(...samples.slice(0, samples.length - HOLDOUT - VAL))
    valRaw.push(...samples.slice(samples.length - HOLDOUT - VAL, samples.length - HOLDOUT))
    testRaw.push(...samples.slice(samples.length - HOLDOUT))
  }
  const scaler = standardize(trainRaw, trainRaw.length)
  const train = trainRaw.map((sample) => applyScale(sample, scaler.mean, scaler.scale))
  const validation = valRaw.map((sample) => applyScale(sample, scaler.mean, scaler.scale))
  const test = testRaw.map((sample) => applyScale(sample, scaler.mean, scaler.scale))
  const net = createNet()
  const adam = createAdam(net)
  let best = copyNet(net)
  let bestVal = Number.POSITIVE_INFINITY
  let bestEpoch = 0
  let wait = 0
  let decays = 0
  const maxEpochs = 80
  const minEpochs = 20
  const order = train.map((_, index) => index)

  for (let epoch = 1; epoch <= maxEpochs; epoch += 1) {
    for (let index = order.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(Math.random() * (index + 1))
      ;[order[index], order[swap]] = [order[swap], order[index]]
    }
    let trainLoss = 0
    for (let start = 0; start < order.length; start += BATCH) {
      const grads = emptyGrads(net)
      const batch = order.slice(start, start + BATCH)
      for (const index of batch) {
        const sample = train[index]
        const passed = forward(sample.x, net)
        trainLoss += (passed.y - sample.y) ** 2
        backward(passed.y, sample.y, passed.caches, net, grads, 1 / batch.length)
      }
      clip(grads)
      applyAdam(net, grads, adam)
    }
    trainLoss /= order.length
    const valLoss = lossOf(validation, net)
    onEpoch?.({ epoch, maxEpochs, trainLoss, valLoss, lr: adam.lr, wait })
    if (valLoss < bestVal * (1 - 1e-4)) {
      bestVal = valLoss
      best = copyNet(net)
      bestEpoch = epoch
      wait = 0
    } else {
      wait += 1
      if (wait === 8 && decays < 3) {
        adam.lr *= 0.5
        decays += 1
        wait = 0
      } else if (wait >= 12 && epoch >= minEpochs) {
        break
      }
    }
  }

  const score = evaluate(test, best)
  return {
    interval,
    mean: Array.from(scaler.mean),
    scale: Array.from(scaler.scale),
    w: Array.from(best.w),
    u: Array.from(best.u),
    b: Array.from(best.b),
    wy: Array.from(best.wy),
    by: best.by,
    hidden: best.h,
    sequence: LSTM_SEQUENCE,
    epochs: bestEpoch,
    trainLoss: lossOf(train, best),
    valLoss: bestVal,
    testHitRate: score.hitRate,
    testMaePct: score.maePct,
    residual: Math.max(score.residual, 1e-6),
    symbols: parts.map((part) => part.symbol),
    bars: groups.reduce((sum, group) => sum + group.candles.length, 0),
  }
}

function netFromBundle(bundle: LstmBundle): Net {
  return {
    h: bundle.hidden,
    x: LSTM_FEATURES,
    w: Float64Array.from(bundle.w),
    u: Float64Array.from(bundle.u),
    b: Float64Array.from(bundle.b),
    wy: Float64Array.from(bundle.wy),
    by: bundle.by,
  }
}

export function inferForecast(candles: Candle[], intervalMs: number, bundle: LstmBundle): NeuralForecast | null {
  const built = buildSamples(candles)
  if (!built) return null
  const mean = Float64Array.from(bundle.mean)
  const scale = Float64Array.from(bundle.scale)
  const net = netFromBundle(bundle)
  const recent = built.samples.slice(-HOLDOUT).map((sample) => applyScale(sample, mean, scale))
  const score = recent.length >= 20 ? evaluate(recent, net) : null
  const sigma = Math.max(score?.residual ?? bundle.residual, 1e-6)
  const closes = candles.map((candle) => candle.close)
  const highs = candles.map((candle) => candle.high)
  const lows = candles.map((candle) => candle.low)
  const volumes = candles.map((candle) => candle.volume)
  const last = closes[closes.length - 1]
  const lastTime = candles[candles.length - 1].time
  const step = intervalMs / 1000
  const extendedClose = closes.slice()
  const extendedHigh = highs.slice()
  const extendedLow = lows.slice()
  const extendedVolume = volumes.slice()
  const path: ForecastPoint[] = [{ time: lastTime, mid: last, upper: last, lower: last }]
  for (let horizon = 1; horizon <= HORIZON; horizon += 1) {
    const ema20 = ema(extendedClose, 20)
    const index = extendedClose.length - 1
    const rows: Float64Array[] = []
    for (let back = LSTM_SEQUENCE - 1; back >= 0; back -= 1) {
      const raw = featureAt(extendedClose, extendedHigh, extendedLow, extendedVolume, ema20, index - back)
      if (!raw) return null
      const scaledRow = zeros(LSTM_FEATURES)
      for (let feature = 0; feature < LSTM_FEATURES; feature += 1) {
        scaledRow[feature] = (raw[feature] - mean[feature]) / scale[feature]
      }
      rows.push(scaledRow)
    }
    const predicted = predict(rows, net)
    const mid = extendedClose[index] * Math.exp(predicted)
    const band = BAND_Z * sigma * Math.sqrt(horizon)
    extendedClose.push(mid)
    extendedHigh.push(mid)
    extendedLow.push(mid)
    extendedVolume.push(extendedVolume[extendedVolume.length - 1] || 1)
    path.push({
      time: lastTime + step * horizon,
      mid,
      upper: mid * Math.exp(band),
      lower: mid * Math.exp(-band),
    })
  }
  const endPct = last === 0 ? 0 : ((path[path.length - 1].mid - last) / last) * 100
  return {
    path,
    endPct,
    hitRate: score?.hitRate ?? bundle.testHitRate,
    holdout: recent.length,
    maePct: score?.maePct ?? bundle.testMaePct,
    bias: Math.abs(endPct) < 0.2 ? '走平' : endPct > 0 ? '看多' : '看空',
    epochs: bundle.epochs,
    trainLoss: bundle.trainLoss,
    valLoss: bundle.valLoss,
    params: bundle.w.length + bundle.u.length + bundle.b.length + bundle.wy.length + 1,
    trainedHitRate: bundle.testHitRate,
    trainedBars: bundle.bars,
    trainedSymbols: bundle.symbols,
  }
}

function probe(read: () => number, write: (value: number) => void, analytic: number, target: number, seq: Float64Array[], net: Net): number {
  const epsilon = 1e-5
  const before = read()
  write(before + epsilon)
  const plus = (forward(seq, net).y - target) ** 2
  write(before - epsilon)
  const minus = (forward(seq, net).y - target) ** 2
  write(before)
  const numeric = (plus - minus) / (2 * epsilon)
  return Math.abs(numeric - analytic) / Math.max(1, Math.abs(numeric), Math.abs(analytic))
}

export function gradientError(): number {
  const net = createNet(2, 2)
  const seq = [Float64Array.from([0.2, -0.1]), Float64Array.from([0.05, 0.3])]
  const target = 0.04
  const passed = forward(seq, net)
  const grads = emptyGrads(net)
  backward(passed.y, target, passed.caches, net, grads, 1)
  return Math.max(
    probe(() => net.w[3], (value) => { net.w[3] = value }, grads.w[3], target, seq, net),
    probe(() => net.u[1], (value) => { net.u[1] = value }, grads.u[1], target, seq, net),
    probe(() => net.wy[0], (value) => { net.wy[0] = value }, grads.wy[0], target, seq, net),
    probe(() => net.by, (value) => { net.by = value }, grads.by, target, seq, net),
  )
}

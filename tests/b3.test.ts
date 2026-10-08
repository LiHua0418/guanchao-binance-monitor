import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, test } from 'node:test'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'
import * as b3 from '../src/lib/b3.ts'
import { bufferCandle, candlesNeedBackfill, fetchKlines, mergeCandle } from '../src/lib/binance.ts'
import { INTERVALS, INTERVAL_MS, type Candle, type Interval } from '../src/types.ts'
import {
  buildB3Features,
  closedCandles,
  forecastB3,
  forwardB3,
  predictB3,
  replayB3,
  standardizeB3,
  validateB3Bundle,
  type B3Bundle,
} from '../src/lib/b3.ts'

type ParityCase = {
  interval: Interval
  symbol: string
  origin: string
  candles: Candle[]
  rawFeatures: number[][]
  standardizedInput: number[][]
  networkOutput: number[]
  forecastLogReturns: number[]
  forecastPrices: number[]
  lowerPrices: number[]
  upperPrices: number[]
}

type Split = { samples: number; firstOrigin: string; lastOrigin: string; lastTarget: string }

const readJson = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'))
const parity = readJson('../reports/b3-parity.json') as { tolerance: number; cases: ParityCase[] }

function near(actual: number, expected: number, tolerance: number, label: string) {
  assert.ok(Number.isFinite(actual), `${label}: non-finite result ${actual}`)
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} differs from ${expected} by more than ${tolerance}`)
}

function nearArray(actual: number[], expected: number[], tolerance: number, label: string) {
  assert.equal(actual.length, expected.length, `${label} length`)
  actual.forEach((value, i) => near(value, expected[i], tolerance, `${label}[${i}]`))
}

function nearMatrix(actual: number[][], expected: number[][], tolerance: number, label: string) {
  assert.equal(actual.length, expected.length, `${label} length`)
  actual.forEach((row, i) => nearArray(row, expected[i], tolerance, `${label}[${i}]`))
}

function syntheticCandles(bundle: B3Bundle, count = 320): Candle[] {
  const step = INTERVAL_MS[bundle.interval as Interval] / 1000
  const start = Date.parse(bundle.dataCutoff) / 1000 - 120 * step
  return Array.from({ length: count }, (_, i) => {
    const price = (j: number) => 100 * Math.exp(j * 0.0005 + Math.sin(j * 0.4) * 0.008)
    const open = price(i - 1)
    const close = price(i)
    return { time: start + i * step, open, close, high: Math.max(open, close) * 1.002,
      low: Math.min(open, close) * 0.998, volume: 1000 + i * 3 }
  })
}

test('portable parity fixtures cover every supported timeframe', () => {
  assert.deepEqual(parity.cases.map((item) => item.interval).sort(), [...INTERVALS].sort())
  assert.equal(new Set(parity.cases.map((item) => item.interval)).size, INTERVALS.length)
})

for (const interval of INTERVALS) {
  describe(`${interval} exported Python / browser inference parity`, () => {
    const bundle = validateB3Bundle(readJson(`../public/models/b3-${interval}.json`), interval)
    const fixture = parity.cases.find((item) => item.interval === interval)!
    const step = INTERVAL_MS[interval]

    test('causal features and normalization agree with the training pipeline', () => {
      const features = buildB3Features(fixture.candles)
      assert.ok(features)
      nearMatrix(features, fixture.rawFeatures, 1e-9, 'raw features')
      nearMatrix(standardizeB3(features, bundle), fixture.standardizedInput, 1e-5, 'standardized features')
    })

    test('unblended CNN, LSTM, attention outputs agree even for a flat fallback model', () => {
      const output = forwardB3(fixture.standardizedInput, bundle)
      nearArray(output, fixture.networkOutput, parity.tolerance, 'network output')
      const prediction = predictB3(fixture.candles, bundle)
      assert.ok(prediction)
      nearArray(prediction.networkOutput, fixture.networkOutput, parity.tolerance, 'end-to-end network output')
      nearArray(prediction.logReturns, fixture.forecastLogReturns, 1e-6, 'cumulative returns')
    })

    test('each forecast horizon uses its own blend instead of broadcasting the first horizon fallback', () => {
      const independent = structuredClone(bundle)
      independent.blendByHorizon = [0, 0.25, 0.5, 0.75, 1, 0.5, 0.25, 1, 0.75, 0.5, 0.25, 1]
      independent.blend = independent.blendByHorizon[0]
      const prediction = predictB3(fixture.candles, validateB3Bundle(independent, interval))
      assert.ok(prediction)
      prediction.logReturns.forEach((value, i) => near(value,
        prediction.networkOutput[i] * independent.targetScale[i] * independent.blendByHorizon[i], 1e-12, `blend horizon ${i + 1}`))
      assert.equal(Math.abs(prediction.logReturns[0]), 0)
      assert.ok(prediction.logReturns.slice(1).some((value) => Math.abs(value) > 1e-12))
    })

    test('forecast has one actual anchor and twelve correctly timed direct prices and calibrated bounds', () => {
      const forecast = forecastB3(fixture.candles, step, bundle)
      assert.ok(forecast)
      assert.equal(forecast.path.length, 13)
      const anchor = fixture.candles.at(-1)!
      assert.deepEqual(forecast.path[0], { time: anchor.time, mid: anchor.close, lower: anchor.close, upper: anchor.close })
      for (let i = 0; i < 12; i += 1) {
        const point = forecast.path[i + 1]
        assert.equal(point.time, anchor.time + (i + 1) * step / 1000)
        near(point.mid, fixture.forecastPrices[i], Math.max(1e-6, fixture.forecastPrices[i] * 1e-6), `price ${i + 1}`)
        near(point.lower, fixture.lowerPrices[i], Math.max(1e-6, fixture.lowerPrices[i] * 1e-6), `lower ${i + 1}`)
        near(point.upper, fixture.upperPrices[i], Math.max(1e-6, fixture.upperPrices[i] * 1e-6), `upper ${i + 1}`)
        assert.ok(point.lower > 0 && point.upper >= point.lower)
      }
    })

    test('an unclosed live candle cannot move the forecast anchor or predictions', () => {
      const last = fixture.candles.at(-1)!
      const live = { ...last, time: last.time + step / 1000, close: last.close * 3,
        high: last.close * 3, volume: last.volume * 10 }
      const now = live.time * 1000 + step - 1
      const onlyClosed = closedCandles([...fixture.candles, live], step, now)
      assert.deepEqual(onlyClosed, fixture.candles)
      assert.deepEqual(forecastB3(onlyClosed, step, bundle), forecastB3(fixture.candles, step, bundle))
      assert.equal(closedCandles([...fixture.candles, live], step, now + 1).at(-1)?.time, live.time)
    })

    test('an explicit non-final exchange candle remains excluded even after its scheduled close', () => {
      const last = fixture.candles.at(-1)!
      const nonFinal = { ...last, time: last.time + step / 1000, isClosed: false }
      const now = nonFinal.time * 1000 + step * 2
      assert.deepEqual(closedCandles([...fixture.candles, nonFinal], step, now), fixture.candles)
      assert.equal(nonFinal.isClosed, false, 'filter must not upgrade a stale snapshot to final')
      const final = { ...nonFinal, isClosed: true }
      assert.equal(closedCandles([...fixture.candles, final], step, now).at(-1)?.time, final.time)
      assert.deepEqual(closedCandles([final], step, final.time * 1000), [], 'exchange flag alone cannot admit a future candle')
    })

    test('insufficient feature history, non-finite candles and invalid OHLC are rejected', () => {
      assert.equal(buildB3Features(fixture.candles.slice(-96)), null)
      assert.equal(predictB3(fixture.candles.slice(-96), bundle), null)
      for (const patch of [{ close: NaN }, { high: Infinity }, { volume: NaN }, { close: 0 }, { low: -1 }]) {
        const damaged = structuredClone(fixture.candles)
        Object.assign(damaged[80], patch)
        assert.equal(predictB3(damaged, bundle), null, `accepted ${Object.keys(patch)[0]}`)
      }
      assert.ok(predictB3(fixture.candles, bundle))
    })

    test('duplicate, out-of-order or missing candle timestamps cannot create a forecast', () => {
      for (const offset of [0, -step / 1000, 2 * step / 1000]) {
        const damaged = structuredClone(fixture.candles)
        damaged[80].time = damaged[79].time + offset
        assert.equal(forecastB3(damaged, step, bundle), null)
      }
      const missing = fixture.candles.filter((_, i) => i !== 20)
      assert.equal(forecastB3([...missing, { ...fixture.candles.at(-1)!, time: fixture.candles.at(-1)!.time + step / 1000 }], step, bundle), null)
    })
  })

  describe(`${interval} historical replay`, () => {
    const bundle = validateB3Bundle(readJson(`../public/models/b3-${interval}.json`), interval)
    const candles = syntheticCandles(bundle)
    const step = INTERVAL_MS[interval]

    test('dense replay covers every eligible bar after fitting/calibration with an observed next-bar target', () => {
      const replay = replayB3(candles, step, bundle)
      const eligible = candles.slice(96, -1).filter((candle) => candle.time * 1000 > Date.parse(bundle.dataCutoff))
      assert.ok(replay.length > 12 && replay.length <= 240)
      assert.deepEqual(replay.map((point) => point.originTime), eligible.slice(-240).map((candle) => candle.time))
      for (const point of replay) {
        assert.ok(point.originTime * 1000 > Date.parse(bundle.dataCutoff))
        assert.equal(point.time, point.originTime + step / 1000)
        assert.ok(point.time <= candles.at(-1)!.time)
        const actual = candles.find((candle) => candle.time === point.time)
        assert.ok(actual)
        assert.equal(point.actual, actual.close)
        assert.ok(Number.isFinite(point.predicted) && point.predicted > 0)
        const prefix = candles.filter((candle) => candle.time <= point.originTime)
        const forecast = forecastB3(prefix, step, bundle)
        assert.ok(forecast)
        near(point.predicted, forecast.path[1].mid, 1e-8, 'historical next-bar prediction')
      }
      const ordered = [...replay].sort((a, b) => a.originTime - b.originTime)
      for (let i = 1; i < ordered.length; i += 1) {
        assert.equal(ordered[i].originTime - ordered[i - 1].originTime, step / 1000)
      }
      assert.deepEqual(replayB3(candles, step, bundle, 3), replay.slice(-3))
    })

    test('dense replay retains the latest 240 eligible bar predictions by default', () => {
      const extended = syntheticCandles(bundle, 500)
      const replay = replayB3(extended, step, bundle)
      assert.equal(replay.length, 240)
      assert.equal(replay.at(-1)?.time, extended.at(-1)?.time)
      assert.equal(replay[0].time, extended.at(-240)?.time)
    })

    test('changing data after an origin changes its realized outcome but not its recorded prediction', () => {
      const original = replayB3(candles, step, bundle)
      const chosen = original[Math.floor(original.length / 2)]
      assert.ok(chosen)
      const changed = candles.map((candle) => candle.time <= chosen.originTime ? { ...candle } : {
        ...candle, open: candle.open * 2, high: candle.high * 2, low: candle.low * 2,
        close: candle.close * 2, volume: candle.volume * 3,
      })
      const replay = replayB3(changed, step, bundle)
      const sameOrigin = replay.find((point) => point.originTime === chosen.originTime)
      assert.ok(sameOrigin)
      assert.equal(sameOrigin.predicted, chosen.predicted)
      assert.equal(sameOrigin.actual, chosen.actual * 2)
    })

    test('one newly closed candle does not shift the origins or repaint existing replay points', () => {
      const original = replayB3(candles, step, bundle)
      const extended = replayB3(syntheticCandles(bundle, candles.length + 1), step, bundle)
      for (const point of original) {
        const sameOrigin = extended.find((candidate) => candidate.originTime === point.originTime)
        assert.deepEqual(sameOrigin, point)
      }
    })

    test('training-era origins and a not-yet-observed next candle never appear in replay', () => {
      const trainingEra = candles.filter((candle) => candle.time * 1000 <= Date.parse(bundle.dataCutoff))
      assert.deepEqual(replayB3(trainingEra, step, bundle), [])
      const shortFuture = candles.filter((candle) => candle.time * 1000 <= Date.parse(bundle.dataCutoff) + step)
      assert.deepEqual(replayB3(shortFuture, step, bundle), [])
      const replay = replayB3(candles.slice(0, -1), step, bundle)
      assert.ok(replay.every((point) => point.time < candles.at(-1)!.time))
    })

    test('the open target candle is excluded before dense history is evaluated', () => {
      const withOpen = [...candles.slice(0, -1), { ...candles.at(-1)!, isClosed: false }]
      const onlyClosed = closedCandles(withOpen, step, withOpen.at(-1)!.time * 1000 + step * 2)
      const replay = replayB3(onlyClosed, step, bundle)
      assert.equal(replay.at(-1)?.time, candles.at(-2)?.time)
      assert.ok(replay.every((point) => point.time < candles.at(-1)!.time))
    })
  })

  describe(`${interval} frozen training partitions`, () => {
    const bundle = readJson(`../public/models/b3-${interval}.json`)
    const step = INTERVAL_MS[interval]
    const splitNames = ['train', 'validation', 'calibration', 'test'] as const
    const splits = bundle.splits as Record<(typeof splitNames)[number], Split> & {
      embargoBars: number
      boundaries: string[]
    }

    test('every target closes before the next partition origin', () => {
      assert.equal(splits.embargoBars, 12)
      for (let i = 0; i < splitNames.length - 1; i += 1) {
        const left = splits[splitNames[i]]
        const right = splits[splitNames[i + 1]]
        assert.ok(left.samples > 100 && right.samples > 100)
        assert.equal(Date.parse(left.lastTarget), Date.parse(left.lastOrigin) + 12 * step)
        assert.ok(Date.parse(left.lastTarget) + step <= Date.parse(right.firstOrigin))
        assert.equal(Date.parse(right.firstOrigin), Date.parse(splits.boundaries[i]))
      }
    })

    test('history cutoff includes all calibration labels and excludes the test partition', () => {
      assert.equal(Date.parse(bundle.dataCutoff), Date.parse(splits.calibration.lastTarget) + step)
      assert.ok(Date.parse(bundle.dataCutoff) <= Date.parse(splits.test.firstOrigin))
      assert.equal(Date.parse(splits.test.lastTarget) + step, Date.parse(bundle.dataEnd))
      assert.equal(bundle.metrics.testSamples, splits.test.samples)
      assert.equal(bundle.trainingFeatureRowsOnly, true)
    })

    test('exported metrics agree with the independently stored training report', () => {
      const report = readJson('../reports/b3-training-report.json').intervals.find(
        (item: { interval: string }) => item.interval === interval,
      )
      assert.ok(report, `missing ${interval} report`)
      assert.deepEqual(bundle.metrics, report.metrics)
      assert.deepEqual(bundle.splits, report.splits)
      assert.equal(bundle.blend, report.blend)
      assert.deepEqual(bundle.blendByHorizon, report.blendByHorizon)
      assert.equal(bundle.blendByHorizon.length, 12)
      assert.equal(bundle.blend, bundle.blendByHorizon[0])
      assert.ok(report.nonoverlappingOriginMetrics.testSamples < bundle.metrics.testSamples)
      assert.equal(report.metrics.byHorizon.length, 12)
    })
  })
}

describe('model schema validation', () => {
  const valid = readJson('../public/models/b3-15m.json')

  test('the requested interval must match the loaded model', () => {
    assert.throws(() => validateB3Bundle(valid, '1h'))
    assert.equal(validateB3Bundle(valid, '15m').interval, '15m')
  })

  test('unsupported shapes, non-finite parameters and incompatible feature order are rejected', () => {
    const mutations: Array<[string, (bundle: typeof valid) => void]> = [
      ['schema version', (bundle) => { bundle.schemaVersion = 999 }],
      ['horizon', (bundle) => { bundle.horizon = 1 }],
      ['sequence', (bundle) => { bundle.sequence = 1 }],
      ['zero scale', (bundle) => { bundle.scale[0] = 0 }],
      ['negative target scale', (bundle) => { bundle.targetScale[0] = -1 }],
      ['non-finite mean', (bundle) => { bundle.mean[0] = NaN }],
      ['convolution shape', (bundle) => { bundle.weights.convWeight[0][0].pop() }],
      ['LSTM shape', (bundle) => { bundle.weights.lstmWeightHH.pop() }],
      ['non-finite output parameter', (bundle) => { bundle.weights.outputWeight[0][0] = Infinity }],
      ['feature order', (bundle) => { [bundle.features[0], bundle.features[1]] = [bundle.features[1], bundle.features[0]] }],
      ['out-of-range blend', (bundle) => { bundle.blend = 2 }],
      ['missing per-horizon blends', (bundle) => { delete bundle.blendByHorizon }],
      ['per-horizon blend shape', (bundle) => { bundle.blendByHorizon.pop() }],
      ['out-of-range per-horizon blend', (bundle) => { bundle.blendByHorizon[1] = 2 }],
      ['non-finite per-horizon blend', (bundle) => { bundle.blendByHorizon[1] = NaN }],
      ['inconsistent next-bar blend alias', (bundle) => { bundle.blend = bundle.blendByHorizon[0] === 0 ? 1 : 0 }],
      ['invalid cutoff', (bundle) => { bundle.dataCutoff = 'not-a-date' }],
      ['reversed calibration bounds', (bundle) => { bundle.residualLower[0] = bundle.residualUpper[0] + 1 }],
    ]
    for (const [label, mutate] of mutations) {
      const damaged = structuredClone(valid)
      mutate(damaged)
      assert.throws(() => validateB3Bundle(damaged), label)
    }
    for (const malformed of [null, [], {}, 'model']) assert.throws(() => validateB3Bundle(malformed))
  })
})

type WorkerRequest = { id: number; interval: Interval; intervalMs: number; symbol: string; candles: Candle[] }
type WorkerResponse = { id: number; status: string; message?: string; model: b3.B3Forecast | null; history: b3.ReplayPoint[] }

function workerHarness(fetchMock: typeof fetch) {
  const messages: WorkerResponse[] = []
  const worker = {
    onmessage: null as null | ((event: { data: WorkerRequest }) => Promise<void>),
    postMessage(message: WorkerResponse) { messages.push(message) },
  }
  // Vite supplies BASE_URL in production. Inject only that build-time value;
  // execute the actual worker source and actual B3 inference in an isolated VM.
  const source = readFileSync(new URL('../src/workers/b3.worker.ts', import.meta.url), 'utf8')
    .replaceAll('import.meta.env.BASE_URL', JSON.stringify('/test-base/'))
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023 } })
  runInNewContext(compiled.outputText, {
    exports: {}, self: worker, fetch: fetchMock,
    require(name: string) {
      assert.equal(name, '../lib/b3')
      return b3
    },
  })
  assert.ok(worker.onmessage)
  return { messages, send: (data: WorkerRequest) => worker.onmessage!({ data }) }
}

function workerRequest(interval: Interval, id: number, symbol = 'BTCUSDT'): WorkerRequest {
  const fixture = parity.cases.find((item) => item.interval === interval)!
  return { id, interval, intervalMs: INTERVAL_MS[interval], symbol, candles: fixture.candles }
}

describe('worker integration', () => {
  test('loads from the configured base URL, caches validated weights and returns real inference', async () => {
    let requests = 0
    const { messages, send } = workerHarness(async (url) => {
      requests += 1
      assert.equal(url, '/test-base/models/b3-15m.json')
      return new Response(JSON.stringify(readJson('../public/models/b3-15m.json')))
    })
    await send(workerRequest('15m', 101))
    await send(workerRequest('15m', 102, 'ETHUSDT'))
    assert.equal(requests, 1)
    assert.deepEqual(messages.map(({ id, status }) => ({ id, status })), [{ id: 101, status: 'ready' }, { id: 102, status: 'ready' }])
    assert.equal(messages[0].model?.path.length, 13)
    const expected = parity.cases.find((item) => item.interval === '15m')!
    near(messages[0].model!.path[12].mid, expected.forecastPrices[11], expected.forecastPrices[11] * 1e-6, 'worker final price')
  })

  test('unsupported symbols cannot inherit a trained symbol model or its evaluation metrics', async () => {
    const { messages, send } = workerHarness(async () => new Response(JSON.stringify(readJson('../public/models/b3-1h.json'))))
    await send(workerRequest('1h', 201, 'XRPUSDT'))
    await send(workerRequest('1h', 202, 'BTCUSDC'))
    for (const message of messages) {
      assert.equal(message.status, 'unsupported')
      assert.equal(message.model, null)
      assert.equal(message.history.length, 0)
    }
  })

  test('a failed or invalid model load does not poison the cache and can be retried', async () => {
    let requests = 0
    const { messages, send } = workerHarness(async () => {
      requests += 1
      if (requests === 1) return new Response('not found', { status: 404 })
      if (requests === 2) return new Response(JSON.stringify({ schemaVersion: 999 }))
      return new Response(JSON.stringify(readJson('../public/models/b3-1h.json')))
    })
    await send(workerRequest('1h', 301))
    await send(workerRequest('1h', 302))
    await send(workerRequest('1h', 303))
    assert.equal(requests, 3)
    assert.deepEqual(messages.map((message) => message.status), ['error', 'error', 'ready'])
    assert.equal(messages[0].model, null)
    assert.equal(messages[1].model, null)
    assert.ok(messages[2].model)
  })

  test('out-of-order model downloads preserve request IDs and timeframe ownership', async () => {
    let release: (response: Response) => void = () => { throw new Error('15m request has not started') }
    const delayed = new Promise<Response>((resolve) => { release = resolve })
    const { messages, send } = workerHarness(async (url) => String(url).includes('15m')
      ? delayed : new Response(JSON.stringify(readJson('../public/models/b3-1h.json'))))
    const oldRequest = send(workerRequest('15m', 401))
    await send(workerRequest('1h', 402))
    assert.equal(messages.length, 1)
    assert.equal(messages[0].id, 402)
    assert.equal(messages[0].model?.bundle.interval, '1h')
    release(new Response(JSON.stringify(readJson('../public/models/b3-15m.json'))))
    await oldRequest
    assert.equal(messages[1].id, 401)
    assert.equal(messages[1].model?.bundle.interval, '15m')
  })
})

describe('market candle finality', () => {
  test('a REST response crossing a candle boundary cannot promote its pre-close snapshot', async (t) => {
    const start = Date.UTC(2026, 8, 1, 12)
    const step = INTERVAL_MS['1h']
    let clock = start + step - 10
    t.mock.method(Date, 'now', () => clock)
    t.mock.method(globalThis, 'fetch', async () => {
      clock = start + step + 10
      return new Response(JSON.stringify([
        [start - step, '100', '101', '99', '100.5', '20', start - 1],
        [start, '100.5', '101', '99', '100.8', '15', start + step - 1],
      ]))
    })
    const candles = await fetchKlines('BTCUSDT', '1h')
    assert.equal(candles[0].isClosed, true)
    assert.equal(candles[1].isClosed, false)
    assert.equal(closedCandles(candles, step, clock).length, 1)
  })

  test('a late final candle can repair an existing historical bar without losing the newer bar', () => {
    const original: Candle = { time: 1800, open: 100, high: 101, low: 99, close: 100.5, volume: 10, isClosed: false }
    const next = { ...original, time: 2700 }
    const final = { ...original, high: 102, close: 101.5, volume: 15, isClosed: true }
    const merged = mergeCandle([original, next], final)
    assert.deepEqual(merged, [final, next])
    assert.equal(original.isClosed, false, 'merge must not mutate the previous React state')
  })

  test('a late non-final snapshot cannot overwrite a confirmed final candle', () => {
    const final: Candle = { time: 1800, open: 100, high: 102, low: 99, close: 101.5, volume: 15, isClosed: true }
    const stale = { ...final, close: 100.5, volume: 10, isClosed: false }
    assert.deepEqual(mergeCandle([final], stale), [final])
  })

  test('a buffered final frame survives the next bar arriving within the same flush interval', () => {
    const partial: Candle = { time: 1800, open: 100, high: 101, low: 99, close: 100.5, volume: 10, isClosed: false }
    const final = { ...partial, high: 102, close: 101.5, volume: 15, isClosed: true }
    const next = { ...partial, time: 2700 }
    const buffer = new Map<number, Candle>()
    bufferCandle(buffer, final)
    bufferCandle(buffer, next)
    bufferCandle(buffer, partial)
    assert.equal(buffer.size, 2)
    const merged = [...buffer.values()].sort((a, b) => a.time - b.time).reduce(mergeCandle, [partial])
    assert.deepEqual(merged, [final, next])
  })

  test('backfill detects a missed final after its grace period without treating a current candle as missing', () => {
    const candle: Candle = { time: 1800, open: 100, high: 101, low: 99, close: 100.5, volume: 10, isClosed: false }
    const step = INTERVAL_MS['15m']
    const scheduledClose = candle.time * 1000 + step
    assert.equal(candlesNeedBackfill([candle], step, scheduledClose - 1), false)
    assert.equal(candlesNeedBackfill([candle], step, scheduledClose + 1499), false)
    assert.equal(candlesNeedBackfill([candle], step, scheduledClose + 1500), true)
    assert.equal(candlesNeedBackfill([{ ...candle, isClosed: true }], step, scheduledClose + 1500), false)
  })

  test('backfill detects a missing bar even when the known surrounding bars are final', () => {
    const candle: Candle = { time: 1800, open: 100, high: 101, low: 99, close: 100.5, volume: 10, isClosed: true }
    const step = INTERVAL_MS['15m']
    assert.equal(candlesNeedBackfill([candle, { ...candle, time: 2700 }], step, 4500000), false)
    assert.equal(candlesNeedBackfill([candle, { ...candle, time: 3600 }], step, 4500000), true)
    assert.equal(candlesNeedBackfill([], step, 4500000), false)
  })
})

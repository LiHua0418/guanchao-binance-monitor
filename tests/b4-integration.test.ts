import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { describe, test } from 'node:test'
import { runInNewContext } from 'node:vm'
import { createElement, type ComponentType } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import { closedCandles } from '../src/lib/b3.ts'
import * as b4 from '../src/lib/b4.ts'
import * as format from '../src/lib/format.ts'
import * as replayAudit from '../src/lib/replay-audit.ts'
import { INTERVAL_LABEL, INTERVAL_MS, INTERVALS, type Candle, type Interval } from '../src/types.ts'

const require = createRequire(import.meta.url)
const start = Date.UTC(2026, 9, 8) / 1000

function candles(interval: Interval = '1h', count = 100): Candle[] {
  return Array.from({ length: count + 1 }, (_, i) => {
    const close = 100 + i / 10
    return { time: start + i * INTERVAL_MS[interval] / 1000, open: close - .02, high: close + .1,
      low: close - .1, close, volume: 100, quoteVolume: 100 * close, trades: 20,
      takerBuyVolume: 40, takerBuyQuoteVolume: 40 * close, isClosed: i < count }
  })
}

function fixture(interval: Interval = '1h', symbol = 'BTCUSDT', origin = candles(interval).at(-2)!): b4.B4Forecast {
  const step = INTERVAL_MS[interval] / 1000
  const history = [1, 2].map((i) => {
    const baseline = origin.close - 3 + i
    const actual = baseline + 1
    return { time: origin.time + (i - 2) * step, originTime: origin.time + (i - 3) * step,
      predicted: baseline + .01, actual, baseline, expectedMovePct: .4, upProbability: .6,
      actualMovePct: Math.expm1(Math.abs(Math.log(actual / baseline))) * 100 }
  })
  return { modelId: 'guanchao-b4', symbol, interval, originTime: origin.time, originPrice: origin.close,
    path: Array.from({ length: 13 }, (_, i) => {
      const expectedMovePct = .2 * Math.sqrt(i)
      return { time: origin.time + i * step, mid: origin.close + i * .001,
        lower: origin.close * (1 - .002 * Math.sqrt(i)), upper: origin.close * (1 + .002 * Math.sqrt(i)),
        upScenario: origin.close * (1 + expectedMovePct / 100), downScenario: origin.close / (1 + expectedMovePct / 100),
        expectedMovePct, ...([1, 3, 6, 12].includes(i) ? { upProbability: .55 + i / 1000 } : {}) }
    }), history,
    evaluation: { direction: { balancedAccuracy: .528, baselineBalancedAccuracy: .524,
      brier: .249, baselineBrier: .25, cvPassed: true },
    magnitude: { mseImprovement: .0974, maeImprovement: .0345, cvPassed: true, heldoutSkill: true },
    samples: 11888, dataCutoff: new Date((origin.time - 4 * step) * 1000).toISOString(),
    dataEnd: new Date((origin.time + 10 * step) * 1000).toISOString() } }
}

describe('B4 response identity, time and numeric contracts', () => {
  for (const interval of INTERVALS) test(`${interval} validates one independent forecast node per actual bar`, () => {
    const origin = candles(interval).at(-2)!
    const model = fixture(interval)
    assert.equal(b4.validateB4Forecast(model, 'BTCUSDT', interval, origin), model)
    assert.equal(model.path[12].time - origin.time, INTERVAL_MS[interval] * 12 / 1000)
    assert.deepEqual(model.path.flatMap((p, i) => p.upProbability === undefined ? [] : [i]), [1, 3, 6, 12])
    const wrongStep = structuredClone(model)
    wrongStep.path[1].time += INTERVAL_MS[interval] / 1000
    assert.throws(() => b4.validateB4Forecast(wrongStep, 'BTCUSDT', interval, origin))
  })

  test('a valid response cannot be reused for another symbol, period or corrected origin', () => {
    const model = fixture()
    const origin = candles().at(-2)!
    for (const [symbol, interval, anchor] of [
      ['ETHUSDT', '1h', origin], ['BTCUSDT', '4h', origin],
      ['BTCUSDT', '1h', { ...origin, time: origin.time + 3600 }],
      ['BTCUSDT', '1h', { ...origin, close: origin.close + .001 }],
    ] as [string, Interval, Candle][]) assert.throws(() => b4.validateB4Forecast(model, symbol, interval, anchor))
  })

  test('malformed paths, fabricated probabilities, inflated scenarios and truthy validation strings fail', () => {
    const mutations: Array<(model: b4.B4Forecast) => void> = [
      (m) => { m.path.pop() },
      (m) => { m.path[0].mid += 1 },
      (m) => { m.path[0].expectedMovePct = .1 },
      (m) => { delete m.path[1].upProbability },
      (m) => { m.path[2].upProbability = .56 },
      (m) => { m.path[1].upProbability = 1.1 },
      (m) => { m.path[1].expectedMovePct = NaN },
      (m) => { m.path[1].mid = Infinity },
      (m) => { m.path[1].downScenario = 0 },
      (m) => { m.path[1].upScenario *= 2 },
      (m) => { m.path[1].lower = m.path[1].upper + 1 },
      (m) => { m.evaluation.direction.cvPassed = 'false' as unknown as boolean },
      (m) => { m.evaluation.magnitude.heldoutSkill = 'true' as unknown as boolean },
      (m) => { m.evaluation.magnitude.mseImprovement = 1.1 },
      (m) => { m.evaluation.samples = .5 },
      (m) => { m.evaluation.direction.brier = NaN },
      (m) => { m.evaluation.dataCutoff = 'invalid date' },
      (m) => { m.evaluation.dataEnd = new Date(start * 1000).toISOString() },
      (m) => { m.evaluation.dataCutoff = m.evaluation.dataEnd },
    ]
    for (const mutate of mutations) {
      const model = fixture()
      mutate(model)
      assert.throws(() => b4.validateB4Forecast(model, 'BTCUSDT', '1h', candles().at(-2)!))
    }
    for (const value of [null, undefined, {}, 1, 'model', []]) {
      assert.throws(() => b4.validateB4Forecast(value, 'BTCUSDT', '1h', candles().at(-2)!))
    }
  })

  test('a skewed calibrated interval need not contain the conditional mean price', () => {
    const model = fixture()
    model.path[1].lower = model.path[1].mid + 1
    model.path[1].upper = model.path[1].mid + 2
    assert.equal(b4.validateB4Forecast(model, 'BTCUSDT', '1h', candles().at(-2)!), model)
  })

  test('historical replay contains contiguous next-bar predictions only after the freeze', () => {
    const mutations: Array<(model: b4.B4Forecast) => void> = [
      (m) => { m.history[0].time = NaN },
      (m) => { m.history[0].originTime = NaN },
      (m) => { m.history[0].originTime -= 11 * 3600 },
      (m) => { m.history[0].originTime = Date.parse(m.evaluation.dataCutoff) / 1000 - 3600; m.history[0].time = m.history[0].originTime + 3600 },
      (m) => { m.history[1].time += 3600; m.history[1].originTime += 3600 },
      (m) => { m.history.reverse() },
      (m) => { m.history[1].baseline += .1 },
      (m) => { m.history[0].actualMovePct += 1 },
      (m) => { m.history[0].upProbability = -.1 },
      (m) => { m.history[1].actual += 1 },
    ]
    for (const mutate of mutations) {
      const model = fixture()
      mutate(model)
      assert.throws(() => b4.validateB4Forecast(model, 'BTCUSDT', '1h', candles().at(-2)!))
    }
    const empty = fixture()
    empty.history = []
    assert.equal(b4.validateB4Forecast(empty, 'BTCUSDT', '1h', candles().at(-2)!), empty)
  })
})

type HookResult = { key: string; status: b4.B4Status; model: b4.B4Forecast | null; retry: () => void; message?: string }
type HookArguments = [Candle[], Interval, string, number, boolean]
type EffectSlot = { dependencies: unknown[]; cleanup?: () => void }

// Execute the actual hook with a deterministic React hook scheduler and
// deferred fetches. We intentionally let aborted requests resolve to verify
// the hook's stale-response protection independently of fetch cancellation.
function hookRunner() {
  let cursor = 0
  const slots: unknown[] = []
  let pending: Array<() => void> = []
  const requests: Array<{ body: { symbol: string; interval: Interval; candles: Candle[] }; signal: AbortSignal;
    resolve: (response: { ok: boolean; json: () => Promise<unknown> }) => void }> = []
  const react = {
    useState(initial: unknown) {
      const i = cursor++
      if (!(i in slots)) slots[i] = initial
      return [slots[i], (value: unknown) => { slots[i] = typeof value === 'function' ? value(slots[i]) : value }]
    },
    useRef(initial: unknown) {
      const i = cursor++
      if (!(i in slots)) slots[i] = { current: initial }
      return slots[i]
    },
    useEffect(effect: () => void | (() => void), dependencies: unknown[]) {
      const i = cursor++
      const previous = slots[i] as EffectSlot | undefined
      if (!previous || dependencies.some((value, j) => !Object.is(value, previous.dependencies[j]))) {
        pending.push(() => {
          previous?.cleanup?.()
          slots[i] = { dependencies, cleanup: effect() }
        })
      }
    },
  }
  const source = readFileSync(new URL('../src/hooks/useB4Forecast.ts', import.meta.url), 'utf8')
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023 } })
  const exports: { useB4Forecast?: (...args: HookArguments) => HookResult } = {}
  runInNewContext(compiled.outputText, { exports, AbortController,
    window: { setTimeout: () => 1, clearTimeout() {} },
    fetch(_url: string, options: { body: string; signal: AbortSignal }) {
      return new Promise((resolve) => requests.push({ body: JSON.parse(options.body), signal: options.signal, resolve }))
    },
    require(name: string) {
      if (name === 'react') return react
      if (name === '../lib/b3') return { closedCandles }
      if (name === '../lib/b4') return b4
      if (name === '../types') return { INTERVAL_MS }
      throw new Error(`Unexpected hook import ${name}`)
    },
  })
  return {
    requests,
    render(...args: HookArguments) {
      cursor = 0
      pending = []
      const value = exports.useB4Forecast!(...args)
      for (const effect of pending) effect()
      return value
    },
    dispose() { for (const value of slots) (value as EffectSlot)?.cleanup?.() },
  }
}

const settle = () => new Promise<void>((resolve) => setImmediate(resolve))

describe('B4 closed-candle request identity and async lifecycle', () => {
  test('live open-bar ticks do not request/repaint; a correction to any closed source field invalidates the key', () => {
    const source = candles()
    const now = source.at(-1)!.time * 1000 + 1000
    const key = (rows: Candle[]) => b4.b4RequestKey('BTCUSDT', '1h', closedCandles(rows, INTERVAL_MS['1h'], now))
    const changedOpen = structuredClone(source)
    changedOpen.at(-1)!.close *= 2
    changedOpen.at(-1)!.isClosed = true // Time still proves this candle is open.
    assert.equal(key(changedOpen), key(source))
    for (const field of ['open', 'high', 'low', 'close', 'volume', 'quoteVolume', 'trades', 'takerBuyVolume', 'takerBuyQuoteVolume'] as const) {
      const changed = structuredClone(source)
      changed[5][field] = changed[5][field]! + .01
      assert.notEqual(key(changed), key(source), `${field} corrections must invalidate the model`)
    }
    const runner = hookRunner()
    try {
      runner.render(source, '1h', 'BTCUSDT', now, true)
      runner.render(changedOpen, '1h', 'BTCUSDT', now, true)
      assert.equal(runner.requests.length, 1)
      assert.equal(runner.requests[0].body.candles.length, 100)
      assert.ok(runner.requests[0].body.candles.every((c) => c.isClosed !== false))
    } finally { runner.dispose() }
  })

  test('switching symbol hides the old model immediately and ignores its delayed response', async () => {
    const runner = hookRunner()
    const source = candles()
    const now = source.at(-1)!.time * 1000 + 1000
    try {
      assert.equal(runner.render(source, '1h', 'BTCUSDT', now, true).status, 'loading')
      const old = runner.requests[0]
      assert.equal(runner.render(source, '1h', 'ETHUSDT', now, true).model, null)
      assert.equal(old.signal.aborted, true)
      old.resolve({ ok: true, json: async () => fixture('1h', 'BTCUSDT') })
      await settle()
      assert.equal(runner.render(source, '1h', 'ETHUSDT', now, true).model, null)
      runner.requests[1].resolve({ ok: true, json: async () => fixture('1h', 'ETHUSDT') })
      await settle()
      const ready = runner.render(source, '1h', 'ETHUSDT', now, true)
      assert.equal(ready.status, 'ready')
      assert.equal(ready.model?.symbol, 'ETHUSDT')
      const changed = structuredClone(source)
      changed[2].trades! += 1
      assert.equal(runner.render(changed, '1h', 'ETHUSDT', now, true).model, null)
      assert.equal(runner.requests.length, 3)
    } finally { runner.dispose() }
  })

  test('a wrong-origin response becomes an error, while retry creates a fresh request', async () => {
    const runner = hookRunner()
    const source = candles()
    const now = source.at(-1)!.time * 1000 + 1000
    try {
      runner.render(source, '1h', 'BTCUSDT', now, true)
      const stale = fixture()
      stale.originTime -= 3600
      runner.requests[0].resolve({ ok: true, json: async () => stale })
      await settle()
      const error = runner.render(source, '1h', 'BTCUSDT', now, true)
      assert.equal(error.status, 'error')
      assert.equal(error.model, null)
      error.retry()
      assert.equal(runner.render(source, '1h', 'BTCUSDT', now, true).status, 'loading')
      assert.equal(runner.requests.length, 2)
    } finally { runner.dispose() }
  })

  test('the next closed bar invalidates the prior ready result at the selected interval', async () => {
    const runner = hookRunner()
    const source = candles('15m')
    const now = source.at(-1)!.time * 1000 + 1000
    try {
      runner.render(source, '15m', 'BTCUSDT', now, true)
      runner.requests[0].resolve({ ok: true, json: async () => fixture('15m') })
      await settle()
      assert.equal(runner.render(source, '15m', 'BTCUSDT', now, true).status, 'ready')
      const newlyClosed = structuredClone(source)
      newlyClosed.at(-1)!.isClosed = true
      const nextTime = now + INTERVAL_MS['15m']
      const state = runner.render(newlyClosed, '15m', 'BTCUSDT', nextTime, true)
      assert.equal(state.status, 'loading')
      assert.equal(state.model, null)
      assert.equal(runner.requests.length, 2)
      assert.equal(runner.requests[1].body.candles.length, 101)
      assert.equal(runner.requests[1].body.candles.at(-1)!.time - runner.requests[0].body.candles.at(-1)!.time, 900)
    } finally { runner.dispose() }
  })

  test('disabled, unsupported and short closed histories never issue inference', () => {
    const runner = hookRunner()
    const source = candles()
    const now = source.at(-1)!.time * 1000 + 1000
    try {
      assert.equal(runner.render(source, '1h', 'BTCUSDT', now, false).model, null)
      assert.equal(runner.render(source, '1h', 'XRPUSDT', now, true).status, 'unsupported')
      assert.equal(runner.render(source.slice(-60), '1h', 'BTCUSDT', now, true).status, 'insufficient')
      assert.equal(runner.requests.length, 0)
    } finally { runner.dispose() }
  })
})

type PanelProps = { model: b4.B4Forecast | null; status: b4.B4Status; message?: string; onRetry: () => void; intervalLabel: string }
const panelSource = readFileSync(new URL('../src/components/B4Panel.tsx', import.meta.url), 'utf8')
const panelCompiled = ts.transpileModule(panelSource, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX } })
const panelExports: { B4Panel?: ComponentType<PanelProps> } = {}
const auditSource = readFileSync(new URL('../src/components/ReplayAudit.tsx', import.meta.url), 'utf8')
const auditCompiled = ts.transpileModule(auditSource, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX } })
const auditExports: { ReplayAudit?: ComponentType<{ model: b4.B4Forecast }> } = {}
runInNewContext(auditCompiled.outputText, { exports: auditExports, require(name: string) {
  if (name === 'react/jsx-runtime') return require(name)
  if (name === '../lib/replay-audit') return replayAudit
  if (name === '../types') return { INTERVAL_LABEL, INTERVAL_MS }
  throw new Error(`Unexpected audit import ${name}`)
} })
runInNewContext(panelCompiled.outputText, { exports: panelExports, require(name: string) {
  if (name === 'react/jsx-runtime') return require(name)
  if (name === '../lib/format') return format
  if (name === './ReplayAudit') return auditExports
  throw new Error(`Unexpected panel import ${name}`)
} })
const panel = (model = fixture(), status: b4.B4Status = 'ready') => renderToStaticMarkup(createElement(panelExports.B4Panel!, { model, status, intervalLabel: '1时', onRetry() {} }))

describe('B4 presentation preserves probability and magnitude semantics', () => {
  test('the actual SSR panel distinguishes unsigned magnitude, scenarios, reused history and h1 validation', () => {
    const html = panel()
    assert.match(html, /下一根.*上涨概率/)
    assert.match(html, /下跌概率/)
    assert.match(html, /概率以发生涨跌为条件，训练不计持平样本/)
    assert.doesNotMatch(html, /下跌／持平概率/)
    assert.match(html, /不含涨跌方向/)
    assert.match(html, /上下行情景不是目标价或置信区间/)
    assert.match(html, /优势验证仅针对下一根/)
    assert.match(html, /完整价格路径仍未证明稳定优势/)
    assert.match(html, /同一历史时段已被版本复评/)
    assert.match(html, /冻结模型按历史行情重算/)
  })

  test('twelve independently predicted magnitude steps do not interpolate eight missing probabilities', () => {
    const html = panel()
    const rows = [...html.matchAll(/<tr><td>\+(\d+)<\/td><td[^>]*>([^<]+)<\/td><td[^>]*>([^<]+)<\/td><\/tr>/g)]
    assert.equal(rows.length, 12)
    for (const row of rows) {
      const step = Number(row[1])
      assert.equal(row[2], `${fixture().path[step].expectedMovePct.toFixed(3)}%`)
      assert.equal(row[3] === '—', ![1, 3, 6, 12].includes(step))
    }
    assert.match(html, /其余不插值/)
  })

  test('unvalidated daily direction and stale/loading states cannot inherit a favorable verdict', () => {
    const model = fixture('1d')
    model.evaluation.direction.cvPassed = false
    model.evaluation.magnitude.heldoutSkill = false
    const html = panel(model)
    assert.match(html, /方向模型未通过滚动验证/)
    assert.match(html, /幅度优势尚未通过全部验证/)
    for (const state of ['loading', 'error', 'insufficient', 'unsupported'] as const) {
      const stale = panel(model, state)
      assert.match(stale, /role="status"/)
      assert.doesNotMatch(stale, /b4-horizons|下一根上涨概率|幅度 MSE 改善/)
    }
  })

  test('rolling one-step history cannot substitute for complete fixed-origin 12-step audit samples', () => {
    const model = fixture()
    model.history[0].actual = model.history[0].baseline
    model.history[0].actualMovePct = 0
    model.history[0].upProbability = .9
    const html = panel(model)
    assert.match(html, /0 个共同起点/)
    assert.match(html, /滚动一步回放不会冒充这组完整路径样本/)
    assert.doesNotMatch(html, /<dt>方向命中率<\/dt>/)
  })
})

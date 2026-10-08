import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, test } from 'node:test'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'
import { overlaySeries, type ForecastPoint } from '../src/lib/analysis.ts'
import { ForecastBand } from '../src/lib/forecast-band.ts'
import * as format from '../src/lib/format.ts'
import { HORIZON_LABEL, INTERVAL_MS, type Candle, type Interval } from '../src/types.ts'

type LogicalRange = { from: number; to: number }
type Node = { type: unknown; props: Record<string, unknown> }
type ChartProps = {
  symbol: string; interval: Interval; candles: Candle[]; forecast: ForecastPoint[]
  modelPath: Array<ForecastPoint & { upScenario: number; downScenario: number }>
  modelHistory: Array<{ time: number; originTime: number; predicted: number; actual: number; baseline?: number; horizon?: number }>
  modelHistoryKind?: 'rolling-next' | 'fixed-origin'
  replayFocus?: { originTime: number; requestId: number } | null
  modelLabel: string; modelBandVisible: boolean; modelScenariosVisible: boolean
  showEma: boolean; showBands: boolean; showForecast: boolean; showModel: boolean; digits: number; theme: 'dark' | 'light'
}

function props(interval: Interval = '1h'): ChartProps {
  const start = Date.UTC(2026, 9, 8) / 1000
  const step = INTERVAL_MS[interval] / 1000
  const candles = Array.from({ length: 200 }, (_, i) => ({ time: start + i * step,
    open: 100 + i / 100, high: 102 + i / 100, low: 98 + i / 100, close: 101 + i / 100, volume: 100 + i }))
  const latest = candles.at(-1)!
  const modelPath = Array.from({ length: 13 }, (_, i) => ({ time: latest.time + i * step,
    mid: latest.close + .02 * i, lower: latest.close - i, upper: latest.close + i,
    upScenario: latest.close + 2 * i, downScenario: latest.close - 2 * i }))
  return { symbol: 'BTCUSDT', interval, candles, forecast: modelPath, modelPath,
    modelHistory: candles.slice(-5).map((c) => ({ time: c.time, originTime: c.time - step, predicted: c.close + .01, actual: c.close })),
    modelLabel: 'guanchao-b4', modelBandVisible: true, modelScenariosVisible: true,
    showEma: true, showBands: true, showForecast: true, showModel: true, digits: 2, theme: 'dark' }
}

function fixedReplayProps(interval: Interval = '1h'): ChartProps {
  const p = props(interval)
  const anchorIndex = 100
  const anchor = p.candles[anchorIndex]
  return { ...p, modelHistoryKind: 'fixed-origin', modelHistory: Array.from({ length: 13 }, (_, horizon) => ({
    time: p.candles[anchorIndex + horizon].time,
    originTime: anchor.time,
    horizon,
    predicted: anchor.close + horizon * .03,
    actual: p.candles[anchorIndex + horizon].close,
    baseline: anchor.close,
  })) }
}

function nodes(value: unknown): Node[] {
  if (Array.isArray(value)) return value.flatMap(nodes)
  if (!value || typeof value !== 'object' || !('props' in value)) return []
  const node = value as Node
  return [node, ...nodes(node.props.children)]
}

function text(value: unknown): string {
  if (Array.isArray(value)) return value.map(text).join('')
  if (value === null || value === undefined || typeof value === 'boolean') return ''
  if (typeof value !== 'object') return String(value)
  return 'props' in value ? text((value as Node).props.children) : ''
}

// Compile and execute the real component. The host models chart commands and
// captured input events; React state/refs/effects persist across each render.
// No assertions depend on specific lines or substrings of the component source.
function mount(initial = props()) {
  type Listener = { name: string; callback: (event: { clientX: number; clientY: number }) => void; capture: boolean; passive: boolean }
  const listeners: Listener[] = []
  const eventOrder: string[] = []
  const host = {
    getBoundingClientRect: () => ({ left: 20, top: 10, width: 900, height: 500 }),
    addEventListener(name: string, callback: Listener['callback'], options: boolean | { capture?: boolean; passive?: boolean } = false) {
      listeners.push({ name, callback, capture: typeof options === 'boolean' ? options : !!options.capture,
        passive: typeof options === 'boolean' ? false : !!options.passive })
    },
    removeEventListener(name: string, callback: Listener['callback'], options: boolean | { capture?: boolean } = false) {
      const capture = typeof options === 'boolean' ? options : !!options.capture
      const index = listeners.findIndex((l) => l.name === name && l.callback === callback && l.capture === capture)
      assert.notEqual(index, -1, `cleanup must remove the exact ${name} listener`)
      listeners.splice(index, 1)
    },
    emit(name: string, event = { clientX: 100, clientY: 100 }) {
      for (const listener of listeners.filter((l) => l.name === name && l.capture)) listener.callback(event)
      eventOrder.push(`chart ${name}`)
      for (const listener of listeners.filter((l) => l.name === name && !l.capture)) listener.callback(event)
    },
  }
  let range: LogicalRange | null = { from: 0, to: 146 }
  let priceRange: { from: number; to: number } | null = { from: 90, to: 115 }
  let autoScale = true
  const rangeWrites: LogicalRange[] = []
  const autoScaleWrites: boolean[] = []
  const chartOptions: Record<string, unknown>[] = []
  let beforeSeriesWrite: (() => void) | undefined
  let removed = false
  let subscribed: ((value: unknown) => void) | undefined
  let subscriptionCount = 0
  let unsubscriptionCount = 0
  const price = {
    width: () => 88,
    getVisibleRange: () => priceRange && { ...priceRange },
    setAutoScale(value: boolean) { autoScale = value; autoScaleWrites.push(value); eventOrder.push(`autoScale ${value}`) },
    applyOptions(_options: unknown) {},
  }
  const time = {
    height: () => 30,
    getVisibleLogicalRange: () => range && { ...range },
    setVisibleLogicalRange(value: LogicalRange) { range = { ...value }; rangeWrites.push({ ...value }) },
    timeToCoordinate: (timestamp: number) => timestamp,
  }
  const series: Array<{
    kind: string; options: Record<string, unknown>; data: unknown[]; sets: number; updates: number;
    primitives: ForecastBand[]; applyOptions: (value: Record<string, unknown>) => void;
    setData: (value: unknown[]) => void; update: (value: unknown, historical?: boolean) => void;
    attachPrimitive: (primitive: ForecastBand) => void; detachPrimitive: (primitive: ForecastBand) => void;
  }> = []
  const chart = {
    addSeries(kind: string, options: Record<string, unknown>) {
      const entry = { kind, options: { ...options }, data: [] as unknown[], sets: 0, updates: 0, primitives: [] as ForecastBand[],
        applyOptions(value: Record<string, unknown>) { Object.assign(entry.options, value) },
        setData(value: unknown[]) { entry.sets++; entry.data = [...value]; beforeSeriesWrite?.() },
        update(value: unknown, _historical?: boolean) {
          entry.updates++
          beforeSeriesWrite?.()
          const timestamp = (value as { time: number }).time
          const index = entry.data.findIndex((point) => (point as { time: number }).time === timestamp)
          if (index < 0) entry.data.push(value)
          else entry.data[index] = value
        },
        attachPrimitive(value: ForecastBand) { entry.primitives.push(value) },
        detachPrimitive(value: ForecastBand) { entry.primitives.splice(entry.primitives.indexOf(value), 1); value.detached() },
      }
      series.push(entry)
      return entry
    },
    timeScale: () => time,
    priceScale: (_name: string) => price,
    applyOptions(options: Record<string, unknown>) { chartOptions.push(options) },
    subscribeCrosshairMove(listener: (value: unknown) => void) { subscribed = listener; subscriptionCount++ },
    unsubscribeCrosshairMove(listener: (value: unknown) => void) { assert.equal(subscribed, listener); subscribed = undefined; unsubscriptionCount++ },
    remove() { removed = true },
  }
  const marker = { detached: false, calls: [] as unknown[], options: {} as Record<string, unknown>,
    setMarkers(value: unknown) { marker.calls.push(value) }, detach() { marker.detached = true } }
  const slots: unknown[] = []
  let cursor = 0
  let effects: Array<() => void> = []
  const react = {
    useState(initial: unknown) {
      const index = cursor++
      if (!(index in slots)) slots[index] = initial
      return [slots[index], (value: unknown) => { slots[index] = typeof value === 'function' ? value(slots[index]) : value }]
    },
    useRef(initial: unknown) {
      const index = cursor++
      if (!(index in slots)) slots[index] = { current: initial }
      return slots[index]
    },
    useEffect(effect: () => void | (() => void), dependencies: unknown[]) {
      const index = cursor++
      const previous = slots[index] as { dependencies: unknown[]; cleanup?: () => void } | undefined
      if (!previous || dependencies.some((value, i) => !Object.is(value, previous.dependencies[i]))) effects.push(() => {
        previous?.cleanup?.()
        slots[index] = { dependencies, cleanup: effect() }
      })
    },
  }
  const exports: { ChartPane?: (value: ChartProps) => Node } = {}
  const source = readFileSync(new URL('../src/components/ChartPane.tsx', import.meta.url), 'utf8')
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX } })
  let creations = 0
  runInNewContext(compiled.outputText, { exports, document: { documentElement: { dataset: { theme: initial.theme } } },
    require(name: string) {
      if (name === 'react') return react
      if (name === 'react/jsx-runtime') return { Fragment: 'Fragment', jsx: (type: unknown, nodeProps: Record<string, unknown>) => ({ type, props: nodeProps }),
        jsxs: (type: unknown, nodeProps: Record<string, unknown>) => ({ type, props: nodeProps }) }
      if (name === 'lightweight-charts') return {
        CandlestickSeries: 'candle', HistogramSeries: 'volume', LineSeries: 'line', ColorType: { Solid: 'solid' },
        CrosshairMode: { Normal: 0 }, LineStyle: { Dashed: 1, Dotted: 2, Solid: 0 }, TickMarkType: { Year: 0, Month: 1, DayOfMonth: 2 },
        createChart(node: unknown, options: Record<string, unknown>) { assert.equal(node, host); chartOptions.push(options); creations++; return chart },
        createSeriesMarkers(_series: unknown, _markers: unknown, options: Record<string, unknown>) { marker.options = options; return marker },
      }
      if (name === '../lib/analysis') return { overlaySeries }
      if (name === '../lib/forecast-band') return { ForecastBand }
      if (name === '../lib/format') return format
      if (name === '../types') return { HORIZON_LABEL, INTERVAL_MS }
      throw new Error(`Unexpected ChartPane runtime dependency: ${name}`)
    },
  })
  let current = initial
  let output: Node
  function render(next = current) {
    current = next
    cursor = 0
    effects = []
    output = exports.ChartPane!(current)
    for (const node of nodes(output)) if (node.props.ref && node.props.className === 'chart-canvas') {
      (node.props.ref as { current: unknown }).current = host
    }
    for (const effect of effects) effect()
    return output
  }
  render()
  return {
    host, listeners, eventOrder, series, marker, rangeWrites, autoScaleWrites, chartOptions,
    render,
    get range() { return range }, get autoScale() { return autoScale }, get created() { return creations },
    get removed() { return removed }, get subscriptionCounts() { return [subscriptionCount, unsubscriptionCount] },
    dragTo(value: LogicalRange) { range = { ...value } },
    noPriceRange() { priceRange = null },
    duringWrite(callback: () => void) { beforeSeriesWrite = callback },
    hover(time: number) { subscribed?.({ time }) },
    click(label: string) {
      const button = nodes(output).find((n) => n.type === 'button' && text(n.props.children) === label)
      assert.ok(button, `missing ${label} button`)
      ;(button.props.onClick as () => void)()
    },
    resetTrace() { rangeWrites.length = 0; autoScaleWrites.length = 0; eventOrder.length = 0 },
    dispose() { for (const value of slots) (value as { cleanup?: () => void })?.cleanup?.() },
  }
}

describe('ChartPane preserves a user-controlled candle viewport', () => {
  test('reference lines, model scenarios, uncertainty and markers cannot expand price autoscale', () => {
    const view = mount()
    try {
      const overlays = view.series.filter((s) => s.kind === 'line')
      assert.equal(overlays.length, 15)
      for (const line of overlays) {
        const provider = line.options.autoscaleInfoProvider as (() => unknown) | undefined
        assert.equal(typeof provider, 'function')
        assert.equal(provider!(), null)
      }
      assert.equal(view.series[0].options.autoscaleInfoProvider, undefined, 'candles remain the autoscale source')
      assert.equal(view.marker.options.autoScale, false)
      assert.equal(view.series.flatMap((s) => s.primitives).length, 1)
      const primitive = view.series.flatMap((s) => s.primitives)[0]
      assert.equal('autoscaleInfo' in primitive, false, 'the filled uncertainty primitive contributes no price bounds')
    } finally { view.dispose() }
  })

  for (const event of ['wheel', 'pointerdown']) test(`${event} captures price scale before chart interaction and retains it across data/model refreshes`, () => {
    const p = props()
    const view = mount(p)
    try {
      view.resetTrace()
      view.host.emit(event)
      assert.equal(view.autoScale, false)
      assert.deepEqual(view.eventOrder.slice(0, 2), ['autoScale false', `chart ${event}`])
      const listener = view.listeners.find((l) => l.name === event)!
      assert.equal(listener.capture, true)
      if (event === 'wheel') assert.equal(listener.passive, true)
      const updated = { ...p, candles: p.candles.map((c, i) => i === p.candles.length - 1 ? { ...c, close: c.close + .01 } : c) }
      view.render(updated)
      view.render({ ...updated, modelPath: p.modelPath.map((point) => ({ ...point, upper: point.upper * 10 })) })
      view.render({ ...updated, theme: 'light', modelBandVisible: false, modelScenariosVisible: false })
      assert.deepEqual(view.autoScaleWrites, [false])
      assert.equal(view.rangeWrites.length, 0)
      assert.match(text(view.render(updated)), /纵轴已固定/)
      assert.equal(view.created, 1)
    } finally { view.dispose() }
  })

  test('a tick cannot restore the old logical range over a range advanced during drag', () => {
    const p = props()
    const view = mount(p)
    try {
      view.host.emit('pointerdown')
      view.dragTo({ from: 15, to: 90 })
      view.resetTrace()
      let invoked = false
      view.duringWrite(() => {
        if (!invoked) { invoked = true; view.dragTo({ from: 10.25, to: 85.25 }) }
      })
      view.render({ ...p, candles: p.candles.map((c, i) => i === p.candles.length - 1 ? { ...c, close: c.close + .1 } : c) })
      assert.equal(invoked, true)
      assert.deepEqual(view.range, { from: 10.25, to: 85.25 })
      assert.equal(view.rangeWrites.length, 0, 'normal ticks must leave the time viewport to the chart interaction')
      assert.equal(view.autoScale, false)
    } finally { view.dispose() }
  })

  test('model arrival does not overwrite a viewport that the chart updated while replacing its series', () => {
    const p = props()
    const initial = { ...p, modelPath: [] }
    const view = mount(initial)
    try {
      view.host.emit('wheel')
      view.dragTo({ from: 20, to: 70 })
      view.resetTrace()
      view.duringWrite(() => view.dragTo({ from: 22.5, to: 72.5 }))
      view.render(p)
      assert.deepEqual(view.range, { from: 22.5, to: 72.5 })
      assert.equal(view.rangeWrites.length, 0)
      assert.equal(view.autoScaleWrites.length, 0)
    } finally { view.dispose() }
  })

  test('discarding the first stored candle compensates indices without changing viewed timestamps or zoom', () => {
    const p = props()
    const view = mount(p)
    try {
      view.host.emit('pointerdown')
      const before = { from: 50.5, to: 110.5 }
      view.dragTo(before)
      view.resetTrace()
      const last = p.candles.at(-1)!
      const next = [...p.candles.slice(1), { ...last, time: last.time + 3600 }]
      view.render({ ...p, candles: next })
      assert.deepEqual(view.range, { from: 49.5, to: 109.5 })
      assert.equal(view.range!.to - view.range!.from, before.to - before.from)
      assert.equal(next[0].time + view.range!.from * 3600, p.candles[0].time + before.from * 3600)
      assert.equal(view.rangeWrites.length, 1)
      assert.equal(view.autoScale, false)
    } finally { view.dispose() }
  })

  test('switching interval or symbol initializes a fresh view and resumes price autoscale', () => {
    const view = mount()
    try {
      view.host.emit('wheel')
      view.dragTo({ from: -25, to: 20 })
      view.resetTrace()
      const next = props('15m')
      view.render(next)
      assert.deepEqual(view.range, { from: 68, to: 214 })
      assert.deepEqual(view.autoScaleWrites, [true])
      view.host.emit('pointerdown')
      view.resetTrace()
      view.render({ ...next, symbol: 'ETHUSDT' })
      assert.deepEqual(view.range, { from: 68, to: 214 })
      assert.deepEqual(view.autoScaleWrites, [true])
      assert.equal(view.created, 1, 'identity changes reuse the chart without leaking handlers')
    } finally { view.dispose() }
  })

  test('fit price resets only the price scale; focus forecast preserves horizontal zoom width', () => {
    const view = mount()
    try {
      view.host.emit('wheel')
      view.dragTo({ from: 30.5, to: 100.5 })
      view.resetTrace()
      view.click('适应价格')
      assert.equal(view.autoScale, true)
      assert.deepEqual(view.range, { from: 30.5, to: 100.5 })
      assert.equal(view.rangeWrites.length, 0)
      view.host.emit('pointerdown')
      view.click('定位预测')
      assert.deepEqual(view.range, { from: 144, to: 214 })
      assert.equal(view.autoScale, true)
      assert.match(text(view.render()), /纵轴自动/)
    } finally { view.dispose() }
  })

  test('double-click resets price only inside the price axis above the time axis', () => {
    const view = mount()
    try {
      view.host.emit('pointerdown')
      view.resetTrace()
      view.host.emit('dblclick', { clientX: 300, clientY: 200 })
      assert.equal(view.autoScale, false)
      view.host.emit('dblclick', { clientX: 880, clientY: 500 })
      assert.equal(view.autoScale, false)
      view.host.emit('dblclick', { clientX: 880, clientY: 200 })
      assert.equal(view.autoScale, true)
      assert.deepEqual(view.autoScaleWrites, [true])
      assert.equal(view.rangeWrites.length, 0)
    } finally { view.dispose() }
  })

  test('no price range does not create a fake lock and unmount removes all interaction resources', () => {
    const view = mount()
    view.noPriceRange()
    view.resetTrace()
    view.host.emit('wheel')
    assert.equal(view.autoScale, true)
    assert.deepEqual(view.autoScaleWrites, [])
    assert.equal(view.listeners.length, 3)
    view.dispose()
    assert.equal(view.listeners.length, 0)
    assert.deepEqual(view.subscriptionCounts, [1, 1])
    assert.equal(view.marker.detached, true)
    assert.equal(view.series.flatMap((s) => s.primitives).length, 0)
    assert.equal(view.removed, true)
  })
})

describe('ChartPane shows a complete fixed-origin replay without taking over the viewport', () => {
  for (const interval of ['15m', '1h', '4h', '1d'] as const) test(`${interval} draws predicted, actual and flat baseline values at every original horizon`, () => {
    const p = fixedReplayProps(interval)
    const view = mount(p)
    try {
      const expected = (field: 'predicted' | 'actual' | 'baseline') => p.modelHistory.map((point) => ({ time: point.time, value: point[field] }))
      const matches = (value: unknown[]) => view.series.filter((s) => s.kind === 'line' && JSON.stringify(s.data) === JSON.stringify(value))
      const predicted = matches(expected('predicted'))
      const actual = matches(expected('actual'))
      const baseline = matches(expected('baseline'))
      for (const [name, found] of [['predicted', predicted], ['actual', actual], ['baseline', baseline]] as const) {
        assert.equal(found.length, 1, `${name} must be one distinct complete line`)
        assert.equal(found[0].options.visible, true)
        assert.equal(found[0].data.length, 13)
      }
      assert.ok(p.modelHistory.every((point) => point.originTime === p.modelHistory[0].time))
      assert.ok((baseline[0].data as { value: number }[]).every((point) => point.value === p.modelHistory[0].actual),
        'the comparison repeats the same origin price, never the previous actual close')
      assert.equal(p.modelHistory[12].time - p.modelHistory[0].time, 12 * INTERVAL_MS[interval] / 1000)
      assert.match(text(view.render()), /同一起点 · 12 步回放/)
      assert.match(text(view.render()), /实际收盘/)
      assert.match(text(view.render()), /持平基线 · 复制起点/)
      view.render({ ...p, showModel: false })
      for (const line of [predicted[0], actual[0], baseline[0]]) {
        assert.equal(line.options.visible, false)
        assert.equal(line.data.length, 0)
      }
      view.render({ ...props(interval), modelHistoryKind: 'rolling-next' })
      assert.equal(predicted[0].options.visible, true)
      for (const line of [actual[0], baseline[0]]) {
        assert.equal(line.options.visible, false)
        assert.equal(line.data.length, 0)
      }
      assert.match(text(view.render()), /滚动 1 步 · 每根重设起点/)
      assert.doesNotMatch(text(view.render()), /持平基线 · 复制起点/)
    } finally { view.dispose() }
  })

  test('fixed-origin horizon 12 hover identifies the original prediction rather than a rolling next-bar forecast', () => {
    const p = fixedReplayProps()
    const view = mount(p)
    try {
      view.hover(p.modelHistory[12].time)
      const content = text(view.render())
      assert.match(content, /同一起点回放 · \+12 根/)
      assert.match(content, /起点收盘.*目标收盘/)
      assert.match(content, /持平基线/)
      assert.doesNotMatch(content, /上一根收盘时预测本根/)
    } finally { view.dispose() }
  })

  test('an explicit replay focus moves once; new candle and forecast arrays preserve subsequent user pan and zoom', () => {
    const p = fixedReplayProps()
    const view = mount(p)
    const focus = { originTime: p.modelHistory[0].originTime, requestId: 1 }
    try {
      view.resetTrace()
      view.render({ ...p, replayFocus: focus })
      assert.deepEqual(view.range, { from: 88, to: 126 })
      assert.equal(view.rangeWrites.length, 1)
      assert.deepEqual(view.autoScaleWrites, [true])
      view.host.emit('pointerdown')
      view.dragTo({ from: 31.25, to: 64.75 })
      view.resetTrace()
      const refreshed: ChartProps = { ...p, replayFocus: focus,
        candles: p.candles.map((point, i) => ({ ...point, ...(i === p.candles.length - 1 ? { close: point.close + .01 } : {}) })),
        modelPath: p.modelPath.map((point) => ({ ...point })),
        modelHistory: p.modelHistory.map((point) => ({ ...point })),
      }
      view.render(refreshed)
      view.render({ ...refreshed, theme: 'light', modelBandVisible: false })
      assert.equal(view.rangeWrites.length, 0)
      assert.equal(view.autoScaleWrites.length, 0)
      assert.deepEqual(view.range, { from: 31.25, to: 64.75 })
      assert.equal(view.autoScale, false)
      view.render({ ...refreshed, replayFocus: { ...focus, requestId: 2 } })
      assert.equal(view.rangeWrites.length, 1, 'a new user navigation action may locate the same origin again')
      assert.deepEqual(view.range, { from: 88, to: 126 })
      assert.deepEqual(view.autoScaleWrites, [true])
      assert.equal(view.created, 1)
    } finally { view.dispose() }
  })

  test('a focus request cannot move the chart when replay is hidden or the origin is absent', () => {
    const p = fixedReplayProps()
    const view = mount(p)
    try {
      view.resetTrace()
      view.render({ ...p, showModel: false, replayFocus: { originTime: p.modelHistory[0].originTime, requestId: 1 } })
      view.render({ ...p, replayFocus: { originTime: p.candles[0].time - 3600, requestId: 2 } })
      assert.equal(view.rangeWrites.length, 0)
      assert.equal(view.autoScaleWrites.length, 0)
    } finally { view.dispose() }
  })
})

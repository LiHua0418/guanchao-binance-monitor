import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { describe, test } from 'node:test'
import { runInNewContext } from 'node:vm'
import { createElement, type ComponentType } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import * as format from '../src/lib/format.ts'
import { replayB3, validateB3Bundle, type B3Forecast, type ReplayPoint } from '../src/lib/b3.ts'
import { INTERVAL_MS, type Candle } from '../src/types.ts'

const require = createRequire(import.meta.url)

// The root tsconfig only references app/node configs, so the test runner's
// default JSX transform differs from Vite. Compile the actual component source
// with the app's react-jsx setting, preserving real React SSR and formatters.
function component<P>(file: string, dependencies: Record<string, unknown> = {}): ComponentType<P> {
  const source = readFileSync(new URL(`../src/components/${file}.tsx`, import.meta.url), 'utf8')
  const compiled = ts.transpileModule(source, {
    fileName: `${file}.tsx`,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2023,
      jsx: ts.JsxEmit.ReactJSX,
    },
  })
  const exports: Record<string, ComponentType<P>> = {}
  runInNewContext(compiled.outputText, {
    exports,
    require(name: string) {
      if (name === 'react/jsx-runtime') return require(name)
      if (name === '../lib/format') return format
      assert.ok(name in dependencies, `unexpected runtime import: ${name}`)
      return dependencies[name]
    },
  })
  assert.equal(typeof exports[file], 'function')
  return exports[file]
}

type ReturnsProps = { path: B3Forecast['path']; nextMaePct: number }
type PanelProps = {
  model: B3Forecast | null
  history: ReplayPoint[]
  status: 'loading' | 'ready' | 'insufficient' | 'unsupported' | 'error'
  message?: string
  onRetry: () => void
  intervalLabel: string
}

const ForecastReturns = component<ReturnsProps>('ForecastReturns')
const ModelPanel = component<PanelProps>('ModelPanel', { './ForecastReturns': { ForecastReturns } })
const deployed = validateB3Bundle(JSON.parse(readFileSync(new URL('../public/models/b3-1h.json', import.meta.url), 'utf8')), '1h')

function forecast(percentages = [0, 0.0012, -0.0024, 0.0036, ...Array<number>(9).fill(0.0012)]): B3Forecast {
  const path = percentages.map((pct, index) => ({
    time: 1800000000 + index * 3600,
    mid: 100 * (1 + pct / 100),
    lower: 99,
    upper: 101,
  }))
  return { path, endPct: percentages.at(-1)!, bundle: structuredClone(deployed) }
}

function panel(model = forecast(), history: ReplayPoint[] = [], status: PanelProps['status'] = 'ready'): string {
  return renderToStaticMarkup(createElement(ModelPanel, { model, history, status, onRetry() {}, intervalLabel: '1时' }))
}

function statistic(markup: string, label: string): string {
  const found = markup.match(new RegExp(`<dt>${label}</dt><dd[^>]*>([^<]*)</dd>`))
  assert.ok(found, `missing statistic ${label}`)
  return found[1]
}

function chartGeometry(markup: string) {
  const path = markup.match(/<path d="([^"]+)" class="b3-returns-line"/)
  assert.ok(path, 'the real SVG path must exist')
  const points = [...path[1].matchAll(/[ML]([^, ]+),([^ ]+)/g)].map((match) => ({ x: Number(match[1]), y: Number(match[2]) }))
  const zero = markup.match(/<line[^>]*y1="([^"]+)"[^>]*class="b3-returns-zero"/)
  const grid = [...markup.matchAll(/<line[^>]*y1="([^"]+)"[^>]*class="b3-returns-grid"/g)].map((match) => Number(match[1]))
  const extent = markup.match(/纵轴范围正负 ([\d.]+)%/)
  assert.ok(zero && extent && grid.length === 2, 'the displayed scale and zero grid must be accessible')
  return { points, zero: Number(zero[1]), halfHeight: Math.abs(grid[0] - grid[1]) / 2, extent: Number(extent[1]) }
}

describe('B3 signal presentation uses evidence rather than line shape', () => {
  test('a moving point estimate and even lower MAE do not imply a validated directional signal', () => {
    for (const direction of [-1, 1]) {
      const model = forecast([0, ...Array<number>(12).fill(3 * direction)])
      model.bundle.metrics.byHorizon[0].maePct = 0.1
      model.bundle.metrics.byHorizon[0].baselineMaePct = 0.2
      model.bundle.metrics.byHorizon[0].hitRate = 0.99
      const html = panel(model)
      assert.match(html, /方向未验证/)
      assert.match(html, /方向与幅度待验证/)
      assert.match(html, /微小差值不代表方向或幅度已验证/)
      assert.match(html, /class="bias model-bias flat"/)
      assert.match(html, /class="b3-test-verdict b3-unproven"/)
      assert.doesNotMatch(html, /看多|看空|b3-improved|model-bias (?:up|down)/)
    }
  })

  test('a zero shrink coefficient identifies baseline fallback instead of forecasting a sideways market', () => {
    const model = forecast(Array<number>(13).fill(0))
    model.bundle.blend = 0
    model.bundle.blendByHorizon = Array<number>(12).fill(0)
    const html = panel(model)
    assert.match(html, /<strong>持平基线<\/strong>/)
    assert.match(html, /下一根回退至起点/)
    assert.equal(statistic(html, '方向命中'), '无方向信号')
    assert.doesNotMatch(html, /看多|看空|<strong>走平<\/strong>/)
  })

  test('next-bar and plotted returns preserve a real 0.0012 percent change', () => {
    const model = forecast()
    const html = panel(model)
    assert.match(html, /点估计上<\/span>\+0\.0012%/)
    assert.equal(statistic(html, '\\+1 根'), '+0.0012%')
    assert.match(html, /<title>\+1 根：\+0\.0012%<\/title>/)
    assert.doesNotMatch(html, /<title>\+1 根：\+?0\.00%<\/title>/)
  })

  test('the reused holdout is explicitly described as a reevaluation', () => {
    const html = panel()
    assert.match(html, /下一根 · 留出时段复评/)
    assert.match(html, /同一留出时段已用于版本复评/)
    assert.doesNotMatch(html, /独立离线测试|独立测试|个测试起点/)
  })

  test('loading and missing-model states cannot retain an old directional or numeric estimate', () => {
    for (const status of ['loading', 'insufficient', 'error'] as const) {
      const html = panel(forecast(), [], status)
      assert.match(html, /role="status"/)
      assert.doesNotMatch(html, /b3-point-estimate|b3-returns-line|方向未验证/)
    }
  })
})

describe('B3 return plot preserves the forecast amplitude', () => {
  test('tiny positive and negative estimates use at least the next-bar MAE scale without being magnified', () => {
    const percentages = [0, 0.0012, -0.0024, ...Array<number>(10).fill(0)]
    const model = forecast(percentages)
    const html = renderToStaticMarkup(createElement(ForecastReturns, { path: model.path, nextMaePct: 0.3 }))
    const geometry = chartGeometry(html)
    assert.equal(geometry.points.length, model.path.length)
    assert.equal(geometry.extent, 0.3)
    for (const [index, value] of percentages.entries()) {
      const visualPercent = (geometry.zero - geometry.points[index].y) / geometry.halfHeight * geometry.extent
      assert.ok(Math.abs(visualPercent - value) < 1e-10, `point ${index} displays ${visualPercent}%, expected ${value}%`)
    }
    assert.ok(Math.abs(geometry.points[1].y - geometry.zero) < geometry.halfHeight * 0.01,
      'a movement below one percent of MAE must remain below one percent of plot height')
    assert.match(html, /偏移为本次 12 步估计，误差为历史下一根 MAE，时间跨度不同/)
  })

  test('large actual forecasts expand the scale to fit instead of clipping or changing their returns', () => {
    const model = forecast([0, 2, -1, ...Array<number>(10).fill(0.5)])
    const geometry = chartGeometry(renderToStaticMarkup(createElement(ForecastReturns, { path: model.path, nextMaePct: 0.3 })))
    assert.ok(geometry.extent >= 2)
    const value = (geometry.zero - geometry.points[1].y) / geometry.halfHeight * geometry.extent
    assert.ok(Math.abs(value - 2) < 1e-10)
    assert.ok(Math.abs(geometry.points[1].y - geometry.zero) < geometry.halfHeight)
  })

  test('a completely flat forecast remains flat at the zero grid', () => {
    const geometry = chartGeometry(renderToStaticMarkup(createElement(ForecastReturns, {
      path: forecast(Array<number>(13).fill(0)).path, nextMaePct: 0.3,
    })))
    assert.ok(geometry.points.every((point) => point.y === geometry.zero))
    assert.equal(geometry.extent, 0.3)
  })

  test('invalid forecast prices do not generate a misleading SVG', () => {
    for (const path of [[], [{ time: 0, mid: 0, lower: 0, upper: 0 }],
      [{ time: 0, mid: 100, lower: 99, upper: 101 }, { time: 1, mid: NaN, lower: 99, upper: 101 }]]) {
      assert.equal(renderToStaticMarkup(createElement(ForecastReturns, { path, nextMaePct: 0.3 })), '')
    }
  })
})

describe('B3 replay compares predictions against the same originating-price baseline', () => {
  test('model and baseline errors use only their shared valid sample set', () => {
    const sample = (predicted: number, actual: number, baseline?: number): ReplayPoint => ({
      time: 2, originTime: 1, predicted, actual, baseline,
    })
    const html = panel(forecast(), [
      sample(110, 100, 90), sample(240, 200, 220),
      sample(1000000, 100), sample(NaN, 100, 100), sample(999, 0, 100), sample(999, 100, NaN),
    ])
    assert.equal(statistic(html, '模型回放误差'), '15.0000%')
    assert.equal(statistic(html, '复制上一收盘误差'), '10.0000%')
    assert.match(html, /2 次逐根回放/)
    assert.match(html, /两项使用同一批回放样本/)
  })

  test('missing baseline samples show no data rather than a fabricated zero error', () => {
    const html = panel(forecast(), [
      { time: 2, originTime: 1, predicted: 110, actual: 100 },
      { time: 3, originTime: 2, predicted: 180, actual: 200 },
    ])
    assert.equal(statistic(html, '模型回放误差'), '10.0000%')
    assert.equal(statistic(html, '复制上一收盘误差'), '暂无数据')
    assert.doesNotMatch(html, /两项使用同一批回放样本/)
    const empty = panel()
    assert.match(empty, /当前窗口还没有已收盘、可对照下一根预测的回放样本/)
    assert.doesNotMatch(empty, /<dt>模型回放误差<\/dt>/)
  })

  test('the actual replay engine stores the previous closed price and never revises it using future data', () => {
    const step = INTERVAL_MS['1h']
    const start = Date.parse(deployed.dataCutoff) / 1000 - 100 * step / 1000
    const candles: Candle[] = Array.from({ length: 125 }, (_, index) => {
      const close = 100 + index * 0.1 + Math.sin(index * 0.7)
      return { time: start + index * step / 1000, open: close - 0.1,
        high: close + 0.2, low: close - 0.2, close, volume: 500 + index, isClosed: true }
    })
    const replay = replayB3(candles, step, deployed, 8)
    assert.equal(replay.length, 8)
    for (const point of replay) {
      const origin = candles.find((candle) => candle.time === point.originTime)!
      assert.equal(point.baseline, origin.close)
      assert.notEqual(point.baseline, origin.open)
      assert.equal(point.actual, candles.find((candle) => candle.time === point.time)!.close)
    }
    const chosen = replay[3]
    const altered = candles.map((candle) => candle.time <= chosen.originTime ? { ...candle } : {
      ...candle, open: candle.open * 2, high: candle.high * 2, low: candle.low * 2,
      close: candle.close * 2, volume: candle.volume * 3,
    })
    const after = replayB3(altered, step, deployed, 8).find((point) => point.originTime === chosen.originTime)!
    assert.equal(after.baseline, chosen.baseline)
    assert.equal(after.predicted, chosen.predicted)
    assert.equal(after.actual, chosen.actual * 2)
  })
})

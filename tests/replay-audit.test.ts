import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { describe, test } from 'node:test'
import { runInNewContext } from 'node:vm'
import { createElement, type ComponentType } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import type { B4Forecast, B4ForwardSummary, B4ReplayPath } from '../src/lib/b4.ts'
import * as audit from '../src/lib/replay-audit.ts'
import { INTERVAL_LABEL, INTERVAL_MS, type Interval } from '../src/types.ts'

const START = Date.UTC(2026, 9, 1) / 1000
const require = createRequire(import.meta.url)
function replay(originPrice = 100, actual = 110, predicted = 105, offset = 0, interval: Interval = '1h'): B4ReplayPath {
  const step = INTERVAL_MS[interval] / 1000
  const originTime = START + offset * step
  return { originTime, originPrice, path: Array.from({ length: 13 }, (_, horizon) => ({
    time: originTime + horizon * step, horizon,
    predicted: horizon === 0 ? originPrice : predicted, actual: horizon === 0 ? originPrice : actual,
  })) }
}
function model(records: B4ReplayPath[] = [], interval: Interval = '1h'): B4Forecast {
  return { modelId: 'guanchao-b4', symbol: 'BTCUSDT', interval, originTime: START + 100 * INTERVAL_MS[interval] / 1000,
    originPrice: 110, path: [], history: [], replayPaths: records,
    evaluation: { direction: { balancedAccuracy: .52, baselineBalancedAccuracy: .52, brier: .249, baselineBrier: .25, cvPassed: true },
      magnitude: { mseImprovement: .1, maeImprovement: .04, cvPassed: true, heldoutSkill: true }, samples: 100,
      dataCutoff: new Date(START * 1000).toISOString(), dataEnd: new Date((START + 10000000) * 1000).toISOString() } }
}
function forward(): B4ForwardSummary {
  return { modelVersion: 'fixed-model-version', eligibilityWindowSeconds: 60,
    recordedOrigins: 0, eligibleOrigins: 0, resolvedOrigins: 0, resolvedNextBarOrigins: 0, pendingOrigins: 0, lateOrigins: 0,
    byHorizon: Array.from({ length: 12 }, (_, index) => ({ horizon: index + 1, samples: 0,
      maePct: null, baselineMaePct: null, skillPct: null, directionAccuracy: null, directionSamples: 0,
      probabilityDirectionAccuracy: null, probabilitySamples: 0, brier: null })), recent: [] }
}

describe('same-origin replay price comparison', () => {
  test('both columns use actual-price denominators and each origin remains fixed through all 12 steps', () => {
    const first = replay(100, 110, 105)
    const second = replay(200, 220, 198, 1)
    const stats = audit.replayAuditMetrics(model([first, second]))
    assert.equal(stats.samples, 2)
    const expectedModel = (Math.abs(105 / 110 - 1) + Math.abs(198 / 220 - 1)) * 50
    const expectedBaseline = (Math.abs(100 / 110 - 1) + Math.abs(200 / 220 - 1)) * 50
    for (const row of stats.byHorizon) {
      assert.equal(row.samples, 2)
      assert.ok(Math.abs(row.maePct! - expectedModel) < 1e-12)
      assert.ok(Math.abs(row.baselineMaePct! - expectedBaseline) < 1e-12)
      assert.ok(Math.abs(row.differencePct! - (expectedModel - expectedBaseline)) < 1e-12)
      assert.ok(Math.abs(row.skillPct! - (1 - expectedModel / expectedBaseline) * 100) < 1e-12)
    }
    assert.ok(stats.byHorizon[11].baselineMaePct! > 9,
      'h12 baseline cannot roll forward to the h11 actual, which would make its error zero here')
  })

  test('incomplete, invalid, future or pre-freeze origins are removed for every horizon together', () => {
    const valid = replay()
    const incomplete = replay(100, 110, 105, 1)
    incomplete.path.pop()
    const invalid = replay(100, 110, 105, 2)
    invalid.path[6].actual = NaN
    const future = replay(100, 110, 105, 99)
    const beforeFreeze = replay(100, 110, 105, -1)
    const wrongStep = replay(100, 110, 105, 3)
    wrongStep.path[5].time += 1
    const wrongAnchor = replay(100, 110, 105, 4)
    wrongAnchor.path[0].predicted = 101
    const stats = audit.replayAuditMetrics(model([valid, incomplete, invalid, future, beforeFreeze, wrongStep, wrongAnchor]))
    assert.equal(stats.samples, 1)
    assert.equal(stats.rejectedRecords, 6)
    assert.ok(stats.byHorizon.every((row) => row.samples === 1))
  })

  test('duplicate origins cannot double their weight or select the more favorable copy', () => {
    const duplicated = replay(100, 110, 110)
    const conflicting = replay(100, 110, 200)
    const unique = replay(100, 110, 105, 1)
    const stats = audit.replayAuditMetrics(model([duplicated, conflicting, unique]))
    assert.equal(stats.samples, 1)
    assert.equal(stats.rejectedRecords, 2)
    assert.ok(Math.abs(stats.byHorizon[0].maePct! - Math.abs(105 / 110 - 1) * 100) < 1e-12)
  })

  test('zero-baseline error has no percentage skill and missing 12-step data has no numeric score', () => {
    const flat = audit.replayAuditMetrics(model([replay(100, 100, 100)]))
    assert.ok(flat.byHorizon.every((row) => row.maePct === 0 && row.baselineMaePct === 0 && row.skillPct === null))
    const absent = model()
    delete absent.replayPaths
    const empty = audit.replayAuditMetrics(absent)
    assert.equal(empty.samples, 0)
    assert.equal(empty.meanMaePct, null)
    assert.ok(empty.byHorizon.every((row) => row.maePct === null && row.baselineMaePct === null && row.skillPct === null))
  })

  for (const interval of ['15m', '1h', '4h', '1d'] as const) test(`${interval} keeps all twelve actual per-bar steps`, () => {
    const stats = audit.replayAuditMetrics(model([replay(100, 110, 105, 0, interval)], interval))
    assert.equal(stats.samples, 1)
    assert.deepEqual(stats.byHorizon.map((row) => row.horizon), Array.from({ length: 12 }, (_, i) => i + 1))
  })
})

const compiled = ts.transpileModule(readFileSync(new URL('../src/components/ReplayAudit.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
})
const exports: { ReplayAudit?: ComponentType<{ model: B4Forecast }> } = {}
runInNewContext(compiled.outputText, { exports, require(name: string) {
  if (name === 'react/jsx-runtime') return require(name)
  if (name === '../lib/replay-audit') return audit
  if (name === '../types') return { INTERVAL_LABEL, INTERVAL_MS }
  throw new Error(`Unexpected replay component dependency ${name}`)
} })
const render = (value: B4Forecast) => renderToStaticMarkup(createElement(exports.ReplayAudit!, { model: value }))

describe('historical reconstruction and actual forward records stay visibly distinct', () => {
  test('worse same-origin prediction is explicit, with the correct common sample count and comparison columns', () => {
    const html = render(model([replay(100, 110, 90)]))
    assert.match(html, /价格预测尚未优于持平基线/)
    assert.match(html, /1 个共同起点/)
    assert.match(html, /<th>模型 MAE<\/th><th>持平 MAE<\/th>/)
    assert.match(html, /18\.1818%/)
    assert.match(html, /9\.0909%/)
    assert.match(html, /不是当时发布的预测/)
    assert.match(html, /滚动一步紫线每根都重设起点/)
    assert.doesNotMatch(html, /相关系数/)
  })

  test('a better historical window is not described as a proven future advantage', () => {
    const html = render(model([replay(100, 110, 109)]))
    assert.match(html, /当前窗口价格误差较小，稳定优势仍需实时前向验证/)
    assert.doesNotMatch(html, /价格预测已经证明|稳定盈利|实盘准确/)
  })

  test('zero recorded or settled samples never turn null errors into 0% accuracy', () => {
    const value = model()
    value.forward = forward()
    const html = render(value)
    assert.match(html, /尚无已结算实时样本/)
    assert.match(html, /尚未存入首发预测/)
    assert.doesNotMatch(html, /0\.0000%/)
    assert.doesNotMatch(html, /forward-audit-table/)
    assert.match(html, /最多延迟 60 秒/)
    assert.match(html, /首发不可覆盖/)
  })

  test('partially settled eligible records show h1 data while h12 stays unresolved, and late records are labeled', () => {
    const value = model([replay()])
    const live = forward()
    live.recordedOrigins = 4
    live.eligibleOrigins = 3
    live.lateOrigins = 1
    live.pendingOrigins = 4
    live.resolvedNextBarOrigins = 1
    live.byHorizon[0] = { ...live.byHorizon[0], samples: 1, maePct: .35, baselineMaePct: .3, skillPct: -16.6667 }
    live.recent = Array.from({ length: 4 }, (_, index) => ({ originTime: START + index * 3600,
      issuedAt: START + (index + 1) * 3600 + (index === 3 ? 65 : 5), eligibility: index === 3 ? 'late' : 'eligible',
      path: Array.from({ length: 12 }, (_, offset) => ({ time: START + (index + offset + 1) * 3600,
        horizon: offset + 1, predicted: 100, ...(offset === 0 ? { actual: 101 } : {}) })) }))
    value.forward = live
    const html = render(value)
    assert.match(html, /0\.3500%/)
    assert.match(html, /0\.3000%/)
    assert.match(html, /<td>\+12<\/td><td[^>]*>0<\/td><td[^>]*>未结算<\/td>/)
    assert.match(html, /迟到 · 不计成绩/)
    assert.match(html, /已结算 1\/12 · 待结算/)
    assert.equal((html.match(/实际发出：/g) ?? []).length, 3)
    assert.match(html, /2026-10-01 04:01:05 UTC/)
    assert.match(html, /全部结算和待结算计数包含迟到记录，误差成绩不包含/)
  })
})

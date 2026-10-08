import { useState, type KeyboardEvent } from 'react'
import { formatCompact, formatPct, formatPrice, formatQty, formatSigned } from '../lib/format'
import type { Analysis } from '../lib/analysis'
import type { B3Forecast, ReplayPoint } from '../lib/b3'
import { ModelPanel } from './ModelPanel'
import { B4Panel } from './B4Panel'
import type { B4Forecast, B4Status } from '../lib/b4'
import type { Depth, Quote } from '../types'

type Props = {
  modelEngine?: 'b3' | 'b4'
  b4?: { model: B4Forecast | null; status: B4Status; message?: string; retry: () => void }
  analysis: Analysis | null
  model: B3Forecast | null
  modelHistory: ReplayPoint[]
  modelStatus: 'loading' | 'ready' | 'insufficient' | 'unsupported' | 'error'
  modelMessage?: string
  onModelRetry: () => void
  loading: boolean
  quote: Quote | undefined
  depth: Depth | null
  hero: number | null
  digits: number
  intervalLabel: string
}

function toneOf(score: number): 'up' | 'down' | 'flat' {
  if (score > 0.2) return 'up'
  if (score < -0.2) return 'down'
  return 'flat'
}

function Spark({ values }: { values: number[] }) {
  if (values.length < 2) return null
  const width = 128
  const height = 36
  const min = Math.min(...values, 30)
  const max = Math.max(...values, 70)
  const span = max - min || 1
  const path = values.map((value, index) => {
    const x = (index / (values.length - 1)) * width
    const y = height - ((value - min) / span) * (height - 4) - 2
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
  const mid = height - ((50 - min) / span) * (height - 4) - 2
  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <line x1="0" x2={width} y1={mid} y2={mid} />
      <path d={path} />
    </svg>
  )
}

export function SignalDeck({ modelEngine = 'b3', b4, analysis, model, modelHistory, modelStatus, modelMessage, onModelRetry, loading, quote, depth, hero, digits, intervalLabel }: Props) {
  const views = [{ id: 'model', label: '模型' }, { id: 'technical', label: '技术面' }, { id: 'book', label: '盘口' }] as const
  const [view, setView] = useState<(typeof views)[number]['id']>('model')
  const moveTab = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const next = event.key === 'ArrowRight' ? (index + 1) % views.length
      : event.key === 'ArrowLeft' ? (index + views.length - 1) % views.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? views.length - 1 : null
    if (next === null) return
    event.preventDefault()
    setView(views[next].id)
    event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('button')[next]?.focus()
  }
  const asks = depth ? [...depth.asks].slice(0, 8).reverse() : []
  const bids = depth ? depth.bids.slice(0, 8) : []
  const maxQty = Math.max(1, ...asks.map((level) => level.qty), ...bids.map((level) => level.qty))
  const bidQty = depth?.bids.reduce((sum, level) => sum + level.qty, 0) ?? 0
  const askQty = depth?.asks.reduce((sum, level) => sum + level.qty, 0) ?? 0
  const bidShare = bidQty + askQty > 0 ? bidQty / (bidQty + askQty) : null
  const bestBid = depth?.bids[0]
  const bestAsk = depth?.asks[0]
  const spread = bestBid && bestAsk ? bestAsk.price - bestBid.price : null
  const mid = bestBid && bestAsk ? (bestAsk.price + bestBid.price) / 2 : null
  const spreadBps = spread != null && mid ? (spread / mid) * 10000 : null
  const amplitude = quote && quote.open ? ((quote.high - quote.low) / quote.open) * 100 : null
  const marker = analysis ? ((analysis.score + analysis.cap) / (analysis.cap * 2)) * 100 : 50

  return (
    <aside className="panel signal" aria-label="市场观察面板">
      <div className="inspector-header"><span className="kicker">市场观察</span><span className="inspector-period">{intervalLabel}周期</span></div>
      <div className="inspector-tabs" role="tablist" aria-label="观察内容">
        {views.map((item, index) => <button key={item.id} id={`inspector-${item.id}`} role="tab" type="button"
          aria-selected={view === item.id} aria-controls={`inspector-panel-${item.id}`} tabIndex={view === item.id ? 0 : -1}
          className={view === item.id ? 'on' : ''} onClick={() => setView(item.id)} onKeyDown={(event) => moveTab(event, index)}>{item.label}</button>)}
      </div>
      <div className="inspector-body">
      <div role="tabpanel" id="inspector-panel-model" aria-labelledby="inspector-model" tabIndex={0} hidden={view !== 'model'}>
      {modelEngine === 'b4' && b4
        ? <B4Panel model={b4.model} status={b4.status} message={b4.message} onRetry={b4.retry} intervalLabel={intervalLabel} />
        : <ModelPanel model={model} history={modelHistory} status={modelStatus} message={modelMessage} onRetry={onModelRetry} intervalLabel={intervalLabel} />}
      </div>

      <div role="tabpanel" id="inspector-panel-technical" aria-labelledby="inspector-technical" tabIndex={0} hidden={view !== 'technical'}>
      <section className="block">
        <p className="kicker">技术面倾向 · {intervalLabel}</p>
        {analysis ? (
          <>
            <div className={`bias ${toneOf(analysis.score)}`}>
              <strong>{analysis.bias}</strong>
              <span>规则合计 {formatSigned(analysis.score, 2)} / {analysis.cap.toFixed(1)}</span>
            </div>
            <div className="meter" aria-hidden="true">
              <i style={{ left: `${Math.min(100, Math.max(0, marker))}%` }} />
            </div>
            <p className="meter-label">只对当前{intervalLabel}。换周期，方向可以不一样。这是规则打分，不是胜率。</p>
            <p className="factor-count">
              <span className="up">多头因子 {analysis.bulls}</span>
              <span className="down">空头因子 {analysis.bears}</span>
            </p>
            <ul className="factors">
              {analysis.factors.map((factor) => (
                <li key={factor.id}>
                  <span className={`vote ${toneOf(factor.score)}`} />
                  <span>
                    <strong>{factor.label}</strong>
                    <em>{factor.detail}</em>
                  </span>
                </li>
              ))}
            </ul>
            <div className="proj">
              <div>
                <span>{intervalLabel} · 12 根后示意中枢</span>
                <strong className="num">{formatPrice(analysis.forecast.at(-1)?.mid ?? 0, digits)}</strong>
              </div>
              <em className={analysis.endPct >= 0 ? 'up' : 'down'}>{formatPct(analysis.endPct)}</em>
            </div>
            <p className="note">{analysis.note}</p>
          </>
        ) : (
          <p className="note quiet">{loading ? 'K 线到齐后才会计算倾向。' : '样本还不够，暂时不给方向。'}</p>
        )}
      </section>

      <section className="block">
        <p className="kicker">指标读数</p>
        <div className="reads">
          <div>
            <span>RSI 14</span>
            <strong className="num">{analysis ? analysis.rsi.toFixed(1) : '—'}</strong>
            {analysis && <Spark values={analysis.rsiSeries} />}
          </div>
          <div>
            <span>MACD 柱</span>
            <strong className={`num ${analysis && analysis.macd.hist >= 0 ? 'up' : 'down'}`}>
              {analysis ? formatSigned(analysis.macd.hist, Math.min(4, digits)) : '—'}
            </strong>
            <em>{analysis ? `DIF ${formatSigned(analysis.macd.dif, Math.min(4, digits))}` : ' '}</em>
          </div>
          <div>
            <span>布林位置</span>
            <strong className="num">{analysis ? `${(analysis.bollPb * 100).toFixed(0)}%` : '—'}</strong>
            <em>{analysis ? `ATR ${formatPrice(analysis.atr, digits)}` : ' '}</em>
          </div>
        </div>
      </section>

      </div>
      <div role="tabpanel" id="inspector-panel-book" aria-labelledby="inspector-book" tabIndex={0} hidden={view !== 'book'}>
      <section className="block">
        <div className="block-title">
          <p className="kicker">买卖盘</p>
          {spread != null && spreadBps != null && (
            <span className="spread num">价差 {formatPrice(spread, digits)} · {spreadBps.toFixed(1)} bp</span>
          )}
        </div>
        {depth && asks.length > 0 ? (
          <div className="book">
            {asks.map((level) => (
              <div key={`a-${level.price}`} className="book-row ask">
                <i style={{ width: `${(level.qty / maxQty) * 100}%` }} />
                <span className="num">{formatPrice(level.price, digits)}</span>
                <span className="num">{formatQty(level.qty)}</span>
              </div>
            ))}
            <div className="book-mid">最优买卖</div>
            {bids.map((level) => (
              <div key={`b-${level.price}`} className="book-row bid">
                <i style={{ width: `${(level.qty / maxQty) * 100}%` }} />
                <span className="num">{formatPrice(level.price, digits)}</span>
                <span className="num">{formatQty(level.qty)}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="note quiet">盘口连接中。</p>
        )}
        {bidShare != null && (
          <p className="note quiet">前 20 档买盘占比 {(bidShare * 100).toFixed(0)}%。这是挂单快照，不是方向判断。</p>
        )}
      </section>

      <section className="block last">
        <p className="kicker">24 小时</p>
        <dl className="stats">
          <div>
            <dt>成交量</dt>
            <dd className="num">{quote ? formatCompact(quote.volume) : '—'}</dd>
          </div>
          <div>
            <dt>成交额</dt>
            <dd className="num">{quote ? formatCompact(quote.quoteVolume) : '—'}</dd>
          </div>
          <div>
            <dt>成交笔数</dt>
            <dd className="num">{quote?.count ? formatCompact(quote.count) : '—'}</dd>
          </div>
          <div>
            <dt>振幅</dt>
            <dd className="num">{amplitude == null ? '—' : `${amplitude.toFixed(2)}%`}</dd>
          </div>
        </dl>
        {hero == null && <p className="note quiet">价格还在路上。</p>}
      </section>
      </div>
      </div>
    </aside>
  )
}

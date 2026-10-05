import { formatCompact, formatPct, formatPrice, formatQty, formatSigned } from '../lib/format'
import type { Analysis } from '../lib/analysis'
import type { NeuralForecast } from '../lib/lstm'
import type { Depth, Quote } from '../types'

type Props = {
  analysis: Analysis | null
  model: NeuralForecast | null
  training: boolean
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

export function SignalDeck({ analysis, model, training, loading, quote, depth, hero, digits, intervalLabel }: Props) {
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
    <aside className="panel signal">
      <section className="block">
        <p className="kicker">预测模型</p>
        {model ? (
          <>
            <div className={`bias model-bias ${model.bias === '看多' ? 'up' : model.bias === '看空' ? 'down' : 'flat'}`}>
              <strong>{model.bias}</strong>
              <span>LSTM · {model.params} 个参数</span>
            </div>
            <div className="proj">
              <div>
                <span>{intervalLabel} · 12 根后模型价格</span>
                <strong className="num">{formatPrice(model.path.at(-1)?.mid ?? 0, digits)}</strong>
              </div>
              <em className={model.endPct >= 0 ? 'up' : 'down'}>{formatPct(model.endPct)}</em>
            </div>
            <p className="note">
              单层 LSTM，16 个隐藏单元，{model.params} 个参数。权重已经在本机训完，用的是 {model.trainedSymbols.join('、')} 共 {model.trainedBars} 根 K 线，停在验证损失最低的第 {model.epochs} 轮，不是打开页面现训的。
            </p>
            <p className="note">
              训练时留出的检验命中 {(model.trainedHitRate * 100).toFixed(0)}%。当前这个交易对最近 {model.holdout} 根命中 {(model.hitRate * 100).toFixed(0)}%，下一根平均误差 {model.maePct.toFixed(2)}%。这都不是以后的胜率。
            </p>
            <p className="note quiet">钢蓝虚线是把预测收盘接回去再递推 12 根。越往右越不可靠。点线是残差波动带，不是目标价。</p>
          </>
        ) : (
          <p className="note quiet">
            {training
              ? 'LSTM 还在本机训练，验证损失没稳住之前不会结束。'
              : loading
                ? 'K 线到齐后才会出模型结果。'
                : '这个周期的网络还没训练好。'}
          </p>
        )}
      </section>

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
    </aside>
  )
}

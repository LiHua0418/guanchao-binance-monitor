import { useEffect, useMemo, useState } from 'react'
import { ChartPane } from './components/ChartPane'
import { SignalDeck } from './components/SignalDeck'
import { Watchlist } from './components/Watchlist'
import { analyze } from './lib/analysis'
import { baseAsset, formatClock, formatPct, formatPrice, formatSigned, inferDigits, quoteAsset } from './lib/format'
import { useMarket } from './hooks/useMarket'
import { useNeuralForecast } from './hooks/useNeuralForecast'
import { useTimeframeBias } from './hooks/useTimeframeBias'
import { HORIZON_LABEL, INTERVALS, INTERVAL_LABEL, INTERVAL_MS, NAMES, QUOTE_MARKETS, symbolsForQuote } from './types'

const THEME_KEY = 'guanchao.theme'
type ThemeName = 'dark' | 'light'

function loadTheme(): ThemeName {
  try {
    return localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

const LINK_LABEL = {
  connecting: '连接中',
  live: '实时',
  reconnecting: '重连中',
}

function TideMark() {
  return (
    <svg className="mark" viewBox="0 0 36 36" aria-hidden="true">
      <rect width="36" height="36" rx="11" />
      <path d="M7 22.2c3.3-5 6.2-5 9.5 0s6.2 5 9.5 0" />
      <path d="M7 16.2c3.3-4.2 6.2-4.2 9.5 0s6.2 4.2 9.5 0" />
    </svg>
  )
}

export default function App() {
  const market = useMarket()
  const timeframeBias = useTimeframeBias(market.selected)
  const [showEma, setShowEma] = useState(true)
  const [showBands, setShowBands] = useState(false)
  const [showForecast, setShowForecast] = useState(true)
  const [showModel, setShowModel] = useState(true)
  const [theme, setTheme] = useState<ThemeName>(loadTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem(THEME_KEY, theme)
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'light' ? '#f3f1ec' : '#0a0b0e')
  }, [theme])
  const quote = market.quotes[market.selected]
  const hero = market.hero
  const digits = hero != null ? inferDigits(hero) : 2
  const change = quote && hero != null && quote.open ? hero - quote.open : null
  const pct = quote && hero != null && quote.open ? (change! / quote.open) * 100 : null
  const tone = pct == null ? 'flat' : pct > 0 ? 'up' : pct < 0 ? 'down' : 'flat'
  const rangePos = quote && hero != null && quote.high > quote.low
    ? ((hero - quote.low) / (quote.high - quote.low)) * 100
    : null
  const analysis = useMemo(
    () => analyze(market.candles, INTERVAL_MS[market.timeframe]),
    [market.candles, market.timeframe],
  )
  const neural = useNeuralForecast(market.candles, INTERVAL_MS[market.timeframe], market.timeframe)
  const model = neural.model
  const clock = formatClock(market.clock)
  const customized = market.symbols.join(',') !== symbolsForQuote(market.quoteMarket).join(',')

  return (
    <div className="app">
      <div className="mast">
      <header className="top">
        <div className="brand">
          <TideMark />
          <div>
            <h1>观潮</h1>
            <p>现货观察台</p>
          </div>
        </div>
        <div className="top-meta">
          <div className="theme-switch" role="group" aria-label="外观">
            <button type="button" className={theme === 'light' ? 'on' : ''} aria-pressed={theme === 'light'} onClick={() => setTheme('light')}>
              白色
            </button>
            <button type="button" className={theme === 'dark' ? 'on' : ''} aria-pressed={theme === 'dark'} onClick={() => setTheme('dark')}>
              夜间
            </button>
          </div>
          <span className={`live ${market.link}`}>
            <i />
            {LINK_LABEL[market.link]}
          </span>
          <time dateTime={market.clock.toISOString()}>
            <strong className="num">{clock.time}</strong>
            <span>{clock.date} · 北京时间</span>
          </time>
        </div>
      </header>
      <div className="quote-bar" role="tablist" aria-label="计价市场">
        <span>计价</span>
        {QUOTE_MARKETS.map((quote) => (
          <button
            key={quote}
            type="button"
            role="tab"
            aria-selected={market.quoteMarket === quote}
            className={market.quoteMarket === quote ? 'on' : ''}
            onClick={() => market.setQuoteMarket(quote)}
          >
            {quote}
          </button>
        ))}
      </div>

      {market.restError && (
        <div className="banner" role="status">
          <span>{market.restError}</span>
          <button type="button" onClick={market.retry}>重试</button>
        </div>
      )}
      </div>

      <main className="stage">
        <Watchlist
          symbols={market.symbols}
          quotes={market.quotes}
          selected={market.selected}
          hero={hero}
          customized={customized}
          onSelect={market.setSelected}
          onRemove={market.removeSymbol}
          onAdd={market.addSymbol}
          onReset={market.resetSymbols}
        />

        <section className="panel chart-panel">
          <div className="chart-head">
            <div className="pair">
              <p className="kicker">{quoteAsset(market.selected)} 现货</p>
              <h2>
                {NAMES[baseAsset(market.selected)] ?? baseAsset(market.selected)}
                <span>{baseAsset(market.selected)} / {quoteAsset(market.selected)}</span>
              </h2>
            </div>
            <div className="intervals" role="tablist" aria-label="K 线周期">
              {INTERVALS.map((item) => {
                const bias = item === market.timeframe ? analysis?.bias ?? timeframeBias[item] : timeframeBias[item]
                const biasTone = bias === '偏多' ? 'up' : bias === '偏空' ? 'down' : 'flat'
                return (
                  <button
                    key={item}
                    type="button"
                    role="tab"
                    aria-selected={market.timeframe === item}
                    className={market.timeframe === item ? 'on' : ''}
                    title={`${INTERVAL_LABEL[item]}的技术面倾向，只代表这个周期`}
                    onClick={() => market.setTimeframe(item)}
                  >
                    {INTERVAL_LABEL[item]}
                    <em className={biasTone}>{bias ?? '…'}</em>
                  </button>
                )
              })}
            </div>
          </div>

          <div className="hero-row">
            <div>
              <p className={`hero-price num ${tone}`}>{hero == null ? '—' : formatPrice(hero, digits)}</p>
              <p className={`chg ${tone}`}>
                <strong className="num">{change == null ? '—' : formatSigned(change, digits)}</strong>
                <span className="num">{pct == null ? '—' : formatPct(pct)}</span>
                <em>24 小时</em>
              </p>
            </div>
            <div className="toggles">
              <button type="button" className={showEma ? 'on' : ''} aria-pressed={showEma} title="EMA20 看近 20 根，拐得快。EMA50 看近 50 根，拐得慢。价格和快线都在慢线上方，这段偏多。" onClick={() => setShowEma((value) => !value)}>
                EMA
              </button>
              <button type="button" className={showBands ? 'on' : ''} aria-pressed={showBands} title="中间是 20 根收盘的平均，上下各 2 倍波动。贴着轨道只说明离平均远，不是到顶或到底。" onClick={() => setShowBands((value) => !value)}>
                布林
              </button>
              <button type="button" className={showForecast ? 'on' : ''} aria-pressed={showForecast} title="从最后一根往右画的未来 12 根，按近 30 根斜率外推，不是已经发生的价格，也不是目标价。" onClick={() => setShowForecast((value) => !value)}>
                漂移区间
              </button>
              <button type="button" className={showModel ? 'on' : ''} aria-pressed={showModel} title="钢蓝虚线是单层 LSTM 递推的未来 12 根。网络在浏览器里用反向传播训练，命中率只统计没参加训练的最近 48 根。" onClick={() => setShowModel((value) => !value)}>
                模型
              </button>
            </div>
          </div>

          {quote && rangePos != null && (
            <div className="range">
              <span className="num">{formatPrice(quote.low, digits)}</span>
              <div className="range-track" aria-hidden="true">
                <i style={{ left: `${Math.min(100, Math.max(0, rangePos))}%` }} />
              </div>
              <span className="num">{formatPrice(quote.high, digits)}</span>
            </div>
          )}

          <div className="chart-stage">
            <ChartPane
              symbol={market.selected}
              interval={market.timeframe}
              candles={market.candles}
              forecast={analysis?.forecast ?? []}
              modelPath={model?.path ?? []}
              showEma={showEma}
              showBands={showBands}
              showForecast={showForecast}
              showModel={showModel}
              digits={digits}
              theme={theme}
            />
            {market.candles.length === 0 && (
              <div className="chart-empty">
                <p>{market.candleError ?? '正在接上 K 线…'}</p>
                {market.candleState === 'error' && (
                  <button type="button" onClick={market.retry}>重试</button>
                )}
              </div>
            )}
            {market.candles.length > 0 && (
              <div className="legend">
                {showEma && <span><i className="swatch ema20" />EMA20</span>}
                {showEma && <span><i className="swatch ema50" />EMA50</span>}
                {showBands && <span><i className="swatch band" />布林 20,2</span>}
                {showForecast && <span><i className="swatch forecast" />12 根漂移</span>}
                {showModel && <span><i className="swatch model" />LSTM 路径</span>}
              </div>
            )}
          </div>
          <p className="guide">
            粗矩形是一根 K 线的开盘到收盘，上下细线是这段的最高和最低。
            {showEma && ' 铜金线是 EMA20，海绿线是 EMA50。'}
            {showBands && ' 灰色虚线是布林上下轨。'}
            {(showForecast || showModel) && ` 图右侧虚线是${HORIZON_LABEL[market.timeframe]}，不是已经走出的价格。`}
          </p>
        </section>

        <SignalDeck
          analysis={analysis}
          model={model}
          training={!neural.ready}
          loading={market.candleState === 'loading'}
          quote={quote}
          depth={market.depth}
          hero={hero}
          digits={digits}
          intervalLabel={INTERVAL_LABEL[market.timeframe]}
        />
      </main>

      <footer>
        <p>
          观潮只读币安公开现货行情。右侧规则只是技术面打分。钢蓝线是浏览器里训练的 LSTM 递推出来的路径，命中率只统计没参加训练的最近 48 根，不代表以后，也不构成投资建议。
          K 线由 <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">TradingView</a> Lightweight Charts 绘制。
        </p>
      </footer>
    </div>
  )
}

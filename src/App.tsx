import { useEffect, useMemo, useState } from 'react'
import { ChartPane } from './components/ChartPane'
import { SignalDeck } from './components/SignalDeck'
import { Watchlist } from './components/Watchlist'
import { analyze } from './lib/analysis'
import { baseAsset, formatClock, formatCompact, formatPct, formatPrice, formatSigned, inferDigits, quoteAsset } from './lib/format'
import { useMarket } from './hooks/useMarket'
import { useNeuralForecast } from './hooks/useNeuralForecast'
import { useB4Forecast } from './hooks/useB4Forecast'
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
  const [showForecast, setShowForecast] = useState(false)
  const [showModel, setShowModel] = useState(true)
  const [showModelBand, setShowModelBand] = useState(true)
  const [showModelHistory, setShowModelHistory] = useState(true)
  const [modelEngine, setModelEngine] = useState<'b3' | 'b4'>('b4')
  const [showScenarios, setShowScenarios] = useState(true)
  const [theme, setTheme] = useState<ThemeName>(loadTheme)
  const [chartFocused, setChartFocused] = useState(false)
  const [replaySelection, setReplaySelection] = useState<{ key: string; time: number } | null>(null)
  const [replayFocus, setReplayFocus] = useState<{ originTime: number; requestId: number } | null>(null)

  useEffect(() => {
    if (!chartFocused) return
    const exitFocus = (event: KeyboardEvent) => { if (event.key === 'Escape') setChartFocused(false) }
    window.addEventListener('keydown', exitFocus)
    return () => window.removeEventListener('keydown', exitFocus)
  }, [chartFocused])

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
  const neural = useNeuralForecast(market.candles, INTERVAL_MS[market.timeframe], market.timeframe, market.selected, market.clock.getTime())
  const b4 = useB4Forecast(market.candles, market.timeframe, market.selected, market.clock.getTime(), modelEngine === 'b4')
  const model = modelEngine === 'b4' ? b4.model : neural.model
  const replayKey = `${market.selected}|${market.timeframe}`
  const replayPaths = b4.model?.replayPaths ?? []
  const replay = replaySelection?.key === replayKey
    // A selected origin can expire from the rolling window. Keep navigation
    // usable at the nearest remaining origin without moving the chart viewport.
    ? replayPaths.find((path) => path.originTime >= replaySelection.time) ?? replayPaths.at(-1)
    : replayPaths.at(-1)
  const replayIndex = replay ? replayPaths.indexOf(replay) : -1
  const selectReplay = (index: number) => {
    const chosen = replayPaths[index]
    if (!chosen) return
    setReplaySelection({ key: replayKey, time: chosen.originTime })
    setReplayFocus((previous) => ({ originTime: chosen.originTime, requestId: (previous?.requestId ?? 0) + 1 }))
  }
  const replayTime = replay ? new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date((replay.originTime * 1000) + INTERVAL_MS[market.timeframe])) : ''
  const modelHistory = modelEngine === 'b4' ? (replay?.path.map((point) => ({
    time: point.time, predicted: point.predicted, actual: point.actual,
    originTime: replay.originTime, baseline: replay.originPrice, horizon: point.horizon,
  })) ?? []) : neural.history
  const clock = formatClock(market.clock)
  const customized = market.symbols.join(',') !== symbolsForQuote(market.quoteMarket).join(',')

  return (
    <div className={`app${chartFocused ? ' chart-focused' : ''}`}>
      <div className="mast">
      <header className="top">
        <div className="brand">
          <TideMark />
          <div>
            <h1>观潮</h1>
            <p>现货观察台</p>
          </div>
        </div>
        <div className="quote-bar" role="tablist" aria-label="计价市场">
          <span>现货市场</span>
          {QUOTE_MARKETS.map((quote) => (
            <button key={quote} type="button" role="tab" aria-selected={market.quoteMarket === quote}
              className={market.quoteMarket === quote ? 'on' : ''} onClick={() => market.setQuoteMarket(quote)}>{quote}</button>
          ))}
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

        <section className="panel chart-panel" aria-label="行情图表">
          <div className="chart-head">
            <div className="pair">
              <div className="pair-label"><span className="spot-badge">现货</span><p className="kicker">{baseAsset(market.selected)} / {quoteAsset(market.selected)}</p></div>
              <h2>
                {NAMES[baseAsset(market.selected)] ?? baseAsset(market.selected)}
              </h2>
            </div>
            <div className="chart-head-actions">
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
            <button className="focus-chart" type="button" aria-label={chartFocused ? '退出专注' : '专注看盘'} aria-pressed={chartFocused} onClick={() => setChartFocused((value) => !value)} title="展开图表，按 Esc 退出专注模式">
              <svg viewBox="0 0 20 20" aria-hidden="true"><path d={chartFocused ? 'M3 7h4V3m6 0v4h4M3 13h4v4m6 0v-4h4' : 'M7 3H3v4m10-4h4v4M3 13v4h4m10-4v4h-4'} /></svg>
              <span>{chartFocused ? '退出专注' : '专注看盘'}</span>
            </button>
            </div>
          </div>

          <div className="hero-row">
            <div className="price-overview">
              <p className={`hero-price num ${tone}`}>{hero == null ? '—' : formatPrice(hero, digits)}<span className="price-unit">{quoteAsset(market.selected)}</span></p>
              <p className={`chg ${tone}`}>
                <strong className="num">{change == null ? '—' : formatSigned(change, digits)}</strong>
                <span className="num">{pct == null ? '—' : formatPct(pct)}</span>
                <em>24 小时</em>
              </p>
            </div>
            <div className="market-summary">
              <dl className="session-stats">
                <div><dt>24h 最高</dt><dd className="num">{quote ? formatPrice(quote.high, digits) : '—'}</dd></div>
                <div><dt>24h 最低</dt><dd className="num">{quote ? formatPrice(quote.low, digits) : '—'}</dd></div>
                <div><dt>24h 成交额</dt><dd className="num">{quote ? formatCompact(quote.quoteVolume) : '—'}</dd></div>
              </dl>
              <div className="session-range" title="现价在 24 小时最高与最低之间的位置">
                <span>日内位置</span><div className="range-track" aria-hidden="true"><i style={{ left: `${Math.min(100, Math.max(0, rangePos ?? 0))}%`, visibility: rangePos === null ? 'hidden' : 'visible' }} /></div>
              </div>
            </div>
          </div>

          <div className="chart-toolbar">
            <div className="toggles" role="group" aria-label="图表叠加指标">
              <span className="toolbar-label">指标</span>
              <button type="button" className={showEma ? 'on' : ''} aria-pressed={showEma} title="EMA20 看近 20 根，拐得快。EMA50 看近 50 根，拐得慢。价格和快线都在慢线上方，这段偏多。" onClick={() => setShowEma((value) => !value)}>
                EMA
              </button>
              <button type="button" className={showBands ? 'on' : ''} aria-pressed={showBands} title="中间是 20 根收盘的平均，上下各 2 倍波动。贴着轨道只说明离平均远，不是到顶或到底。" onClick={() => setShowBands((value) => !value)}>
                布林
              </button>
              <button type="button" className={showForecast ? 'on' : ''} aria-pressed={showForecast} title="从最后一根往右画的未来 12 根，按近 30 根斜率外推，不是已经发生的价格，也不是目标价。" onClick={() => setShowForecast((value) => !value)}>
                漂移区间
              </button>
              <span className="toolbar-divider" aria-hidden="true" />
              <button type="button" className={showModel ? 'on' : ''} aria-pressed={showModel} title="根据已收盘 K 线逐步估计未来 12 根；蓝线是均值，接近水平不代表没有波动。" onClick={() => setShowModel((value) => !value)}>
                模型
              </button>
              <button type="button" className={showModelBand && showModel ? 'on' : ''} disabled={!showModel} aria-pressed={showModelBand} title="独立校准数据形成的 80% 经验误差区间，不保证未来覆盖率。" onClick={() => setShowModelBand((value) => !value)}>误差带</button>
              <button type="button" className={showModelHistory && showModel ? 'on' : ''} disabled={!showModel} aria-pressed={showModelHistory} title={modelEngine === 'b4' ? '选择一个历史收盘起点，一次预测后续12根，并和真实收盘及复制起点的基线比较；每根一个节点。' : 'B3 旧版滚动一步：每根拿新的真实收盘价重新预测下一根。'} onClick={() => setShowModelHistory((value) => !value)}>{modelEngine === 'b4' ? '同起点回放' : '滚动一步'}</button>
            </div>
          <div className="model-engine" role="group" aria-label="预测模型版本">
            <span>预测模型</span>
            <button type="button" className={modelEngine === 'b4' ? 'on' : ''} aria-pressed={modelEngine === 'b4'} onClick={() => setModelEngine('b4')}>B4 方向与幅度</button>
            <button type="button" className={modelEngine === 'b3' ? 'on' : ''} aria-pressed={modelEngine === 'b3'} onClick={() => setModelEngine('b3')}>B3 对照</button>
            <label className={modelEngine === 'b3' ? 'scenario-disabled' : ''}><input type="checkbox" checked={showScenarios} disabled={modelEngine === 'b3'} onChange={(event) => setShowScenarios(event.target.checked)} />幅度情景</label>
          </div>
          </div>
          {modelEngine === 'b4' && showModel && showModelHistory && <div className="replay-controls" aria-label="历史固定起点回放">
            <div><strong>固定起点 → 未来 12 根</strong><span>{replay ? `${replayTime} 收盘后 · 北京时间` : '等待完整、已收盘的回放窗口'}</span></div>
            <button type="button" onClick={() => selectReplay(replayIndex - 1)} disabled={replayIndex <= 0} aria-label="回放起点前移一根">‹</button>
            <input type="range" aria-label="选择历史预测起点" min={0} max={Math.max(0, replayPaths.length - 1)} value={Math.max(0, replayIndex)} step={1} disabled={!replayPaths.length} onChange={(event) => selectReplay(Number(event.target.value))} />
            <button type="button" onClick={() => selectReplay(replayIndex + 1)} disabled={replayIndex < 0 || replayIndex >= replayPaths.length - 1} aria-label="回放起点后移一根">›</button>
            <button type="button" onClick={() => selectReplay(replayIndex)} disabled={!replay}>定位回放</button>
          </div>}
          <div className="chart-stage">
            <ChartPane
              symbol={market.selected}
              interval={market.timeframe}
              candles={market.candles}
              forecast={analysis?.forecast ?? []}
              modelPath={model?.path ?? []}
              modelHistory={showModelHistory ? modelHistory : []}
              modelHistoryKind={modelEngine === 'b4' ? 'fixed-origin' : 'rolling-next'}
              replayFocus={replayFocus}
              modelLabel={modelEngine === 'b4' ? 'guanchao-b4' : 'guanchao-b3'}
              modelScenariosVisible={modelEngine === 'b4' && showScenarios}
              modelBandVisible={showModelBand}
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
              </div>
            )}
          </div>
          <div className="chart-footnote">
            <span><i className="interaction-dot" />滚轮缩放 · 拖动浏览 · 双击价格轴适配</span>
            <details className="chart-help"><summary>图表说明</summary><div>
              <strong>读懂图表</strong>
              <p>粗矩形表示开盘到收盘，细线表示本根最高和最低。铜金线为 EMA20，海绿线为 EMA50；灰色虚线为布林轨道。</p>
              <p>{HORIZON_LABEL[market.timeframe]}。蓝线是当前价格均值估计；B4 紫线固定一个历史收盘起点，一次预测12根，金线为真实收盘，灰线复制起点作基线。绿色虚线是幅度情景。B3 保留的滚动一步会每根重新起算，不能当作12步路径精度。</p>
              <p>拖动时纵轴固定。点击“适应价格”或双击右侧价格轴重新适配，点击“定位预测”返回最新区域。</p>
            </div></details>
          </div>
        </section>

        <SignalDeck
          modelEngine={modelEngine}
          b4={market.candleState === 'loading' ? { ...b4, model: null, status: 'loading', message: '正在加载当前周期的行情…' } : b4}
          analysis={analysis}
          model={neural.model}
          modelHistory={neural.history}
          modelStatus={market.candleState === 'loading' ? 'loading' : neural.status}
          modelMessage={market.candleState === 'loading' ? undefined : neural.message}
          onModelRetry={neural.retry}
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
          <span>BINANCE · 公开现货行情</span><span>只读观察 · 历史回放不代表未来表现</span>
          <span>K 线由 <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">TradingView</a> Lightweight Charts 绘制</span>
        </p>
      </footer>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
  TickMarkType,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import { overlaySeries, type ForecastPoint, type LinePoint } from '../lib/analysis'
import { ForecastBand } from '../lib/forecast-band'
import type { B4Point, B4HistoryPoint } from '../lib/b4'
import { formatCompact, formatPct, formatPrice } from '../lib/format'
import { HORIZON_LABEL, INTERVAL_MS, type Candle, type Interval } from '../types'

export type ChartTheme = 'dark' | 'light'

const PALETTE = {
  dark: {
    panel: '#141820',
    text: '#89909d',
    grid: 'rgba(244, 239, 230, 0.045)',
    cross: 'rgba(228, 181, 131, 0.35)',
    label: '#2a241c',
    border: 'rgba(244, 239, 230, 0.08)',
    up: '#e4b583',
    down: '#d89298',
    sea: '#8ec3b8',
    band: 'rgba(183, 178, 168, 0.55)',
    fore: 'rgba(228, 181, 131, 0.92)',
    foreBand: 'rgba(228, 181, 131, 0.38)',
    model: '#8eb4d4',
    modelBand: 'rgba(142, 180, 212, 0.55)',
    modelFill: 'rgba(142, 180, 212, 0.12)',
    history: '#b8a0d8',
    volUp: 'rgba(228, 181, 131, 0.38)',
    volDown: 'rgba(216, 146, 152, 0.36)',
  },
  light: {
    panel: '#ffffff',
    text: '#6d645b',
    grid: 'rgba(28, 24, 20, 0.06)',
    cross: 'rgba(154, 98, 48, 0.38)',
    label: '#2a241c',
    border: 'rgba(28, 24, 20, 0.1)',
    up: '#c17a3a',
    down: '#c45d68',
    sea: '#2f7d72',
    band: 'rgba(109, 100, 91, 0.5)',
    fore: 'rgba(154, 98, 48, 0.9)',
    foreBand: 'rgba(154, 98, 48, 0.38)',
    model: '#2f6394',
    modelBand: 'rgba(47, 99, 148, 0.45)',
    modelFill: 'rgba(47, 99, 148, 0.10)',
    history: '#8357a4',
    volUp: 'rgba(193, 122, 58, 0.36)',
    volDown: 'rgba(196, 93, 104, 0.32)',
  },
} as const

type SeriesBag = {
  candle: ISeriesApi<'Candlestick'>
  volume: ISeriesApi<'Histogram'>
  ema20: ISeriesApi<'Line'>
  ema50: ISeriesApi<'Line'>
  upper: ISeriesApi<'Line'>
  lower: ISeriesApi<'Line'>
  mid: ISeriesApi<'Line'>
  high: ISeriesApi<'Line'>
  low: ISeriesApi<'Line'>
  model: ISeriesApi<'Line'>
  modelHigh: ISeriesApi<'Line'>
  modelLow: ISeriesApi<'Line'>
  scenarioUp: ISeriesApi<'Line'>
  scenarioDown: ISeriesApi<'Line'>
  history: ISeriesApi<'Line'>
  historyActual: ISeriesApi<'Line'>
  historyBaseline: ISeriesApi<'Line'>
}

export type ModelHistoryPoint = { time: number; predicted: number; actual: number; originTime: number; horizon?: number } & Partial<B4HistoryPoint>

type Props = {
  symbol: string
  interval: Interval
  candles: Candle[]
  forecast: ForecastPoint[]
  modelPath: (ForecastPoint & Partial<B4Point>)[]
  modelHistory?: ModelHistoryPoint[]
  modelHistoryKind?: 'rolling-next' | 'fixed-origin'
  replayFocus?: { originTime: number; requestId: number } | null
  modelLabel?: string
  modelBandVisible?: boolean
  modelScenariosVisible?: boolean
  showEma: boolean
  showBands: boolean
  showForecast: boolean
  showModel: boolean
  digits: number
  theme: ChartTheme
}

function asTime(seconds: number): UTCTimestamp {
  return seconds as UTCTimestamp
}

function priceFormat(digits: number) {
  return {
    type: 'price' as const,
    precision: digits,
    minMove: 10 ** -digits,
  }
}

function tickLabel(time: number, type: TickMarkType): string {
  const date = new Date(time * 1000)
  const zone = 'Asia/Shanghai'
  if (type === TickMarkType.Year) {
    return new Intl.DateTimeFormat('zh-CN', { timeZone: zone, year: 'numeric' }).format(date)
  }
  if (type === TickMarkType.Month) {
    return new Intl.DateTimeFormat('zh-CN', { timeZone: zone, month: 'short' }).format(date)
  }
  if (type === TickMarkType.DayOfMonth) {
    return new Intl.DateTimeFormat('zh-CN', { timeZone: zone, month: '2-digit', day: '2-digit' }).format(date)
  }
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: zone,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date)
}

function toLine(points: LinePoint[]) {
  return points.map((point) => ({ time: asTime(point.time), value: point.value }))
}

// Keep stable timestamps in place. setData is needed only for a new window or identity;
// live ticks update changed points, and unchanged model paths perform no chart writes.
function dataWriter<Data extends { time: Time }>(series: {
  setData: (data: Data[]) => void
  update: (data: Data, historicalUpdate?: boolean) => void
}) {
  let previous: Data[] = []
  return (next: Data[], reset = false) => {
    if (!reset && previous.length === 0 && next.length === 0) return
    const sameTimeline = previous.length > 0
      && next.length >= previous.length
      && previous.every((point, index) => point.time === next[index].time)
    if (reset || !sameTimeline) {
      series.setData(next)
    } else {
      next.forEach((point, index) => {
        const before = previous[index]
        if (!before || Object.keys(point).some((key) => (
          point[key as keyof Data] !== before[key as keyof Data]
        ))) {
          series.update(point, index < previous.length - 1)
        }
      })
    }
    previous = next
  }
}

function dataWriters(series: SeriesBag) {
  return {
    candle: dataWriter(series.candle),
    volume: dataWriter(series.volume),
    ema20: dataWriter(series.ema20),
    ema50: dataWriter(series.ema50),
    upper: dataWriter(series.upper),
    lower: dataWriter(series.lower),
    mid: dataWriter(series.mid),
    high: dataWriter(series.high),
    low: dataWriter(series.low),
    model: dataWriter(series.model),
    modelHigh: dataWriter(series.modelHigh),
    modelLow: dataWriter(series.modelLow),
    scenarioUp: dataWriter(series.scenarioUp),
    scenarioDown: dataWriter(series.scenarioDown),
    history: dataWriter(series.history),
    historyActual: dataWriter(series.historyActual),
    historyBaseline: dataWriter(series.historyBaseline),
  }
}

function timeLabel(time: number): string {
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).format(new Date(time * 1000))
}

const NO_HISTORY: ModelHistoryPoint[] = []
const NO_FORECAST: ForecastPoint[] = []

export function ChartPane({
  symbol,
  interval,
  candles,
  forecast,
  modelPath,
  modelHistory = NO_HISTORY,
  modelHistoryKind = 'rolling-next',
  replayFocus = null,
  modelLabel = 'guanchao-b3',
  modelBandVisible = true,
  modelScenariosVisible = false,
  showEma,
  showBands,
  showForecast,
  showModel,
  digits,
  theme,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<SeriesBag | null>(null)
  const writersRef = useRef<ReturnType<typeof dataWriters> | null>(null)
  const bandRef = useRef<ForecastBand | null>(null)
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null)
  const identityRef = useRef('')
  const firstTimeRef = useRef<number | null>(null)
  const [hover, setHover] = useState<{ time: number; identity: string } | null>(null)
  const [priceLocked, setPriceLocked] = useState(false)
  const identity = `${symbol}|${interval}`
  const identityNow = useRef(identity)
  const hasCandles = candles.length > 0
  const hasForecast = hasCandles && showForecast && forecast.length > 1
  const hasModel = hasCandles && showModel && modelPath.length > 1
  const hasHistory = hasCandles && showModel && modelHistory.length > 0
  const fixedReplay = modelHistoryKind === 'fixed-origin'
  const focusedReplayRef = useRef<object | null>(null)
  const hasScenarios = hasModel && modelScenariosVisible && modelPath.every((point) => point.upScenario !== undefined && point.downScenario !== undefined)

  useEffect(() => {
    identityNow.current = identity
  }, [identity])

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const look = PALETTE[document.documentElement.dataset.theme === 'light' ? 'light' : 'dark']
    const chart = createChart(host, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: look.panel },
        textColor: look.text,
        fontFamily: 'Outfit, "Noto Sans SC", sans-serif',
        fontSize: 12,
        attributionLogo: true,
      },
      grid: {
        vertLines: { color: look.grid },
        horzLines: { color: look.grid },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: look.cross, labelBackgroundColor: look.label },
        horzLine: { color: look.cross, labelBackgroundColor: look.label },
      },
      rightPriceScale: {
        borderColor: look.border,
        scaleMargins: { top: 0.08, bottom: 0.22 },
        minimumWidth: 88,
      },
      // Scrolling changes the time window, not the scale of the candles.
      kineticScroll: { mouse: false, touch: false },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
      handleScale: { axisDoubleClickReset: { time: true, price: false } },
      timeScale: {
        borderColor: look.border,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 6,
        barSpacing: 7,
        shiftVisibleRangeOnNewBar: false,
        lockVisibleTimeRangeOnResize: true,
        tickMarkFormatter: (time: Time, type: TickMarkType) => (
          typeof time === 'number' ? tickLabel(time, type) : ''
        ),
      },
      localization: {
        locale: 'zh-CN',
        timeFormatter: (time: Time) => (
          typeof time === 'number'
            ? new Intl.DateTimeFormat('zh-CN', {
              timeZone: 'Asia/Shanghai',
              month: '2-digit',
              day: '2-digit',
              hour: '2-digit',
              minute: '2-digit',
              hour12: false,
            }).format(new Date(time * 1000))
            : ''
        ),
      },
    })

    const candle = chart.addSeries(CandlestickSeries, {
      upColor: look.up,
      downColor: look.down,
      borderUpColor: look.up,
      borderDownColor: look.down,
      wickUpColor: look.up,
      wickDownColor: look.down,
      priceLineWidth: 1,
    })
    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      lastValueVisible: false,
      priceLineVisible: false,
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
      visible: false,
    })
    const quiet = {
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      // Indicators and forecasts must not squeeze the candle price scale when
      // their farthest point enters or leaves the visible time window.
      autoscaleInfoProvider: () => null,
    }
    const ema20 = chart.addSeries(LineSeries, { color: look.up, lineWidth: 2, ...quiet })
    const ema50 = chart.addSeries(LineSeries, { color: look.sea, lineWidth: 1, ...quiet })
    const upper = chart.addSeries(LineSeries, { color: look.band, lineWidth: 1, lineStyle: LineStyle.Dashed, ...quiet })
    const lower = chart.addSeries(LineSeries, { color: look.band, lineWidth: 1, lineStyle: LineStyle.Dashed, ...quiet })
    const mid = chart.addSeries(LineSeries, {
      color: look.fore,
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      ...quiet,
    })
    const high = chart.addSeries(LineSeries, {
      color: look.foreBand,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      ...quiet,
      autoscaleInfoProvider: () => null,
    })
    const low = chart.addSeries(LineSeries, {
      color: look.foreBand,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      ...quiet,
      autoscaleInfoProvider: () => null,
    })
    const model = chart.addSeries(LineSeries, {
      color: look.model,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      pointMarkersVisible: true,
      pointMarkersRadius: 2,
      ...quiet,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
    })
    const modelHigh = chart.addSeries(LineSeries, {
      color: look.modelBand,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      ...quiet,
      autoscaleInfoProvider: () => null,
    })
    const modelLow = chart.addSeries(LineSeries, {
      color: look.modelBand,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      ...quiet,
      autoscaleInfoProvider: () => null,
    })
    const history = chart.addSeries(LineSeries, {
      color: look.history,
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      pointMarkersVisible: true,
      pointMarkersRadius: 2,
      ...quiet,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
    })
    const scenarioUp = chart.addSeries(LineSeries, { color: look.sea, lineWidth: 1, lineStyle: LineStyle.Dashed, ...quiet })
    const scenarioDown = chart.addSeries(LineSeries, { color: look.sea, lineWidth: 1, lineStyle: LineStyle.Dashed, ...quiet })
    const historyActual = chart.addSeries(LineSeries, { color: look.up, lineWidth: 2, ...quiet })
    const historyBaseline = chart.addSeries(LineSeries, { color: look.text, lineWidth: 1, lineStyle: LineStyle.Dotted, ...quiet })
    const band = new ForecastBand()
    model.attachPrimitive(band)
    bandRef.current = band
    markersRef.current = createSeriesMarkers(model, [], { autoScale: false })

    chartRef.current = chart
    seriesRef.current = { candle, volume, ema20, ema50, upper, lower, mid, high, low, model, modelHigh, modelLow, scenarioUp, scenarioDown, history, historyActual, historyBaseline }
    writersRef.current = dataWriters(seriesRef.current)
    const onMove = (param: { time?: Time }) => {
      if (typeof param.time !== 'number') {
        setHover(null)
        return
      }
      setHover({ time: param.time, identity: identityNow.current })
    }
    chart.subscribeCrosshairMove(onMove)
    const lockPrice = () => {
      const scale = chart.priceScale('right')
      if (!scale.getVisibleRange()) return
      scale.setAutoScale(false)
      setPriceLocked(true)
    }
    const resetPriceOnDoubleClick = (event: MouseEvent) => {
      const bounds = host.getBoundingClientRect()
      const inPriceAxis = event.clientX - bounds.left >= bounds.width - chart.priceScale('right').width()
      const aboveTimeAxis = event.clientY - bounds.top < bounds.height - chart.timeScale().height()
      if (inPriceAxis && aboveTimeAxis) {
        chart.priceScale('right').setAutoScale(true)
        setPriceLocked(false)
      }
    }
    // Capture runs before the chart's wheel/drag handler can autoscale. Once
    // browsing, retain this scale through live ticks and asynchronous forecasts.
    host.addEventListener('pointerdown', lockPrice, true)
    host.addEventListener('wheel', lockPrice, { capture: true, passive: true })
    host.addEventListener('dblclick', resetPriceOnDoubleClick, true)
    return () => {
      host.removeEventListener('pointerdown', lockPrice, true)
      host.removeEventListener('wheel', lockPrice, true)
      host.removeEventListener('dblclick', resetPriceOnDoubleClick, true)
      chart.unsubscribeCrosshairMove(onMove)
      model.detachPrimitive(band)
      markersRef.current?.detach()
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      writersRef.current = null
      bandRef.current = null
      markersRef.current = null
      identityRef.current = ''
      firstTimeRef.current = null
      focusedReplayRef.current = null
    }
  }, [])

  useEffect(() => {
    const chart = chartRef.current
    const series = seriesRef.current
    if (!chart || !series) return
    const look = PALETTE[theme]
    chart.applyOptions({
      layout: {
        background: { type: ColorType.Solid, color: look.panel },
        textColor: look.text,
      },
      grid: {
        vertLines: { color: look.grid },
        horzLines: { color: look.grid },
      },
      crosshair: {
        vertLine: { color: look.cross, labelBackgroundColor: look.label },
        horzLine: { color: look.cross, labelBackgroundColor: look.label },
      },
      rightPriceScale: { borderColor: look.border },
      timeScale: {
        borderColor: look.border,
        timeVisible: interval !== '1d',
        secondsVisible: false,
      },
    })
    const format = priceFormat(digits)
    series.candle.applyOptions({
      priceFormat: format,
      upColor: look.up,
      downColor: look.down,
      borderUpColor: look.up,
      borderDownColor: look.down,
      wickUpColor: look.up,
      wickDownColor: look.down,
    })
    series.ema20.applyOptions({ visible: showEma, priceFormat: format, color: look.up })
    series.ema50.applyOptions({ visible: showEma, priceFormat: format, color: look.sea })
    series.upper.applyOptions({ visible: showBands, priceFormat: format, color: look.band })
    series.lower.applyOptions({ visible: showBands, priceFormat: format, color: look.band })
    series.mid.applyOptions({ visible: hasForecast, priceFormat: format, color: look.fore })
    series.high.applyOptions({ visible: hasForecast, priceFormat: format, color: look.foreBand })
    series.low.applyOptions({ visible: hasForecast, priceFormat: format, color: look.foreBand })
    series.model.applyOptions({ visible: hasModel, priceFormat: format, color: look.model })
    series.modelHigh.applyOptions({ visible: hasModel && modelBandVisible, priceFormat: format, color: look.modelBand })
    series.modelLow.applyOptions({ visible: hasModel && modelBandVisible, priceFormat: format, color: look.modelBand })
    series.history.applyOptions({ visible: hasHistory, priceFormat: format, color: look.history })
    series.historyActual.applyOptions({ visible: hasHistory && fixedReplay, priceFormat: format, color: look.up })
    series.historyBaseline.applyOptions({ visible: hasHistory && fixedReplay, priceFormat: format, color: look.text })
    series.scenarioUp.applyOptions({ visible: hasScenarios, priceFormat: format, color: look.sea })
    series.scenarioDown.applyOptions({ visible: hasScenarios, priceFormat: format, color: look.sea })
  }, [digits, fixedReplay, hasForecast, hasHistory, hasModel, hasScenarios, interval, modelBandVisible, showBands, showEma, theme])

  useEffect(() => {
    const chart = chartRef.current
    const writers = writersRef.current
    if (!chart || !writers) return
    if (!candles.length) {
      Object.values(writers).forEach((write) => write([]))
      bandRef.current?.setData(NO_FORECAST, PALETTE[theme].modelFill, false)
      identityRef.current = ''
      firstTimeRef.current = null
      return
    }

    const look = PALETTE[theme]
    const reset = identityRef.current !== identity
    const range = chart.timeScale().getVisibleLogicalRange()
    const oldFirst = firstTimeRef.current
    const overlays = overlaySeries(candles)
    const bars = candles.map((candle) => ({
      time: asTime(candle.time),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }))
    const volumes = candles.map((candle) => ({
      time: asTime(candle.time),
      value: candle.volume,
      color: candle.close >= candle.open ? look.volUp : look.volDown,
    }))
    const visibleModel = hasModel ? modelPath : NO_FORECAST
    const visibleBand = modelBandVisible ? visibleModel : NO_FORECAST
    writers.candle(bars, reset)
    writers.volume(volumes, reset)
    // Keep overlays current while hidden so re-enabling EMA never leaves a stale segment.
    writers.ema20(toLine(overlays.ema20), reset)
    writers.ema50(toLine(overlays.ema50), reset)
    writers.upper(toLine(overlays.upper), reset)
    writers.lower(toLine(overlays.lower), reset)
    // Keep the already-computed drift data current while hidden, just like EMA.
    // Its fixed twelve-bar horizon keeps the time axis stable while the async
    // model clears/reappears; visibility alone controls what is actually drawn.
    writers.mid(forecast.map((point) => ({ time: asTime(point.time), value: point.mid })), reset)
    writers.high(forecast.map((point) => ({ time: asTime(point.time), value: point.upper })), reset)
    writers.low(forecast.map((point) => ({ time: asTime(point.time), value: point.lower })), reset)
    writers.model(visibleModel.map((point) => ({ time: asTime(point.time), value: point.mid })), reset)
    writers.modelHigh(visibleBand.map((point) => ({ time: asTime(point.time), value: point.upper })), reset)
    writers.modelLow(visibleBand.map((point) => ({ time: asTime(point.time), value: point.lower })), reset)
    writers.scenarioUp(hasScenarios ? modelPath.map((point) => ({ time: asTime(point.time), value: point.upScenario! })) : [], reset)
    writers.scenarioDown(hasScenarios ? modelPath.map((point) => ({ time: asTime(point.time), value: point.downScenario! })) : [], reset)
    writers.history(hasHistory ? modelHistory.map((point) => ({
      time: asTime(point.time), value: point.predicted,
    })) : [], reset)
    writers.historyActual(hasHistory && fixedReplay ? modelHistory.map((point) => ({ time: asTime(point.time), value: point.actual })) : [], reset)
    writers.historyBaseline(hasHistory && fixedReplay ? modelHistory.map((point) => ({ time: asTime(point.time), value: point.baseline! })) : [], reset)
    bandRef.current?.setData(visibleModel, look.modelFill, hasModel && modelBandVisible)

    if (reset) {
      chart.priceScale('right').setAutoScale(true)
      setPriceLocked(false)
      const count = candles.length
      chart.timeScale().setVisibleLogicalRange({
        from: Math.max(0, count - 132),
        // Model weights arrive asynchronously; reserve the requested horizon on
        // first load so the new path is not cropped to the last few candles.
        to: count + (hasForecast || showModel ? 14 : 3),
      })
    } else if (range && oldFirst !== null && oldFirst !== candles[0].time) {
      // Only a changed historical window needs index compensation. Restoring a
      // snapshot on live ticks/model refreshes competes with an active gesture
      // and stops the chart's own scrolling/zooming transition.
      const removed = oldFirst === null ? 0 : (candles[0].time - oldFirst) / (INTERVAL_MS[interval] / 1000)
      const keep = { from: range.from - removed, to: range.to - removed }
      const after = chart.timeScale().getVisibleLogicalRange()
      if (!after || Math.abs(after.from - keep.from) > 0.001 || Math.abs(after.to - keep.to) > 0.001) {
        chart.timeScale().setVisibleLogicalRange(keep)
      }
    }
    identityRef.current = identity
    firstTimeRef.current = candles[0].time
  }, [candles, fixedReplay, forecast, hasForecast, hasHistory, hasModel, hasScenarios, identity, interval, modelBandVisible, modelHistory, modelPath, showModel, theme])

  useEffect(() => {
    if (!replayFocus || focusedReplayRef.current === replayFocus || !hasHistory || !fixedReplay) return
    const index = candles.findIndex((candle) => candle.time === replayFocus.originTime)
    const chart = chartRef.current
    if (!chart || index < 0) return
    // Navigation is an explicit user action. Data refreshes never refocus it.
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, index - 12), to: index + 26 })
    chart.priceScale('right').setAutoScale(true)
    setPriceLocked(false)
    focusedReplayRef.current = replayFocus
  }, [candles, fixedReplay, hasHistory, replayFocus])

  useEffect(() => {
    const first = hasModel ? modelPath[0] : null
    const last = hasModel ? modelPath.at(-1) : null
    markersRef.current?.setMarkers(first && last ? [
      { time: asTime(first.time), position: 'inBar', color: PALETTE[theme].model, shape: 'circle', size: 0.8, text: '起点' },
      { time: asTime(last.time), position: 'inBar', color: PALETTE[theme].model, shape: 'circle', size: 0.8, text: '+12' },
    ] : [])
  }, [hasModel, modelPath, theme])

  const hoverTime = hover?.identity === identity ? hover.time : null
  const hoverCandle = hoverTime === null ? null : candles.find((item) => item.time === hoverTime)
  const hoverForecastIndex = hasModel ? modelPath.findIndex((item) => item.time === hoverTime) : -1
  const hoverForecast = hoverForecastIndex >= 0 ? modelPath[hoverForecastIndex] : null
  const hoverHistory = hasHistory ? modelHistory.find((item) => item.time === hoverTime) : null
  const historyError = hoverHistory && hoverHistory.actual > 0
    ? (hoverHistory.predicted / hoverHistory.actual - 1) * 100 : null
  const future = hasForecast || hasModel

  const fitPrice = () => {
    chartRef.current?.priceScale('right').setAutoScale(true)
    setPriceLocked(false)
  }

  const focusLatest = () => {
    const chart = chartRef.current
    if (!chart || !hasCandles) return
    const range = chart.timeScale().getVisibleLogicalRange()
    const width = range ? range.to - range.from : 146
    const to = candles.length + (future ? 14 : 3)
    chart.timeScale().setVisibleLogicalRange({ from: to - width, to })
    fitPrice()
  }

  return (
    <>
      <div ref={hostRef} className="chart-canvas" />
      {(hasModel || hasHistory) && (
        <div className="model-legend" aria-label="模型图例">
          {hasModel && <span title="这条线是价格点估计；当前模型尚未证明稳定的方向预测优势。"><i className="swatch model-path" />{modelLabel} · 12 根参考估计</span>}
          {hasModel && modelBandVisible && <span><i className="swatch model-fill" />误差区间</span>}
          {hasScenarios && <span title="预测的绝对幅度分别向上、向下换算的情景，非目标价格或置信区间。"><i className="swatch model-scenario" />上下行情景 · 非目标价</span>}
          {hasHistory && <span title={fixedReplay ? '固定同一个历史收盘起点，一次预测后续 12 根；不使用后来真实价格重新起算。' : '每根使用新的真实收盘价重设起点，只预测下一根，不能代表一次预测未来 12 根的准确性。'}><i className="swatch model-history" />{fixedReplay ? '同一起点 · 12 步回放' : '滚动 1 步 · 每根重设起点'}</span>}
          {hasHistory && fixedReplay && <><span><i className="swatch" />实际收盘</span><span><i className="swatch replay-baseline" />持平基线 · 复制起点</span></>}
        </div>
      )}
      {hasCandles && <div className={`ohlc${hoverForecast || hoverHistory ? ' model-readout' : ''}`}>
        {hoverForecast && (hoverForecastIndex > 0 || !hoverHistory) ? (
          <>
            <strong className="readout-title">{hoverForecastIndex === 0 ? '预测起点 · 已闭合 K 线' : hoverForecastIndex === 1 ? `${modelLabel} · 下一根` : `${modelLabel} · +${hoverForecastIndex} 根`}</strong>
            <span className="readout-time">{timeLabel(hoverForecast.time)} · 北京时间</span>
            <span>{hoverForecastIndex === 0 ? '收盘' : '点估计'} <b className="num">{formatPrice(hoverForecast.mid, digits)}</b></span>
            {hoverForecastIndex > 0 && <span>相对起点 <b className="num">{((hoverForecast.mid / modelPath[0].mid - 1) * 100).toFixed(4)}%</b></span>}
            {hoverForecastIndex > 0 && modelBandVisible && <span>区间 <b className="num">{formatPrice(hoverForecast.lower, digits)} – {formatPrice(hoverForecast.upper, digits)}</b></span>}
            {hoverForecastIndex > 0 && hoverForecast.expectedMovePct !== undefined && <span>预期幅度 <b className="num">{hoverForecast.expectedMovePct.toFixed(3)}%</b></span>}
            {hoverForecastIndex > 0 && hoverForecast.upProbability !== undefined && <span>上涨概率 <b className="num">{(hoverForecast.upProbability * 100).toFixed(1)}%</b></span>}
            {hoverForecastIndex > 0 && hasScenarios && <span>下行／上行情景 <b className="num">{formatPrice(hoverForecast.downScenario!, digits)} / {formatPrice(hoverForecast.upScenario!, digits)}</b></span>}
          </>
        ) : hoverHistory ? (
          <>
            <strong className="readout-title history-color">{fixedReplay ? `同一起点回放 · +${hoverHistory.horizon ?? 0} 根` : '滚动 1 步 · 下一根'}</strong>
            <span className="readout-time">起点收盘 {timeLabel(hoverHistory.originTime + INTERVAL_MS[interval] / 1000)} · 目标收盘 {timeLabel(hoverHistory.time + INTERVAL_MS[interval] / 1000)} · 北京时间</span>
            <span>预测 <b className="num">{formatPrice(hoverHistory.predicted, digits)}</b></span>
            <span>实际 <b className="num">{formatPrice(hoverHistory.actual, digits)}</b></span>
            {fixedReplay && <span>持平基线 <b className="num">{formatPrice(hoverHistory.baseline!, digits)}</b></span>}
            {historyError !== null && <span>偏差 <b className="num">{formatPct(historyError)}</b></span>}
            {hoverHistory.upProbability !== undefined && <span>预测上涨概率 <b className="num">{(hoverHistory.upProbability * 100).toFixed(1)}%</b></span>}
            {hoverHistory.expectedMovePct !== undefined && <span>预测／实际幅度 <b className="num">{hoverHistory.expectedMovePct.toFixed(3)}% / {hoverHistory.actualMovePct?.toFixed(3)}%</b></span>}
            <span className="readout-caption">{fixedReplay ? '同一起点一次预测未来 12 根；期间不更新输入' : '上一根收盘时预测本根'}；偏差 = (预测 − 实际) / 实际</span>
          </>
        ) : hoverCandle ? (
          <>
            <span>开 <b className="num">{formatPrice(hoverCandle.open, digits)}</b></span>
            <span>高 <b className="num">{formatPrice(hoverCandle.high, digits)}</b></span>
            <span>低 <b className="num">{formatPrice(hoverCandle.low, digits)}</b></span>
            <span className={hoverCandle.close >= hoverCandle.open ? 'up' : 'down'}>收 <b className="num">{formatPrice(hoverCandle.close, digits)}</b></span>
            <span>量 <b className="num">{formatCompact(hoverCandle.volume)}</b></span>
          </>
        ) : (
          <span>{hasModel || hasHistory ? '悬停查看 K 线、预测与回放误差' : '停在 K 线上，看这一根的开高低收'}</span>
        )}
      </div>}
      {hasCandles && <div className="chart-controls" role="group" aria-label="图表视图">
        <button type="button" onClick={focusLatest}>{future ? '定位预测' : '回到最新'}</button>
        <button type="button" onClick={fitPrice} title="按当前可见 K 线重新适配价格轴；下一次拖动或缩放会固定纵轴比例。">适应价格</button>
        <span className="chart-scale-state" title="拖动或滚轮缩放会固定纵轴，避免 K 线忽大忽小。右侧价格轴仍可拖动调整。">{priceLocked ? '纵轴已固定' : '纵轴自动'}</span>
      </div>}
      {future && <div className="future-tag">{HORIZON_LABEL[interval]}</div>}
    </>
  )
}

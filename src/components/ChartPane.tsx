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
  type IChartApi,
  type ISeriesApi,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import { overlaySeries, type ForecastPoint, type LinePoint } from '../lib/analysis'
import { formatCompact, formatPrice } from '../lib/format'
import { HORIZON_LABEL, type Candle, type Interval } from '../types'

export type ChartTheme = 'dark' | 'light'

const PALETTE = {
  dark: {
    panel: '#14171e',
    text: '#a3988c',
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
}

type Props = {
  symbol: string
  interval: Interval
  candles: Candle[]
  forecast: ForecastPoint[]
  modelPath: ForecastPoint[]
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

export function ChartPane({
  symbol,
  interval,
  candles,
  forecast,
  modelPath,
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
  const identityRef = useRef('')
  const themeRef = useRef(theme)
  const candlesRef = useRef(candles)
  const [hover, setHover] = useState<Candle | null>(null)
  candlesRef.current = candles

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
        mode: CrosshairMode.Magnet,
        vertLine: { color: look.cross, labelBackgroundColor: look.label },
        horzLine: { color: look.cross, labelBackgroundColor: look.label },
      },
      rightPriceScale: {
        borderColor: look.border,
        scaleMargins: { top: 0.08, bottom: 0.22 },
      },
      timeScale: {
        borderColor: look.border,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 6,
        barSpacing: 7,
        shiftVisibleRangeOnNewBar: true,
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
      autoscaleInfoProvider: () => null,
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
      lineStyle: LineStyle.LargeDashed,
      ...quiet,
      autoscaleInfoProvider: () => null,
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

    chartRef.current = chart
    seriesRef.current = { candle, volume, ema20, ema50, upper, lower, mid, high, low, model, modelHigh, modelLow }
    const onMove = (param: { time?: Time }) => {
      if (typeof param.time !== 'number') {
        setHover(null)
        return
      }
      setHover(candlesRef.current.find((item) => item.time === param.time) ?? null)
    }
    chart.subscribeCrosshairMove(onMove)
    return () => {
      chart.unsubscribeCrosshairMove(onMove)
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      identityRef.current = ''
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
    series.mid.applyOptions({ visible: showForecast, priceFormat: format, color: look.fore })
    series.high.applyOptions({ visible: showForecast, priceFormat: format, color: look.foreBand })
    series.low.applyOptions({ visible: showForecast, priceFormat: format, color: look.foreBand })
    series.model.applyOptions({ visible: showModel, priceFormat: format, color: look.model })
    series.modelHigh.applyOptions({ visible: showModel, priceFormat: format, color: look.modelBand })
    series.modelLow.applyOptions({ visible: showModel, priceFormat: format, color: look.modelBand })

    if (!candles.length) {
      series.candle.setData([])
      series.volume.setData([])
      series.ema20.setData([])
      series.ema50.setData([])
      series.upper.setData([])
      series.lower.setData([])
      series.mid.setData([])
      series.high.setData([])
      series.low.setData([])
      series.model.setData([])
      series.modelHigh.setData([])
      series.modelLow.setData([])
      identityRef.current = ''
      return
    }

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
    const forecastMid = showForecast ? forecast.map((point) => ({ time: asTime(point.time), value: point.mid })) : []
    const forecastHigh = showForecast ? forecast.map((point) => ({ time: asTime(point.time), value: point.upper })) : []
    const forecastLow = showForecast ? forecast.map((point) => ({ time: asTime(point.time), value: point.lower })) : []
    const modelMid = showModel ? modelPath.map((point) => ({ time: asTime(point.time), value: point.mid })) : []
    const modelHigh = showModel ? modelPath.map((point) => ({ time: asTime(point.time), value: point.upper })) : []
    const modelLow = showModel ? modelPath.map((point) => ({ time: asTime(point.time), value: point.lower })) : []
    const extend = showForecast || showModel
    const identity = `${symbol}|${interval}`
    const themeChanged = themeRef.current !== theme

    if (identityRef.current !== identity) {
      series.candle.setData(bars)
      series.volume.setData(volumes)
      series.ema20.setData(toLine(overlays.ema20))
      series.ema50.setData(toLine(overlays.ema50))
      series.upper.setData(showBands ? toLine(overlays.upper) : [])
      series.lower.setData(showBands ? toLine(overlays.lower) : [])
      series.mid.setData(forecastMid)
      series.high.setData(forecastHigh)
      series.low.setData(forecastLow)
      series.model.setData(modelMid)
      series.modelHigh.setData(modelHigh)
      series.modelLow.setData(modelLow)
      const count = candles.length
      chart.timeScale().setVisibleLogicalRange({
        from: Math.max(0, count - 132),
        to: count + (extend ? 14 : 3),
      })
      identityRef.current = identity
      themeRef.current = theme
      return
    }

    if (themeChanged) {
      const range = chart.timeScale().getVisibleLogicalRange()
      series.volume.setData(volumes)
      themeRef.current = theme
      if (range) chart.timeScale().setVisibleLogicalRange(range)
    }

    const last = candles[candles.length - 1]
    series.candle.update(bars[bars.length - 1])
    series.volume.update(volumes[volumes.length - 1])
    const ema20Last = overlays.ema20.at(-1)
    const ema50Last = overlays.ema50.at(-1)
    if (showEma && ema20Last && ema20Last.time === last.time) series.ema20.update({ time: asTime(ema20Last.time), value: ema20Last.value })
    if (showEma && ema50Last && ema50Last.time === last.time) series.ema50.update({ time: asTime(ema50Last.time), value: ema50Last.value })
    if (showBands) {
      series.upper.setData(toLine(overlays.upper))
      series.lower.setData(toLine(overlays.lower))
    } else {
      series.upper.setData([])
      series.lower.setData([])
    }
    const range = chart.timeScale().getVisibleLogicalRange()
    series.mid.setData(forecastMid)
    series.high.setData(forecastHigh)
    series.low.setData(forecastLow)
    series.model.setData(modelMid)
    series.modelHigh.setData(modelHigh)
    series.modelLow.setData(modelLow)
    if (range && extend && range.to >= candles.length - 3) {
      chart.timeScale().setVisibleLogicalRange({ from: range.from, to: Math.max(range.to, candles.length + 12) })
    } else if (range) {
      chart.timeScale().setVisibleLogicalRange(range)
    }
  }, [candles, digits, forecast, interval, modelPath, showBands, showEma, showForecast, showModel, symbol, theme])

  const future = showForecast || showModel
  return (
    <>
      <div ref={hostRef} className="chart-canvas" />
      <div className="ohlc">
        {hover ? (
          <>
            <span>开 <b className="num">{formatPrice(hover.open, digits)}</b></span>
            <span>高 <b className="num">{formatPrice(hover.high, digits)}</b></span>
            <span>低 <b className="num">{formatPrice(hover.low, digits)}</b></span>
            <span className={hover.close >= hover.open ? 'up' : 'down'}>收 <b className="num">{formatPrice(hover.close, digits)}</b></span>
            <span>量 <b className="num">{formatCompact(hover.volume)}</b></span>
          </>
        ) : (
          <span>停在 K 线上，看这一根的开高低收</span>
        )}
      </div>
      {future && <div className="future-tag">{HORIZON_LABEL[interval]}</div>}
    </>
  )
}

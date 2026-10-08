import { useEffect, useRef, useState } from 'react'
import { bufferCandle, candlesNeedBackfill, fetchKlines, fetchTicker, fetchTickers, mergeCandle, openStream, toSeconds, ApiError } from '../lib/binance'
import { baseAsset, normalizeSymbol } from '../lib/format'
import {
  LEGACY_USDT_SYMBOLS,
  INTERVAL_MS,
  QUOTE_MARKETS,
  symbolsForQuote,
  type Candle,
  type Depth,
  type Interval,
  type LinkState,
  type Quote,
  type QuoteMarket,
} from '../types'

const STORAGE_KEY = 'guanchao.watch'
const MAX_SYMBOLS = 24

type AddResult = 'ok' | 'invalid' | 'network' | 'full'

type KlineEvent = {
  s?: string
  k?: {
    s?: string
    i?: string
    t: number
    o: string
    h: string
    l: string
    c: string
    v: string
    q?: string
    n?: number
    V?: string
    Q?: string
    x?: boolean
  }
}

function isQuoteMarket(value: string): value is QuoteMarket {
  return (QUOTE_MARKETS as readonly string[]).includes(value)
}

function loadWatch(): { symbols: string[]; selected: string; quote: QuoteMarket } {
  const fallback = { symbols: symbolsForQuote('USDT'), selected: 'BTCUSDT', quote: 'USDT' as const }
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return fallback
    const parsed = JSON.parse(raw) as { symbols?: unknown; selected?: unknown; quote?: unknown }
    const symbols = Array.isArray(parsed.symbols)
      ? parsed.symbols.filter((item): item is string => typeof item === 'string' && /^[A-Z0-9]{5,20}$/.test(item)).slice(0, MAX_SYMBOLS)
      : []
    if (symbols.join(',') === LEGACY_USDT_SYMBOLS.join(',')) return fallback
    const list = symbols.length ? symbols : fallback.symbols
    const selected = typeof parsed.selected === 'string' && list.includes(parsed.selected)
      ? parsed.selected
      : list[0]
    const quote = typeof parsed.quote === 'string' && isQuoteMarket(parsed.quote) ? parsed.quote : 'USDT'
    return { symbols: list, selected, quote }
  } catch {
    return fallback
  }
}

function upsert(prev: Quote | undefined, patch: Partial<Quote> & { symbol: string }): Quote {
  return {
    symbol: patch.symbol,
    last: patch.last ?? prev?.last ?? 0,
    open: patch.open ?? prev?.open ?? 0,
    high: patch.high ?? prev?.high ?? 0,
    low: patch.low ?? prev?.low ?? 0,
    volume: patch.volume ?? prev?.volume ?? 0,
    quoteVolume: patch.quoteVolume ?? prev?.quoteVolume ?? 0,
    count: patch.count && patch.count > 0 ? patch.count : prev?.count ?? 0,
  }
}

export function useMarket() {
  const initial = loadWatch()
  const [symbols, setSymbols] = useState(initial.symbols)
  const [selected, setSelected] = useState(initial.selected)
  const [quoteMarket, setQuoteMarket] = useState<QuoteMarket>(initial.quote)
  const [timeframe, setTimeframe] = useState<Interval>('1h')
  const [quotes, setQuotes] = useState<Record<string, Quote>>({})
  const [candles, setCandles] = useState<Candle[]>([])
  const [readyKey, setReadyKey] = useState('')
  const [errorKey, setErrorKey] = useState('')
  const [errorText, setErrorText] = useState<string | null>(null)
  const [depth, setDepth] = useState<(Depth & { symbol: string }) | null>(null)
  const [live, setLive] = useState<{ symbol: string; price: number } | null>(null)
  const [watchOpen, setWatchOpen] = useState(false)
  const [seenLive, setSeenLive] = useState(false)
  const [restError, setRestError] = useState<string | null>(null)
  const [clock, setClock] = useState(() => new Date())
  const [nonce, setNonce] = useState(0)

  const selectedRef = useRef(selected)
  const timeframeRef = useRef(timeframe)
  const requestKey = `${selected}|${timeframe}`

  useEffect(() => {
    selectedRef.current = selected
    timeframeRef.current = timeframe
  }, [selected, timeframe])

  useEffect(() => {
    const id = window.setInterval(() => setClock(new Date()), 1000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ symbols, selected, quote: quoteMarket }))
  }, [symbols, selected, quoteMarket])

  useEffect(() => {
    const ac = new AbortController()
    let stop = false
    const load = async () => {
      try {
        const next = await fetchTickers(symbols, ac.signal)
        if (stop) return
        setQuotes((prev) => {
          const merged = { ...prev }
          for (const [symbol, quote] of Object.entries(next)) merged[symbol] = upsert(prev[symbol], quote)
          return merged
        })
        setRestError(null)
      } catch (error) {
        if (stop || (error instanceof DOMException && error.name === 'AbortError')) return
        setRestError('行情列表暂时没有取到，正在重试。')
      }
    }
    void load()
    const id = window.setInterval(() => void load(), 45000)
    return () => {
      stop = true
      ac.abort()
      window.clearInterval(id)
    }
  }, [symbols, nonce])

  useEffect(() => {
    const streams = symbols.map((symbol) => `${symbol.toLowerCase()}@miniTicker`).join('/')
    const buffer: Record<string, Partial<Quote> & { symbol: string }> = {}
    let timer = 0
    const queue = (symbol: string, patch: Partial<Quote>) => {
      buffer[symbol] = { ...buffer[symbol], ...patch, symbol }
      if (timer) return
      timer = window.setTimeout(() => {
        timer = 0
        const batch = Object.values(buffer)
        for (const key of Object.keys(buffer)) delete buffer[key]
        setQuotes((prev) => {
          const next = { ...prev }
          for (const item of batch) next[item.symbol] = upsert(prev[item.symbol], item)
          return next
        })
        setRestError(null)
      }, 200)
    }
    const close = openStream(`/stream?streams=${streams}`, (raw) => {
      const message = raw as { data?: { s?: string; c?: string; o?: string; h?: string; l?: string; v?: string; q?: string } }
      const data = message.data
      if (!data?.s || !data.c) return
      queue(data.s, {
        last: Number(data.c),
        open: Number(data.o),
        high: Number(data.h),
        low: Number(data.l),
        volume: Number(data.v),
        quoteVolume: Number(data.q),
      })
    }, (open) => {
      setWatchOpen(open)
      if (open) setSeenLive(true)
    })
    return () => {
      window.clearTimeout(timer)
      close()
    }
  }, [symbols])

  useEffect(() => {
    const key = requestKey
    const symbol = selected
    const interval = timeframe
    const intervalMs = INTERVAL_MS[interval]
    const queued = new Map<number, Candle>()
    const duringLoad = new Map<number, Candle>()
    let current: Candle[] = []
    let loaded = false
    let stopped = false
    let opened = false
    let request: AbortController | null = null
    let lastRequestAt = 0
    let flushTimer = 0
    let reloadTimer = 0

    const active = () => !stopped && selectedRef.current === symbol && timeframeRef.current === interval
    const scheduleLoad = (delay = 0) => {
      if (!active() || request || reloadTimer) return
      // Coalesce reconnect, boundary and gap repair requests; failures retry at
      // most every five seconds without restarting the websocket or other feeds.
      const wait = Math.max(delay, 5000 - (Date.now() - lastRequestAt), 0)
      reloadTimer = window.setTimeout(() => { reloadTimer = 0; void load() }, wait)
    }
    const publish = (next: Candle[]) => {
      current = next
      setCandles(next)
    }
    async function load() {
      if (!active() || request) return
      const ac = new AbortController()
      request = ac
      lastRequestAt = Date.now()
      duringLoad.clear()
      for (const candle of queued.values()) bufferCandle(duringLoad, candle)
      let retry = false
      try {
        let rows = await fetchKlines(symbol, interval, ac.signal)
        if (!active()) return
        // Retain every timestamp received during the REST request, including a
        // final old bar followed immediately by the first tick of the next one.
        for (const candle of [...duringLoad.values()].sort((a, b) => a.time - b.time)) rows = mergeCandle(rows, candle)
        queued.clear()
        window.clearTimeout(flushTimer)
        flushTimer = 0
        publish(rows)
        loaded = true
        setReadyKey(key)
        setErrorKey('')
      } catch (error) {
        if (!active() || (error instanceof DOMException && error.name === 'AbortError')) return
        const invalid = error instanceof ApiError && error.status === 400
        if (!loaded) {
          setErrorKey(key)
          setErrorText(invalid ? '这个交易对没有现货 K 线。' : 'K 线暂时没有取到，正在重试。')
        }
        retry = !invalid
      } finally {
        if (request === ac) request = null
        duringLoad.clear()
        if (retry) scheduleLoad(5000)
      }
    }
    const flush = () => {
      flushTimer = 0
      if (!active() || !loaded) return
      let next = current
      for (const candle of [...queued.values()].sort((a, b) => a.time - b.time)) next = mergeCandle(next, candle)
      queued.clear()
      publish(next)
      if (candlesNeedBackfill(next, intervalMs, Date.now())) scheduleLoad()
    }
    const queue = (candle: Candle) => {
      if (!active()) return
      bufferCandle(queued, candle)
      if (request) bufferCandle(duringLoad, candle)
      if (!flushTimer) flushTimer = window.setTimeout(flush, 250)
    }
    void load()
    const close = openStream(`/ws/${symbol.toLowerCase()}@kline_${interval}`, (raw) => {
      const event = raw as KlineEvent
      const kline = event.k
      if (!kline || kline.s !== symbol || kline.i !== interval) return
      queue({
        time: toSeconds(kline.t),
        open: Number(kline.o),
        high: Number(kline.h),
        low: Number(kline.l),
        close: Number(kline.c),
        volume: Number(kline.v),
        quoteVolume: Number(kline.q),
        trades: Number(kline.n),
        takerBuyVolume: Number(kline.V),
        takerBuyQuoteVolume: Number(kline.Q),
        isClosed: kline.x === true,
      })
    }, (isOpen) => {
      if (!isOpen || !active()) return
      if (opened || loaded) scheduleLoad()
      opened = true
    })
    const repairTimer = window.setInterval(() => {
      if (active() && loaded && candlesNeedBackfill(current, intervalMs, Date.now())) scheduleLoad()
    }, 5000)
    return () => {
      stopped = true
      request?.abort()
      window.clearTimeout(flushTimer)
      window.clearTimeout(reloadTimer)
      window.clearInterval(repairTimer)
      queued.clear()
      duringLoad.clear()
      close()
    }
  }, [selected, timeframe, nonce, requestKey])

  useEffect(() => {
    const symbol = selected
    let pending = 0
    let timer = 0
    const flush = () => {
      timer = 0
      if (selectedRef.current !== symbol || pending <= 0) return
      setLive({ symbol, price: pending })
    }
    const close = openStream(`/ws/${symbol.toLowerCase()}@trade`, (raw) => {
      const event = raw as { s?: string; p?: string }
      if (event.s !== symbol || !event.p) return
      pending = Number(event.p)
      if (!timer) timer = window.setTimeout(flush, 80)
    }, () => undefined)
    return () => {
      window.clearTimeout(timer)
      close()
    }
  }, [selected])

  useEffect(() => {
    const symbol = selected
    const close = openStream(`/ws/${symbol.toLowerCase()}@depth20@1000ms`, (raw) => {
      if (selectedRef.current !== symbol) return
      const book = raw as { bids?: [string, string][]; asks?: [string, string][] }
      const parse = (rows: [string, string][] | undefined) => (rows ?? [])
        .slice(0, 20)
        .map(([price, qty]) => ({ price: Number(price), qty: Number(qty) }))
        .filter((level) => Number.isFinite(level.price) && Number.isFinite(level.qty))
      setDepth({ symbol, bids: parse(book.bids), asks: parse(book.asks) })
    }, () => undefined)
    return close
  }, [selected])

  const link: LinkState = watchOpen ? 'live' : seenLive ? 'reconnecting' : 'connecting'

  function applyQuote(nextQuote: QuoteMarket) {
    if (nextQuote === quoteMarket) return
    const next = symbolsForQuote(nextQuote)
    const currentBase = baseAsset(selectedRef.current)
    setQuoteMarket(nextQuote)
    setSymbols(next)
    setSelected(next.find((symbol) => baseAsset(symbol) === currentBase) ?? next[0])
  }

  async function addSymbol(raw: string): Promise<AddResult> {
    const cleaned = raw.toUpperCase().replace(/[^A-Z0-9]/g, '')
    if (isQuoteMarket(cleaned)) {
      applyQuote(cleaned)
      return 'ok'
    }
    const symbol = normalizeSymbol(raw, quoteMarket)
    if (!symbol) return 'invalid'
    if (symbols.includes(symbol)) {
      setSelected(symbol)
      return 'ok'
    }
    if (symbols.length >= MAX_SYMBOLS) return 'full'
    try {
      const quote = await fetchTicker(symbol)
      setQuotes((prev) => ({ ...prev, [symbol]: quote }))
      setSymbols((list) => (list.includes(symbol) ? list : [...list, symbol]))
      setSelected(symbol)
      return 'ok'
    } catch (error) {
      if (error instanceof ApiError && (error.status === 400 || error.status === 404)) return 'invalid'
      return 'network'
    }
  }

  function removeSymbol(symbol: string) {
    if (symbols.length <= 1) return
    const next = symbols.filter((item) => item !== symbol)
    setSymbols(next)
    if (selected === symbol) {
      const index = symbols.indexOf(symbol)
      setSelected(next[Math.max(0, index - 1)] ?? next[0])
    }
  }

  function resetSymbols() {
    const next = symbolsForQuote(quoteMarket)
    setSymbols(next)
    setSelected(next[0])
  }

  const visibleCandles = readyKey === requestKey ? candles : []
  const visibleDepth = depth?.symbol === selected ? depth : null
  const candleState = errorKey === requestKey ? 'error' : readyKey === requestKey ? 'ready' : 'loading'
  const candleError = errorKey === requestKey ? errorText : null
  const hero = live?.symbol === selected ? live.price : quotes[selected]?.last ?? visibleCandles.at(-1)?.close ?? null

  return {
    symbols,
    selected,
    setSelected,
    quoteMarket,
    setQuoteMarket: applyQuote,
    timeframe,
    setTimeframe,
    quotes,
    candles: visibleCandles,
    depth: visibleDepth,
    hero,
    link,
    candleState,
    candleError,
    restError: link === 'live' ? null : restError,
    clock,
    retry: () => setNonce((value) => value + 1),
    addSymbol,
    removeSymbol,
    resetSymbols,
  }
}

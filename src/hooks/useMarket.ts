import { useEffect, useRef, useState } from 'react'
import { fetchKlines, fetchTicker, fetchTickers, mergeCandle, openStream, toSeconds, ApiError } from '../lib/binance'
import { baseAsset, normalizeSymbol } from '../lib/format'
import {
  LEGACY_USDT_SYMBOLS,
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
  const readyKeyRef = useRef('')
  const requestKey = `${selected}|${timeframe}`

  const pendingKline = useRef<Candle | null>(null)

  useEffect(() => {
    selectedRef.current = selected
    timeframeRef.current = timeframe
    readyKeyRef.current = readyKey
  }, [selected, timeframe, readyKey])

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
    const ac = new AbortController()
    const key = requestKey
    let ignore = false
    let retryTimer = 0
    pendingKline.current = null

    const load = async () => {
      try {
        const rows = await fetchKlines(selected, timeframe, ac.signal)
        if (ignore) return
        const pending = pendingKline.current
        setCandles(pending && pending.time >= (rows.at(-1)?.time ?? 0) ? mergeCandle(rows, pending) : rows)
        setReadyKey(key)
        setErrorKey('')
      } catch (error) {
        if (ignore || (error instanceof DOMException && error.name === 'AbortError')) return
        const invalid = error instanceof ApiError && error.status === 400
        setErrorKey(key)
        setErrorText(invalid ? '这个交易对没有现货 K 线。' : 'K 线暂时没有取到，正在重试。')
        if (!invalid) retryTimer = window.setTimeout(() => setNonce((value) => value + 1), 5000)
      }
    }
    void load()
    return () => {
      ignore = true
      ac.abort()
      window.clearTimeout(retryTimer)
    }
  }, [selected, timeframe, nonce])

  useEffect(() => {
    const symbol = selected
    const interval = timeframe
    let queued: Candle | null = null
    let timer = 0
    const queue = (candle: Candle) => {
      if (selectedRef.current !== symbol || timeframeRef.current !== interval) return
      queued = candle
      pendingKline.current = candle
      if (timer) return
      timer = window.setTimeout(() => {
        timer = 0
        const next = queued
        queued = null
        if (pendingKline.current === next) pendingKline.current = null
        if (!next || selectedRef.current !== symbol || timeframeRef.current !== interval) return
        if (readyKeyRef.current !== `${symbol}|${interval}`) return
        setCandles((prev) => (prev.length ? mergeCandle(prev, next) : prev))
      }, 250)
    }
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
      })
    }, () => undefined)
    return () => {
      window.clearTimeout(timer)
      close()
    }
  }, [selected, timeframe])

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

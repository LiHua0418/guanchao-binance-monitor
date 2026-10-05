import type { Candle, Interval, Quote } from '../types'

export const REST_ORIGIN = 'https://data-api.binance.vision'
export const WS_ORIGIN = 'wss://data-stream.binance.vision'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

type MiniTicker = {
  symbol: string
  lastPrice: string
  openPrice: string
  highPrice: string
  lowPrice: string
  volume: string
  quoteVolume: string
  count?: number
}

export function toSeconds(value: number): number {
  if (value > 1e14) return Math.floor(value / 1e6)
  if (value > 1e11) return Math.floor(value / 1e3)
  return Math.floor(value)
}

export function toQuote(row: MiniTicker): Quote {
  return {
    symbol: row.symbol,
    last: Number(row.lastPrice),
    open: Number(row.openPrice),
    high: Number(row.highPrice),
    low: Number(row.lowPrice),
    volume: Number(row.volume),
    quoteVolume: Number(row.quoteVolume),
    count: Number(row.count ?? 0),
  }
}

async function readJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal, cache: 'no-store' })
  if (!response.ok) throw new ApiError(response.status, `HTTP ${response.status}`)
  return response.json() as Promise<T>
}

export async function fetchTickers(symbols: string[], signal?: AbortSignal): Promise<Record<string, Quote>> {
  const query = encodeURIComponent(JSON.stringify(symbols))
  const rows = await readJson<MiniTicker[]>(
    `${REST_ORIGIN}/api/v3/ticker/24hr?symbols=${query}&type=MINI`,
    signal,
  )
  return Object.fromEntries(rows.map((row) => [row.symbol, toQuote(row)]))
}

export async function fetchTicker(symbol: string, signal?: AbortSignal): Promise<Quote> {
  const row = await readJson<MiniTicker>(
    `${REST_ORIGIN}/api/v3/ticker/24hr?symbol=${symbol}&type=MINI`,
    signal,
  )
  if (!row.symbol) throw new ApiError(400, 'invalid symbol')
  return toQuote(row)
}

export async function fetchKlines(
  symbol: string,
  interval: Interval,
  signal?: AbortSignal,
  limit = 1000,
): Promise<Candle[]> {
  const rows = await readJson<unknown[][]>(
    `${REST_ORIGIN}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`,
    signal,
  )
  const byTime = new Map<number, Candle>()
  for (const row of rows) {
    const time = toSeconds(Number(row[0]))
    byTime.set(time, {
      time,
      open: Number(row[1]),
      high: Number(row[2]),
      low: Number(row[3]),
      close: Number(row[4]),
      volume: Number(row[5]),
    })
  }
  return [...byTime.values()].sort((a, b) => a.time - b.time)
}

export function mergeCandle(prev: Candle[], candle: Candle): Candle[] {
  if (!prev.length) return [candle]
  const last = prev[prev.length - 1]
  if (candle.time === last.time) {
    const next = prev.slice()
    next[next.length - 1] = candle
    return next
  }
  if (candle.time > last.time) return [...prev.slice(-999), candle]
  return prev
}

export function openStream(
  path: string,
  onMessage: (data: unknown) => void,
  onOpen: (open: boolean) => void,
): () => void {
  let socket: WebSocket | null = null
  let stopped = false
  let timer = 0
  let attempt = 0

  const connect = () => {
    socket = new WebSocket(`${WS_ORIGIN}${path}`)
    socket.onopen = () => {
      attempt = 0
      onOpen(true)
    }
    socket.onmessage = (event) => {
      try {
        onMessage(JSON.parse(String(event.data)))
      } catch {
        /* ignore malformed frames */
      }
    }
    socket.onerror = () => {
      socket?.close()
    }
    socket.onclose = () => {
      onOpen(false)
      if (stopped) return
      attempt += 1
      timer = window.setTimeout(connect, Math.min(8000, 700 * attempt))
    }
  }

  connect()
  return () => {
    stopped = true
    window.clearTimeout(timer)
    socket?.close()
  }
}

const QUOTE_ASSETS = ['USDT', 'USDC', 'FDUSD', 'TUSD', 'BTC', 'ETH', 'BNB']

export function normalizeSymbol(raw: string, quote = 'USDT'): string {
  const cleaned = raw.toUpperCase().replace(/[^A-Z0-9]/g, '')
  if (cleaned.length < 2) return ''
  const paired = QUOTE_ASSETS.some((item) => cleaned.endsWith(item) && cleaned.length > item.length + 1)
  const symbol = paired ? cleaned : `${cleaned}${quote}`
  return /^[A-Z0-9]{5,20}$/.test(symbol) ? symbol : ''
}

export function baseAsset(symbol: string): string {
  const quote = QUOTE_ASSETS.find((item) => symbol.endsWith(item) && symbol.length > item.length)
  return quote ? symbol.slice(0, -quote.length) : symbol
}

export function quoteAsset(symbol: string): string {
  const quote = QUOTE_ASSETS.find((item) => symbol.endsWith(item) && symbol.length > item.length)
  return quote ?? 'USDT'
}

export function inferDigits(price: number): number {
  const abs = Math.abs(price)
  if (abs >= 100) return 2
  if (abs >= 10) return 3
  if (abs >= 1) return 4
  if (abs >= 0.1) return 4
  if (abs >= 0.01) return 5
  if (abs >= 0.001) return 6
  return 8
}

export function formatPrice(value: number, digits = inferDigits(value)): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function formatSigned(value: number, digits: number): string {
  const body = Math.abs(value).toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
  if (value > 0) return `+${body}`
  if (value < 0) return `-${body}`
  return body
}

export function formatPct(value: number): string {
  const body = `${Math.abs(value).toFixed(2)}%`
  if (value > 0) return `+${body}`
  if (value < 0) return `-${body}`
  return body
}

export function formatCompact(value: number): string {
  const abs = Math.abs(value)
  const sign = value < 0 ? '-' : ''
  if (abs >= 1e8) return `${sign}${(abs / 1e8).toFixed(2)}亿`
  if (abs >= 1e4) return `${sign}${(abs / 1e4).toFixed(2)}万`
  if (abs >= 100) return `${sign}${abs.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
  if (abs >= 1) return `${sign}${abs.toFixed(2)}`
  return `${sign}${abs.toFixed(4)}`
}

export function formatQty(value: number): string {
  if (value >= 1000) return value.toLocaleString('en-US', { maximumFractionDigits: 2 })
  if (value >= 1) return value.toFixed(3)
  if (value >= 0.01) return value.toFixed(4)
  return value.toFixed(6)
}

export function formatClock(date: Date): { time: string; date: string } {
  return {
    time: date.toLocaleTimeString('zh-CN', {
      hour12: false,
      timeZone: 'Asia/Shanghai',
    }),
    date: date.toLocaleDateString('zh-CN', {
      month: 'numeric',
      day: 'numeric',
      weekday: 'short',
      timeZone: 'Asia/Shanghai',
    }),
  }
}

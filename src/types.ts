export const INTERVALS = ['15m', '1h', '4h', '1d'] as const

export type Interval = (typeof INTERVALS)[number]

export const INTERVAL_MS: Record<Interval, number> = {
  '15m': 15 * 60 * 1000,
  '1h': 60 * 60 * 1000,
  '4h': 4 * 60 * 60 * 1000,
  '1d': 24 * 60 * 60 * 1000,
}

export const INTERVAL_LABEL: Record<Interval, string> = {
  '15m': '15分',
  '1h': '1时',
  '4h': '4时',
  '1d': '1日',
}

export const HORIZON_LABEL: Record<Interval, string> = {
  '15m': '未来 12 根 · 往后 3 小时',
  '1h': '未来 12 根 · 往后 12 小时',
  '4h': '未来 12 根 · 往后 2 天',
  '1d': '未来 12 根 · 往后 12 天',
}

export const QUOTE_MARKETS = ['USDT', 'USDC', 'FDUSD', 'BTC', 'ETH'] as const

export type QuoteMarket = (typeof QUOTE_MARKETS)[number]

export const MARKET_BASES: Record<QuoteMarket, readonly string[]> = {
  USDT: ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'LTC', 'BCH', 'TRX', 'USDC', 'FDUSD', 'TUSD'],
  USDC: ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'LTC', 'BCH', 'TRX', 'FDUSD'],
  FDUSD: ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'LINK', 'LTC'],
  BTC: ['ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'LINK', 'LTC', 'TRX'],
  ETH: ['BNB', 'SOL', 'XRP', 'LINK', 'LTC', 'ADA', 'TRX'],
}

export function symbolsForQuote(quote: QuoteMarket): string[] {
  return MARKET_BASES[quote].map((base) => `${base}${quote}`)
}

export const DEFAULT_SYMBOLS = symbolsForQuote('USDT')

export const LEGACY_USDT_SYMBOLS = [
  'BTCUSDT',
  'ETHUSDT',
  'BNBUSDT',
  'SOLUSDT',
  'XRPUSDT',
  'DOGEUSDT',
  'ADAUSDT',
  'AVAXUSDT',
  'LINKUSDT',
  'LTCUSDT',
  'BCHUSDT',
  'TRXUSDT',
]

export const NAMES: Record<string, string> = {
  BTC: '比特币',
  ETH: '以太坊',
  BNB: 'BNB',
  SOL: 'Solana',
  XRP: '瑞波',
  DOGE: '狗狗币',
  ADA: '艾达币',
  AVAX: '雪崩',
  LINK: 'Chainlink',
  LTC: '莱特币',
  BCH: '比特币现金',
  TRX: '波场',
  USDC: 'USD Coin',
  FDUSD: 'FDUSD',
  TUSD: 'TrueUSD',
  EUR: '欧元',
}

export type Candle = {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export type Quote = {
  symbol: string
  last: number
  open: number
  high: number
  low: number
  volume: number
  quoteVolume: number
  count: number
}

export type BookLevel = {
  price: number
  qty: number
}

export type Depth = {
  bids: BookLevel[]
  asks: BookLevel[]
}

export type LinkState = 'connecting' | 'live' | 'reconnecting'

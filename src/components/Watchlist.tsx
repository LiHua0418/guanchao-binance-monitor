import { useState, type FormEvent } from 'react'
import { baseAsset, formatPct, formatPrice, inferDigits, quoteAsset } from '../lib/format'
import { NAMES, type Quote } from '../types'

type Props = {
  symbols: string[]
  quotes: Record<string, Quote>
  selected: string
  hero: number | null
  customized: boolean
  onSelect: (symbol: string) => void
  onRemove: (symbol: string) => void
  onAdd: (raw: string) => Promise<'ok' | 'invalid' | 'network' | 'full'>
  onReset: () => void
}

const ADD_MESSAGE = {
  invalid: '没有这个现货交易对',
  network: '查询失败，稍后再试',
  full: '自选最多 24 个',
  ok: '',
}

export function Watchlist({
  symbols,
  quotes,
  selected,
  hero,
  customized,
  onSelect,
  onRemove,
  onAdd,
  onReset,
}: Props) {
  const [draft, setDraft] = useState('')
  const [status, setStatus] = useState<'idle' | 'checking' | 'invalid' | 'network' | 'full'>('idle')
  const items = symbols.map((symbol) => {
    const quote = quotes[symbol]
    const price = symbol === selected && hero != null ? hero : quote?.last
    const last = price != null && Number.isFinite(price) && price > 0 ? price : null
    const pct = quote && last != null && Number.isFinite(quote.open) && quote.open > 0
      ? ((last - quote.open) / quote.open) * 100 : null
    return { symbol, base: baseAsset(symbol), last, pct }
  })
  const available = items.filter((item) => item.pct != null)
  const rising = available.filter((item) => item.pct! > 0).length
  const falling = available.filter((item) => item.pct! < 0).length

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!draft.trim() || status === 'checking') return
    setStatus('checking')
    const result = await onAdd(draft)
    setStatus(result === 'ok' ? 'idle' : result)
    if (result === 'ok') setDraft('')
  }

  return (
    <aside className="panel watch">
      <div className="panel-head">
        <div>
          <p className="kicker">自选</p>
          <h2>正在观察</h2>
        </div>
        {customized && (
          <button type="button" className="text-btn" onClick={onReset}>
            恢复默认
          </button>
        )}
      </div>
      <div
        className="watch-breadth"
        aria-label={`当前自选内 24 小时上涨 ${rising} 个，下跌 ${falling} 个；${available.length} 个有行情，缺失行情不统计`}
        title="仅统计当前自选中有有效行情的交易对，缺失行情不计入"
      >
        <span>自选内 · 24h</span>
        <span className="up">↑ <b className="num">{rising}</b></span>
        <span className="down">↓ <b className="num">{falling}</b></span>
        <small className="num">{available.length}/{symbols.length} 有行情</small>
      </div>
      <form className="search" onSubmit={submit}>
        <input
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value)
            if (status !== 'idle' && status !== 'checking') setStatus('idle')
          }}
          placeholder="添加交易对，如 SOL、USDC"
          aria-label="添加交易对"
          spellCheck={false}
          autoCapitalize="characters"
        />
        <button type="submit" disabled={status === 'checking'}>
          {status === 'checking' ? '查找' : '添加'}
        </button>
      </form>
      {status !== 'idle' && status !== 'checking' && <p className="form-error">{ADD_MESSAGE[status]}</p>}
      <div className="watch-columns" aria-hidden="true">
        <span>币种</span>
        <span>价格</span>
        <span>24h</span>
      </div>
      <div className="watch-list">
        {items.map(({ symbol, base, last, pct }) => {
          const tone = pct == null ? 'flat' : pct > 0 ? 'up' : pct < 0 ? 'down' : 'flat'
          return (
            <div key={symbol} className={`row ${symbol === selected ? 'active' : ''}`}>
              <button
                type="button"
                className="row-main"
                aria-pressed={symbol === selected}
                aria-label={`查看 ${base} / ${quoteAsset(symbol)}${symbol === selected ? '，当前选中' : ''}`}
                onClick={() => onSelect(symbol)}
              >
                <span className="asset-identity">
                  <span className="asset-token" aria-hidden="true">{base.slice(0, 1)}</span>
                  <span className="asset-meta" title={symbol}>
                    <strong>{base}</strong>
                    <em>{NAMES[base] && NAMES[base] !== base ? `${NAMES[base]} · ` : ''}{quoteAsset(symbol)}</em>
                  </span>
                </span>
                <span className="row-price">
                  <strong className="num">{last == null ? '—' : formatPrice(last, inferDigits(last))}</strong>
                </span>
                <span className={`row-change num ${tone}`}>{pct == null ? '—' : formatPct(pct)}</span>
              </button>
              {symbols.length > 1 && (
                <button type="button" className="row-remove" aria-label={`移除 ${symbol}`} onClick={() => onRemove(symbol)}>
                  ×
                </button>
              )}
            </div>
          )
        })}
      </div>
    </aside>
  )
}

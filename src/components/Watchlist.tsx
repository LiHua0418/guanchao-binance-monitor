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
      <div className="watch-list">
        {symbols.map((symbol) => {
          const quote = quotes[symbol]
          const last = symbol === selected && hero != null ? hero : quote?.last
          const pct = quote && last != null && quote.open ? ((last - quote.open) / quote.open) * 100 : null
          const tone = pct == null ? 'flat' : pct > 0 ? 'up' : pct < 0 ? 'down' : 'flat'
          return (
            <div key={symbol} className={`row ${symbol === selected ? 'active' : ''}`}>
              <button type="button" className="row-main" onClick={() => onSelect(symbol)}>
                <span>
                  <strong>{NAMES[baseAsset(symbol)] ?? baseAsset(symbol)}</strong>
                  <em>{baseAsset(symbol)} / {quoteAsset(symbol)}</em>
                </span>
                <span className={`row-price ${tone}`}>
                  <strong className="num">{last == null ? '—' : formatPrice(last, inferDigits(last))}</strong>
                  <em className="num">{pct == null ? '—' : formatPct(pct)}</em>
                </span>
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

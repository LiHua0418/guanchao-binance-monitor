import { useEffect, useRef, useState } from 'react'
import { closedCandles } from '../lib/b3'
import { B4_SYMBOLS, b4RequestKey, validateB4Forecast, type B4Forecast, type B4Status } from '../lib/b4'
import { INTERVAL_MS, type Candle, type Interval } from '../types'

type State = { key: string; status: B4Status; model: B4Forecast | null; message?: string }

export function useB4Forecast(candles: Candle[], interval: Interval, symbol: string, nowMs: number, enabled: boolean) {
  const closed = closedCandles(candles, INTERVAL_MS[interval], nowMs)
  const [revision, setRevision] = useState(0)
  const key = `${b4RequestKey(symbol, interval, closed)}|${revision}`
  const closedRef = useRef(closed)
  const [state, setState] = useState<State>({ key: '', status: 'loading', model: null })
  useEffect(() => { closedRef.current = closed }, [closed])
  useEffect(() => {
    if (!enabled) return
    if (!B4_SYMBOLS.includes(symbol)) return
    const input = closedRef.current
    if (input.length < 100) return
    const controller = new AbortController()
    let active = true
    const timer = window.setTimeout(() => controller.abort(), 45000)
    fetch('/api/b4/forecast', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({ symbol, interval, candles: input }),
    }).then(async (response) => {
      if (!response.ok) {
        const error = await response.json().catch(() => null)
        throw new Error(error?.error || 'B4 本地推理服务未连接，请启动服务后重试。')
      }
      const model = validateB4Forecast(await response.json(), symbol, interval, input.at(-1)!)
      if (active) setState({ key, status: 'ready', model })
    }).catch((error: Error) => {
      if (active) setState({ key, status: 'error', model: null, message: error.name === 'AbortError'
        ? 'B4 推理超时，请检查本地服务后重试。' : error.message })
    }).finally(() => window.clearTimeout(timer))
    return () => { active = false; controller.abort(); window.clearTimeout(timer) }
  }, [key, symbol, interval, enabled])
  const current: State = !enabled ? { key, status: 'loading', model: null }
    : !B4_SYMBOLS.includes(symbol) ? { key, status: 'unsupported', model: null, message: 'B4 支持 BTC、ETH、BNB、SOL / USDT。' }
      : closed.length < 100 ? { key, status: 'insufficient', model: null, message: '需要至少 100 根连续、已收盘的完整行情。' }
        : state.key === key ? state : { key, status: 'loading', model: null }
  return { ...current, retry: () => setRevision((value) => value + 1) }
}

import { useEffect, useRef, useState } from 'react'
import { closedCandles, type B3Forecast, type ReplayPoint } from '../lib/b3'
import type { Candle } from '../types'

type State = {
  key: string
  status: 'loading' | 'ready' | 'insufficient' | 'unsupported' | 'error'
  model: B3Forecast | null
  history: ReplayPoint[]
  message?: string
}

export function useNeuralForecast(candles: Candle[], intervalMs: number, timeframe: string, symbol: string, nowMs: number) {
  const workerRef = useRef<Worker | null>(null)
  const requestRef = useRef(0)
  const [retryCount, setRetryCount] = useState(0)
  const [state, setState] = useState<State>({ key: '', status: 'loading', model: null, history: [] })
  const closed = closedCandles(candles, intervalMs, nowMs)
  const last = closed.at(-1)
  // The open candle cannot move a forecast; recompute only when a bar closes.
  const key = `${symbol}|${timeframe}|${last?.time ?? 0}|${last?.close ?? 0}|${last?.volume ?? 0}|${closed.length}|${retryCount}`
  const closedRef = useRef(closed)
  useEffect(() => { closedRef.current = closed }, [closed])

  useEffect(() => {
    const worker = new Worker(new URL('../workers/b3.worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker
    return () => { worker.terminate(); workerRef.current = null }
  }, [])

  useEffect(() => {
    const worker = workerRef.current
    if (!worker) return
    const id = ++requestRef.current
    worker.onmessage = (event: MessageEvent<Omit<State, 'key'> & { id: number }>) => {
      if (event.data.id !== requestRef.current) return
      setState({ ...event.data, key })
    }
    worker.onerror = () => {
      if (id === requestRef.current) setState({ key, status: 'error', message: '模型计算失败，请重试', model: null, history: [] })
    }
    if (closedRef.current.length < 97) {
      setState({ key, status: 'insufficient', message: '需要至少 97 根已收盘 K 线', model: null, history: [] })
      return
    }
    worker.postMessage({ id, interval: timeframe, intervalMs, symbol, candles: closedRef.current })
  }, [key, intervalMs, timeframe, symbol])

  const current = state.key === key ? state : { key, status: 'loading' as const, model: null, history: [] }
  return { ...current, retry: () => setRetryCount((v) => v + 1) }
}

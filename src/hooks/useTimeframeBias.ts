import { useEffect, useState } from 'react'
import { analyze, type Bias } from '../lib/analysis'
import { fetchKlines } from '../lib/binance'
import { INTERVALS, INTERVAL_MS, type Interval } from '../types'

export function useTimeframeBias(symbol: string) {
  const [map, setMap] = useState<Partial<Record<Interval, Bias | null>>>({})

  useEffect(() => {
    const ac = new AbortController()
    let ignore = false
    const load = async () => {
      const pairs = await Promise.all(INTERVALS.map(async (interval) => {
        try {
          const candles = await fetchKlines(symbol, interval, ac.signal, 200)
          return [interval, analyze(candles, INTERVAL_MS[interval])?.bias ?? null] as const
        } catch {
          return [interval, null] as const
        }
      }))
      if (!ignore) setMap(Object.fromEntries(pairs))
    }
    void load()
    const id = window.setInterval(() => void load(), 60_000)
    return () => {
      ignore = true
      ac.abort()
      window.clearInterval(id)
    }
  }, [symbol])

  return map
}

import { useMemo } from 'react'
import { inferForecast } from '../lib/lstm'
import { PRETRAINED } from '../lib/pretrained-lstm'
import type { Candle } from '../types'

export function useNeuralForecast(candles: Candle[], intervalMs: number, timeframe: string) {
  const bundle = PRETRAINED?.[timeframe] ?? null
  const model = useMemo(
    () => (bundle && candles.length >= 80 ? inferForecast(candles, intervalMs, bundle) : null),
    [bundle, candles, intervalMs],
  )
  return {
    model,
    progress: null,
    training: false,
    ready: bundle != null,
  }
}

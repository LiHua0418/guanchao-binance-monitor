import { forecastB3, replayB3, validateB3Bundle, type B3Bundle } from '../lib/b3'
import type { Candle } from '../types'

const bundles = new Map<string, Promise<B3Bundle>>()
async function loadBundle(interval: string): Promise<B3Bundle> {
  if (!['15m', '1h', '4h', '1d'].includes(interval)) throw new Error('不支持的模型周期')
  if (!bundles.has(interval)) {
    const request = fetch(`${import.meta.env.BASE_URL}models/b3-${interval}.json`)
      .then(async (response) => {
        if (!response.ok) throw new Error('模型文件暂时无法载入')
        return validateB3Bundle(await response.json(), interval)
      }).catch((error: unknown) => { bundles.delete(interval); throw error })
    bundles.set(interval, request)
  }
  return bundles.get(interval)!
}
self.onmessage = async (event: MessageEvent<{ id: number; interval: string; intervalMs: number; symbol: string; candles: Candle[] }>) => {
  const { id, interval, intervalMs, symbol, candles } = event.data
  try {
    const bundle = await loadBundle(interval)
    if (!bundle.trainedSymbols.includes(symbol)) {
      self.postMessage({ id, status: 'unsupported', message: '该交易对不在本模型的训练与评估范围内', model: null, history: [] })
      return
    }
    const model = forecastB3(candles, intervalMs, bundle)
    if (!model) {
      self.postMessage({ id, status: 'insufficient', message: '需要至少 97 根连续、已收盘的 K 线', model: null, history: [] })
      return
    }
    self.postMessage({ id, status: 'ready', model, history: replayB3(candles, intervalMs, bundle) })
  } catch (error) {
    self.postMessage({ id, status: 'error', message: error instanceof Error ? error.message : '模型推断失败', model: null, history: [] })
  }
}

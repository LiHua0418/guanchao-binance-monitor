import { appendFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const LOG = '/tmp/opencode/lstm-train.log'
function log(line: string) {
  console.log(line)
  appendFileSync(LOG, `${line}\n`)
}
import { trainSeriesBundle, type LstmBundle } from '../src/lib/lstm.ts'
import type { Candle } from '../src/types.ts'

const JOBS: { interval: string; bars: number }[] = [
  { interval: '15m', bars: 8000 },
  { interval: '1h', bars: 8000 },
  { interval: '4h', bars: 5000 },
  { interval: '1d', bars: 2200 },
]
const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

function toCandle(row: number[]): Candle {
  return {
    time: Math.floor(row[0] / 1000),
    open: Number(row[1]),
    high: Number(row[2]),
    low: Number(row[3]),
    close: Number(row[4]),
    volume: Number(row[5]),
  }
}

async function fetchHistory(symbol: string, interval: string, bars: number): Promise<Candle[]> {
  const candles: Candle[] = []
  let end = Date.now()
  while (candles.length < bars) {
    const url = `https://data-api.binance.vision/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=1000&endTime=${end}`
    const response = await fetch(url)
    if (!response.ok) throw new Error(`${symbol} ${interval} HTTP ${response.status}`)
    const rows = await response.json() as number[][]
    if (!rows.length) break
    const batch = rows.map(toCandle)
    candles.unshift(...batch)
    end = rows[0][0] - 1
    if (rows.length < 1000) break
    await new Promise((resolve) => setTimeout(resolve, 80))
  }
  const unique = new Map<number, Candle>()
  for (const candle of candles) unique.set(candle.time, candle)
  return [...unique.values()].sort((a, b) => a.time - b.time).slice(-bars)
}

function writeBundles(bundles: LstmBundle[]) {
  const body = `import type { LstmBundle } from './lstm'

export const PRETRAINED: Record<string, LstmBundle> = ${JSON.stringify(Object.fromEntries(bundles.map((bundle) => [bundle.interval, bundle])), null, 2)}
`
  writeFileSync(fileURLToPath(new URL('../src/lib/pretrained-lstm.ts', import.meta.url)), body)
}

async function main() {
  const bundles: LstmBundle[] = []
  for (const job of JOBS) {
    log(`\n== ${job.interval} 下载 ==`)
    const groups = []
    for (const symbol of SYMBOLS) {
      const candles = await fetchHistory(symbol, job.interval, job.bars)
      log(`${symbol} ${candles.length}`)
      groups.push({ symbol, candles })
    }
    log(`== ${job.interval} 训练，验证损失不再下降才停 ==`)
    const started = Date.now()
    const bundle = await trainSeriesBundle(groups, job.interval, (info) => {
      if (info.epoch === 1 || info.epoch % 5 === 0 || info.wait === 0) {
        log(`${job.interval} epoch ${info.epoch}/${info.maxEpochs} train ${info.trainLoss.toExponential(3)} val ${info.valLoss.toExponential(3)} lr ${info.lr} wait ${info.wait}`)
      }
    })
    if (!bundle) throw new Error(`${job.interval} 训练失败`)
    log(`${job.interval} 完成 ${Math.round((Date.now() - started) / 1000)}s epochs ${bundle.epochs} 留出命中 ${(bundle.testHitRate * 100).toFixed(1)}% 误差 ${bundle.testMaePct.toFixed(3)}%`)
    bundles.push(bundle)
    writeBundles(bundles)
  }
  log('全部训练结束')
}

void main()

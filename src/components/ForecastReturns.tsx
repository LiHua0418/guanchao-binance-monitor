import type { B3Forecast } from '../lib/b3'
import { formatSigned } from '../lib/format'

type Props = { path: B3Forecast['path']; nextMaePct: number }

const percent = (value: number) => `${formatSigned(value, 4)}%`

export function ForecastReturns({ path, nextMaePct }: Props) {
  const origin = path[0]?.mid
  if (!origin || origin <= 0 || path.length < 2) return null
  const returns = path.map((point) => (point.mid / origin - 1) * 100)
  if (returns.some((value) => !Number.isFinite(value))) return null

  const maxOffset = Math.max(...returns.map(Math.abs))
  // Keep the price forecast's actual amplitude. Error provides a minimum scale
  // so a near-zero estimate cannot become a large swing through auto-zooming.
  const extent = Math.max(nextMaePct, maxOffset * 1.1, 0.0001)
  const left = 60, right = 304, top = 15, bottom = 119
  const zero = (top + bottom) / 2
  const x = (index: number) => left + index / (returns.length - 1) * (right - left)
  const y = (value: number) => zero - value / extent * (bottom - top) / 2
  const line = returns.map((value, index) => `${index ? 'L' : 'M'}${x(index)},${y(value)}`).join(' ')
  const steps = [1, 3, 6, 12].filter((step) => step < returns.length)

  return (
    <div className="b3-returns">
      <h4>模型估计变化 · 相对起点</h4>
      <svg viewBox="0 0 320 146" role="img" aria-label={`未来 ${returns.length - 1} 根累计涨跌幅，最大绝对偏移 ${maxOffset.toFixed(4)}%，纵轴范围正负 ${extent.toFixed(4)}%`}>
        {[extent, 0, -extent].map((tick) => (
          <g key={tick}>
            <line x1={left} x2={right} y1={y(tick)} y2={y(tick)} className={tick === 0 ? 'b3-returns-zero' : 'b3-returns-grid'} />
            <text x={left - 7} y={y(tick) + 3} textAnchor="end">{tick === 0 ? '0%' : `${formatSigned(tick, extent < 0.01 ? 4 : 2)}%`}</text>
          </g>
        ))}
        <path d={line} className="b3-returns-line" />
        {steps.map((step) => (
          <g key={step}>
            <circle cx={x(step)} cy={y(returns[step])} r="2.5" className="b3-returns-point">
              <title>{`+${step} 根：${percent(returns[step])}`}</title>
            </circle>
            <text x={x(step)} y="140" textAnchor="middle">+{step}</text>
          </g>
        ))}
      </svg>
      <dl className="b3-return-steps">
        {steps.map((step) => (
          <div key={step}><dt>+{step} 根</dt><dd className="num">{percent(returns[step])}</dd></div>
        ))}
      </dl>
      <dl className="stats b3-stats">
        <div><dt>12 步最大绝对偏移</dt><dd className="num">{maxOffset.toFixed(4)}%</dd></div>
        <div><dt>下一根平均误差</dt><dd className="num">{nextMaePct.toFixed(4)}%</dd></div>
      </dl>
      <p className="b3-caption">偏移为本次 12 步估计，误差为历史下一根 MAE，时间跨度不同。纵轴至少覆盖 ± 下一根 MAE，接近零即真实幅度很小。</p>
    </div>
  )
}

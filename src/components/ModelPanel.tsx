import type { B3Forecast, ReplayPoint } from '../lib/b3'
import { formatPrice, formatSigned } from '../lib/format'
import { ForecastReturns } from './ForecastReturns'

type Props = {
  model: B3Forecast | null
  history: ReplayPoint[]
  status: 'loading' | 'ready' | 'insufficient' | 'unsupported' | 'error'
  message?: string
  onRetry: () => void
  intervalLabel: string
}

function dateLabel(value: string): string {
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).format(new Date(value))
}

const EMPTY_TEXT = {
  loading: '正在加载这个周期的已训练模型与行情。',
  insufficient: '已收盘的连续 K 线不足，暂时无法生成预测。',
  unsupported: '当前模型仅覆盖 BTC、ETH、BNB、SOL 的 USDT 交易对；其他交易对暂不提供模型预测。',
  error: '模型加载失败，请重试。',
  ready: '当前没有可用的预测结果。',
} as const

export function ModelPanel({ model, history, status, message, onRetry, intervalLabel }: Props) {
  const active = status === 'ready' ? model : null
  const origin = active?.path[0]
  const next = active?.path[1]
  const endpoint = active?.path.at(-1)
  const bundle = active?.bundle
  const metrics = bundle?.metrics
  const nextMetrics = metrics?.byHorizon[0]
  const baselineOnly = bundle?.blend === 0
  const replay = history.filter((point) => Number.isFinite(point.predicted)
    && Number.isFinite(point.actual) && point.actual > 0)
  const comparableReplay = replay.filter((point): point is ReplayPoint & { baseline: number } =>
    'baseline' in point && typeof point.baseline === 'number' && Number.isFinite(point.baseline) && point.baseline > 0)
  const replayForMetrics = comparableReplay.length ? comparableReplay : replay
  const replayMae = replayForMetrics.length
    ? replayForMetrics.reduce((sum, point) => sum + Math.abs(point.predicted / point.actual - 1) * 100, 0) / replayForMetrics.length
    : null
  const replayBaselineMae = comparableReplay.length
    ? comparableReplay.reduce((sum, point) => sum + Math.abs(point.baseline / point.actual - 1) * 100, 0) / comparableReplay.length
    : null
  const maeDifference = nextMetrics ? nextMetrics.maePct - nextMetrics.baselineMaePct : 0
  const nextPct = next && origin && origin.mid > 0 ? (next.mid / origin.mid - 1) * 100 : 0
  const endpointPct = endpoint && origin && origin.mid > 0 ? (endpoint.mid / origin.mid - 1) * 100 : 0
  const pointDirection = Math.abs(nextPct) < 0.00005 ? '点估计接近起点' : nextPct > 0 ? '点估计上' : '点估计下'

  return (
    <section className="block b3-panel" aria-label="guanchao-b3 模型">
      <div className="block-title">
        <p className="kicker">预测模型 · {intervalLabel}</p>
        <span className="b3-badge">B3</span>
      </div>
      <h3 className="b3-name">guanchao-b3</h3>
      <p className="b3-architecture">CNN–LSTM–Attention</p>
      {active && next && endpoint && bundle && metrics && nextMetrics ? (
        <>
          <div className="bias model-bias flat">
            <strong>{baselineOnly ? '持平基线' : '方向未验证'}</strong>
            <span>{baselineOnly ? '下一根回退至起点' : '方向与幅度待验证'}</span>
          </div>
          <div className="proj">
            <div>
              <span>{intervalLabel} · 下一根价格估计</span>
              <strong className="num">{formatPrice(next.mid)}</strong>
            </div>
            <em className="b3-point-estimate"><span>{pointDirection}</span>{formatSigned(nextPct, 4)}%</em>
          </div>
          <p className="b3-caption">每根收盘预测下一根，仅用已收盘 K 线；变化相对起点收盘价。</p>
          <div className="b3-range">
            <span>下一根 · 标称 80% 经验误差区间</span>
            <strong className="num">{formatPrice(next.lower)} – {formatPrice(next.upper)}</strong>
          </div>
          <ForecastReturns path={active.path} nextMaePct={nextMetrics.maePct} />

          <div className="b3-evaluation">
            <h4>下一根 · 留出时段复评</h4>
            <p className="b3-caption">四个训练币种 · 仅统计一步预测</p>
            <dl className="stats b3-stats">
              <div><dt>下一根 MAE</dt><dd className="num">{nextMetrics.maePct.toFixed(5)}%</dd></div>
              <div><dt>持平基线 MAE</dt><dd className="num">{nextMetrics.baselineMaePct.toFixed(5)}%</dd></div>
              <div>
                <dt>方向命中</dt>
                <dd className="num">{baselineOnly ? '无方向信号' : `${(nextMetrics.hitRate * 100).toFixed(1)}%`}</dd>
              </div>
              <div><dt>区间实际覆盖</dt><dd className="num">{(nextMetrics.coverage80 * 100).toFixed(1)}%</dd></div>
            </dl>
            <p className="b3-test-verdict b3-unproven">
              模型 MAE − 持平 MAE：{formatSigned(maeDifference, 5)} 个百分点。负值表示误差更小，微小差值不代表方向或幅度已验证。
            </p>
            <p className="b3-caption">{metrics.testSamples.toLocaleString('zh-CN')} 个复评起点，同一留出时段已用于版本复评。MAE 以目标实际价格为分母；80% 为标称覆盖。</p>
          </div>

          <div className="b3-evaluation">
            <h4>当前币种 · 逐根回放</h4>
            <p className="b3-caption">每根收盘预测下一根，最多对照最近 240 根。</p>
            {replayMae !== null ? (
              <>
                <dl className="stats b3-stats">
                  <div><dt>模型回放误差</dt><dd className="num">{replayMae.toFixed(4)}%</dd></div>
                  <div><dt>复制上一收盘误差</dt><dd className="num">{replayBaselineMae !== null ? `${replayBaselineMae.toFixed(4)}%` : '暂无数据'}</dd></div>
                </dl>
                <p className="b3-caption">{replayForMetrics.length} 次逐根回放，误差均以实际价格为分母。{replayBaselineMae !== null ? '两项使用同一批回放样本。' : ''}上方是四币种留出时段复评，样本范围不同。</p>
              </>
            ) : (
              <p className="note quiet">当前窗口还没有已收盘、可对照下一根预测的回放样本。</p>
            )}
          </div>

          <details className="b3-details">
            <summary>未来 12 根路径与全步评估</summary>
            <p className="b3-caption">图中蓝线保留未来 12 根的实际估计值；每一步单独评估，回退为零的步数仅表示持平基线。</p>
            <dl>
              <div><dt>+12 根价格估计</dt><dd className="num">{formatPrice(endpoint.mid)} · {formatSigned(endpointPct, 4)}%</dd></div>
              <div><dt>所有 1–12 步平均 MAE</dt><dd className="num">{metrics.modelMaePct.toFixed(3)}%</dd></div>
              <div><dt>所有 1–12 步持平基线 MAE</dt><dd className="num">{metrics.baselineMaePct.toFixed(3)}%</dd></div>
              <div><dt>所有 1–12 步区间覆盖</dt><dd className="num">{(metrics.coverage80 * 100).toFixed(1)}%</dd></div>
            </dl>
          </details>
          <details className="b3-details">
            <summary>模型训练与数据范围</summary>
            <dl>
              <div><dt>拟合与校准截止</dt><dd>{dateLabel(bundle.dataCutoff)}</dd></div>
              <div><dt>评估数据截止</dt><dd>{dateLabel(bundle.dataEnd)}</dd></div>
              <div><dt>模型生成时间</dt><dd>{dateLabel(bundle.trainedAt)}</dd></div>
              <div><dt>训练范围</dt><dd>{bundle.trainedSymbols.join('、')}</dd></div>
            </dl>
            <p className="b3-caption">以上时间均为北京时间。历史回放是在当前模型下重算的结果。</p>
          </details>
        </>
      ) : (
        <div className="b3-empty" role="status">
          <p className="note quiet">{status === 'unsupported' ? EMPTY_TEXT.unsupported : message || EMPTY_TEXT[status]}</p>
          {status === 'error' && <button type="button" className="text-btn" onClick={onRetry}>重新加载模型</button>}
          {status !== 'unsupported' && <p className="b3-caption">支持 BTC、ETH、BNB、SOL / USDT，四个周期分别评估。</p>}
        </div>
      )}
    </section>
  )
}

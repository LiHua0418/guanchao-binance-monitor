import type { B4Forecast } from '../lib/b4'
import { replayAuditMetrics } from '../lib/replay-audit'
import { INTERVAL_LABEL, INTERVAL_MS } from '../types'

type Props = { model: B4Forecast }
const CHECKPOINTS = [1, 3, 6, 12]
const percentage = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(4)}%` : '—'
const difference = (value: number | null) => value === null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(4)} pp`
const utc = (seconds: number) => new Date(seconds * 1000).toISOString().replace('T', ' ').replace('.000Z', ' UTC')

export function ReplayAudit({ model }: Props) {
  const metrics = replayAuditMetrics(model)
  const forward = model.forward
  const hasSettledForward = forward?.byHorizon.some((row) => row.samples > 0 && row.maePct !== null && row.baselineMaePct !== null) ?? false
  const currentWindowBetter = metrics.meanMaePct !== null && metrics.meanBaselineMaePct !== null
    && metrics.meanMaePct < metrics.meanBaselineMaePct
  const recent = forward?.recent.slice().sort((a, b) => b.issuedAt - a.issuedAt).slice(0, 3) ?? []
  return <>
    <section className="replay-audit b4-replay-summary" aria-label="同一起点12步复评">
      <h4>同一起点 12 步复评</h4>
      <p className="b4-evidence-note">{metrics.samples === 0 ? '价格预测优势尚未证明；当前没有完整的 12 步复评样本。'
        : currentWindowBetter ? '当前窗口价格误差较小，稳定优势仍需实时前向验证。' : '价格预测尚未优于持平基线。'}</p>
      <p className="b3-caption">{model.symbol} · {INTERVAL_LABEL[model.interval]} · {metrics.samples} 个共同起点。每个起点一次预测未来 12 根，期间不重新追随实际价格；每根仍是一个节点。</p>
      {metrics.samples > 0 ? <>
        <div className="b4-horizon-scroll"><table className="b4-horizons replay-audit-table">
          <thead><tr><th>预测步数</th><th>模型 MAE</th><th>持平 MAE</th><th>差值</th></tr></thead>
          <tbody>{metrics.byHorizon.filter((row) => CHECKPOINTS.includes(row.horizon)).map((row) => <tr key={row.horizon}>
            <td>+{row.horizon}</td><td className="num">{percentage(row.maePct)}</td><td className="num">{percentage(row.baselineMaePct)}</td><td className="num">{difference(row.differencePct)}</td>
          </tr>)}</tbody>
        </table></div>
        <p className="b3-caption">全部 1–12 步平均 MAE：模型 {percentage(metrics.meanMaePct)}，持平基线 {percentage(metrics.meanBaselineMaePct)}；上方结论使用这组全步平均误差。</p>
        <p className="b3-caption">持平基线在全部 12 步始终复制同一预测起点的收盘价。两列使用完全相同的 {metrics.samples} 个起点，以各步实际收盘价为分母；差值 = 模型 − 基线，负值才是误差更小。</p>
      </> : <p className="note quiet">需要拟合与校准截止之后、未来 12 根均已收盘的历史起点。滚动一步回放不会冒充这组完整路径样本。</p>}
      {metrics.rejectedRecords > 0 && <p className="b3-caption">{metrics.rejectedRecords} 条不完整、重复或未对齐的记录未纳入共同样本。</p>}
      <p className="b3-caption">这是冻结模型按历史行情重算的回放，不是当时发布的预测。滚动一步紫线每根都重设起点，贴近价格并不等于一次预测对了未来 12 根。</p>
    </section>
    <section className="forward-audit b4-replay-summary" aria-label="实时首发预测存档">
      <h4>实时首发预测存档</h4>
      {!hasSettledForward && <p className="b4-evidence-note">尚无已结算实时样本。</p>}
      {forward ? <>
        <dl className="stats b3-stats forward-audit-counts">
          <div><dt>已记录起点</dt><dd className="num">{forward.recordedOrigins}</dd></div>
          <div><dt>按时 / 迟到</dt><dd className="num">{forward.eligibleOrigins} / {forward.lateOrigins}</dd></div>
          <div><dt>按时 +1 已结算</dt><dd className="num">{forward.resolvedNextBarOrigins}</dd></div>
          <div><dt>12 步全部结算</dt><dd className="num">{forward.resolvedOrigins}</dd></div>
          <div><dt>仍待完整结算</dt><dd className="num">{forward.pendingOrigins}</dd></div>
        </dl>
        {hasSettledForward && <div className="b4-horizon-scroll"><table className="b4-horizons forward-audit-table">
          <thead><tr><th>预测步数</th><th>结算数</th><th>模型 MAE</th><th>持平 MAE</th></tr></thead>
          <tbody>{[1, 12].map((horizon) => {
            const row = forward.byHorizon.find((item) => item.horizon === horizon)
            const ready = !!row && row.samples > 0 && row.maePct !== null && row.baselineMaePct !== null
            return <tr key={horizon}><td>+{horizon}</td><td className="num">{row?.samples ?? 0}</td>
              <td className="num">{ready ? percentage(row.maePct) : '未结算'}</td><td className="num">{ready ? percentage(row.baselineMaePct) : '未结算'}</td></tr>
          })}</tbody>
        </table></div>}
        <p className="b3-caption">仅统计真实发出且按时的预测；每个起点首发不可覆盖。允许在起点收盘后最多延迟 {forward.eligibilityWindowSeconds} 秒发出，超过此窗口的记录保留但不计成绩。这不是提前完整一根 K 线发出的保证。</p>
        <p className="b3-caption">+1 与 +12 分别统计已结算样本；同一步模型与持平基线使用同样样本及实际价格分母。全部结算和待结算计数包含迟到记录，误差成绩不包含。</p>
        <details className="b4-detail-card"><summary>最近首发记录 <span>最多 3 个起点</span></summary>
          <div className="b4-detail-body">
            {recent.length ? <ol className="forward-audit-records">{recent.map((record) => {
              const settled = record.path.filter((point) => point.horizon >= 1 && point.horizon <= 12 && typeof point.actual === 'number' && Number.isFinite(point.actual) && point.actual > 0).length
              return <li key={record.originTime}>
                <strong>{record.eligibility === 'eligible' ? '按时首发' : '迟到 · 不计成绩'} · {settled === 12 ? '12 步已结算' : `已结算 ${settled}/12 · 待结算`}</strong>
                <span>实际发出：{utc(record.issuedAt)}</span>
                <span>起点 K 线：{utc(record.originTime)}；收盘：{utc(record.originTime + INTERVAL_MS[model.interval] / 1000)}</span>
              </li>
            })}</ol> : <p className="note quiet">尚未存入首发预测；历史重算不会导入实时成绩。</p>}
            <p className="b3-caption">模型版本：<span className="forward-audit-version">{forward.modelVersion}</span></p>
          </div>
        </details>
      </> : <p className="note quiet">当前服务尚未返回实时存档信息，历史重算不会视作真实前向成绩。</p>}
    </section>
  </>
}

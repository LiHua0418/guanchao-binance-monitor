import type { B4Forecast, B4Status } from '../lib/b4'
import { formatPrice } from '../lib/format'
import { ReplayAudit } from './ReplayAudit'

type Props = { model: B4Forecast | null; status: B4Status; message?: string; onRetry: () => void; intervalLabel: string }

export function B4Panel({ model, status, message, onRetry, intervalLabel }: Props) {
  const active = status === 'ready' ? model : null
  const next = active?.path[1]
  const evaluation = active?.evaluation
  return <section className="block b3-panel b4-panel" aria-label="guanchao-b4 模型">
    <div className="block-title"><p className="kicker">方向与幅度 · {intervalLabel}</p><span className="b3-badge">B4 · 实验</span></div>
    <h3 className="b3-name">guanchao-b4</h3>
    <p className="b4-scope-note">下一根为主 · 优势验证仅针对下一根，后续步数为实验输出</p>
    {active && next && evaluation ? <>
      <div className="b4-summary">
        <div className="b4-probability-card">
          <div className="b4-probabilities">
            <span>下一根上涨概率 <strong className="num b4-primary-value">{((next.upProbability ?? 0.5) * 100).toFixed(1)}%</strong></span>
            <span>下跌概率 <strong className="num b4-primary-value">{((1 - (next.upProbability ?? 0.5)) * 100).toFixed(1)}%</strong></span>
          </div>
          <div className="b4-probability-bar" aria-hidden="true"><i style={{ width: `${(next.upProbability ?? 0.5) * 100}%` }} /></div>
          <p className="b4-summary-note">{evaluation.direction.cvPassed ? '方向信号较弱，接近 50% 时没有明确倾向。' : '这个周期的方向模型未通过滚动验证，概率仅作实验观察。'}</p>
          <p className="b4-summary-note">概率以发生涨跌为条件，训练不计持平样本。</p>
        </div>
        <div className="b4-magnitude-card">
          <span>下一根 · 预期变化幅度</span>
          <strong className="num b4-primary-value">{next.expectedMovePct.toFixed(3)}%</strong>
          <span className="b4-scope">不含涨跌方向</span>
          <p className="b4-summary-note">{evaluation.magnitude.cvPassed && evaluation.magnitude.heldoutSkill ? '下一根幅度：滚动验证通过。' : '下一根幅度优势尚未通过全部验证。'}</p>
        </div>
      </div>
      <div className="b4-scenarios">
        <h4>下一根 · 同幅度上下行情景</h4>
        <div className="b4-scenario-prices">
          <div><span>下行情景</span><strong className="num">{formatPrice(next.downScenario)}</strong></div>
          <div><span>上行情景</span><strong className="num">{formatPrice(next.upScenario)}</strong></div>
        </div>
        <p className="b4-summary-note">上下行情景不是目标价或置信区间。</p>
      </div>
      <ReplayAudit model={active} />
      <details className="b4-detail-card"><summary>逐步明细 <span>未来 12 根 · 每根一个节点</span></summary>
        <div className="b4-detail-body">
          <div className="b4-horizon-scroll"><table className="b4-horizons">
            <thead><tr><th>步数</th><th>幅度</th><th>上涨概率</th></tr></thead>
            <tbody>{active.path.slice(1).map((point, i) => <tr key={point.time}><td>+{i + 1}</td><td className="num">{point.expectedMovePct.toFixed(3)}%</td><td className="num">{point.upProbability === undefined ? '—' : `${(point.upProbability * 100).toFixed(1)}%`}</td></tr>)}</tbody>
          </table></div>
          <p className="b3-caption">幅度为相对当前收盘的累计变化；概率仅训练 +1、+3、+6、+12 四个独立目标，其余不插值。优势验证仅针对下一根。</p>
        </div>
      </details>
      <details className="b4-detail-card"><summary>历史评估 <span>下一根 · 留出时段复评</span></summary>
        <div className="b4-detail-body">
          <dl className="stats b3-stats">
            <div><dt>方向平衡准确率</dt><dd className="num">{(evaluation.direction.balancedAccuracy * 100).toFixed(2)}%</dd></div>
            <div><dt>上一根反转基线</dt><dd className="num">{(evaluation.direction.baselineBalancedAccuracy * 100).toFixed(2)}%</dd></div>
            <div><dt>幅度 MSE 改善</dt><dd className="num">{(evaluation.magnitude.mseImprovement * 100).toFixed(2)}%</dd></div>
            <div><dt>幅度 MAE 改善</dt><dd className="num">{(evaluation.magnitude.maeImprovement * 100).toFixed(2)}%</dd></div>
          </dl>
          <p className="b3-caption">幅度基线：过去 20 根平均绝对变化。正值表示误差下降；{evaluation.samples.toLocaleString('zh-CN')} 个四币种复评起点。同一历史时段已被版本复评，不等于新的实盘结果。</p>
          <p className="b4-evidence-note">{evaluation.magnitude.cvPassed && evaluation.magnitude.heldoutSkill ? '下一根幅度通过滚动验证，并在历史复评中优于波动基线。' : '下一根幅度优势尚未通过全部验证。'} 完整价格路径仍未证明稳定优势；后续步数是实验输出。</p>
        </div>
      </details>
      <details className="b4-detail-card"><summary>模型说明 <span>概率质量与数据范围</span></summary>
        <div className="b4-detail-body">
          <p className="b3-architecture">独立方向概率与幅度模型 · CatBoost</p>
          <p className="b3-caption">按预测的绝对对数变化幅度换算，上下行情景不是目标价或置信区间。蓝色均值线接近水平时，仍可通过情景线查看波动规模。</p>
          <dl className="b4-metadata">
            <div><dt>模型 Brier / 训练先验 Brier（越低越好）</dt><dd className="num">{evaluation.direction.brier.toFixed(5)} / {evaluation.direction.baselineBrier.toFixed(5)}</dd></div>
            <div><dt>拟合与校准截止 · UTC</dt><dd>{evaluation.dataCutoff}</dd></div>
            <div><dt>数据截止 · UTC</dt><dd>{evaluation.dataEnd}</dd></div>
          </dl>
          <p className="b3-caption">57 项因果特征，四个币种、四个周期分别训练。方向概率、价格均值和绝对幅度独立学习，不能相乘后当作已验证的价格预测。</p>
        </div>
      </details>
    </> : <div className="b3-empty" role="status">
      <p className="note quiet">{message || (status === 'loading' ? '正在计算方向概率、幅度和逐根回放…' : '当前没有可用的 B4 结果。')}</p>
      {status === 'error' && <><button type="button" className="text-btn" onClick={onRetry}>重新连接模型</button><p className="b3-caption">请在项目目录运行 npm run model:serve。也可切换 B3 查看已保存的旧模型。</p></>}
    </div>}
  </section>
}

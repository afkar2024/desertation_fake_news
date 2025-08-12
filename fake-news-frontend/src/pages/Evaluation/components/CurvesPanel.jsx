import BaseChart from '../../../components/charts/BaseChart'

export default function CurvesPanel({ roc, pr, calibration }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {roc && (
        <div className="h-64">
          <BaseChart type="line" data={{
            datasets: [
              { label: 'ROC', data: (roc.points || []).map(p => ({ x: p.fpr, y: p.tpr })), borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.2, pointRadius: 0 },
              { label: 'Random', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: 'rgba(156,163,175,0.8)', borderDash: [5,5], fill: false, pointRadius: 0 }
            ]
          }} options={{ plugins: { legend: { display: false }, title: { display: !!roc.auc, text: roc.auc ? `AUC: ${roc.auc}` : '' } }, scales: { x: { min: 0, max: 1, title: { display: true, text: 'FPR' } }, y: { min: 0, max: 1, title: { display: true, text: 'TPR' } } } }} />
          <p className="text-xs text-gray-500 mt-1">ROC Curve: trade-off between true positive rate and false positive rate.</p>
        </div>
      )}
      {pr && (
        <div className="h-64">
          <BaseChart type="line" data={{
            datasets: [
              { label: 'PR', data: (pr.points || []).map(p => ({ x: p.recall, y: p.precision })), borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', fill: true, tension: 0.2, pointRadius: 0 }
            ]
          }} options={{ plugins: { legend: { display: false }, title: { display: !!pr.auc, text: pr.auc ? `PR-AUC: ${pr.auc}` : '' } }, scales: { x: { min: 0, max: 1, title: { display: true, text: 'Recall' } }, y: { min: 0, max: 1, title: { display: true, text: 'Precision' } } } }} />
          <p className="text-xs text-gray-500 mt-1">Precision-Recall Curve: performance under class imbalance.</p>
        </div>
      )}
      {calibration?.bins?.length > 0 && (
        <div className="h-64 lg:col-span-2">
          <BaseChart type="line" data={{
            datasets: [
              { label: 'Perfect', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: 'rgba(107,114,128,0.8)', borderDash: [4,4], fill: false, pointRadius: 0 },
              { label: 'Reliability', data: calibration.bins.map(b => ({ x: b.avg_conf ?? ((b.low + b.high) / 2), y: b.accuracy })), borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.2, pointRadius: 2 }
            ]
          }} options={{ plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 1, title: { display: true, text: 'Confidence' } }, y: { min: 0, max: 1, title: { display: true, text: 'Observed Accuracy' } } } }} />
          <p className="text-xs text-gray-500 mt-1">Reliability Diagram: perfectly calibrated models lie on the diagonal.</p>
        </div>
      )}
    </div>
  )
}



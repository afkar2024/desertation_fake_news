import BaseChart from '../../../components/charts/BaseChart'

export default function BaselinesPanel({ modelAcc, baselineAcc, traditionalAcc }) {
  if (modelAcc == null && baselineAcc == null && traditionalAcc == null) return null
  const labels = ['Model']
  const data = [modelAcc || 0]
  if (baselineAcc != null) { labels.push('Heuristic'); data.push(baselineAcc) }
  if (traditionalAcc != null) { labels.push('Traditional'); data.push(traditionalAcc) }
  return (
    <div className="bg-white border rounded p-4">
      <div className="text-sm font-semibold mb-2">Baseline Comparison</div>
      <div className="h-48">
        <BaseChart type="bar" data={{ labels, datasets: [{ label: 'Accuracy', data, backgroundColor: ['#3b82f6', '#f97316', '#10b981'] }] }} options={{ scales: { y: { min: 0, max: 1 } } }} />
      </div>
      <p className="text-xs text-gray-500 mt-1">Comparing model against heuristic and traditional baselines.</p>
    </div>
  )
}



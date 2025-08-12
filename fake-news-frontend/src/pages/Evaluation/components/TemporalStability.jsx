import BaseChart from '../../../components/charts/BaseChart'

export default function TemporalStability({ periods }) {
  if (!periods) return null
  const labels = Object.keys(periods)
  const data = Object.values(periods).map(v => v.accuracy)
  return (
    <div className="bg-white border rounded p-4">
      <div className="text-sm font-semibold mb-2">Temporal Stability</div>
      <div className="h-48">
        <BaseChart type="line" data={{ labels, datasets: [{ label: 'Accuracy', data, borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.1)', tension: 0.2, fill: true }] }} options={{ scales: { y: { min: 0, max: 1 } } }} />
      </div>
      <p className="text-xs text-gray-500 mt-1">Accuracy over time; variability indicates stability.</p>
    </div>
  )
}



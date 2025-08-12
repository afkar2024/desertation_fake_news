import BaseChart from '../../../components/charts/BaseChart'

export default function AbstentionPanel({ curve = [] }) {
  if (!Array.isArray(curve) || curve.length === 0) return null
  return (
    <div className="bg-white border rounded p-4">
      <div className="text-sm font-semibold mb-2">Coverage vs Accuracy</div>
      <div className="h-48">
        <BaseChart type="line" data={{
          labels: curve.map(c => c.coverage),
          datasets: [{ label: 'Accuracy', data: curve.map(c => c.accuracy), borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.1)', fill: true, tension: 0.2 }]
        }} options={{ scales: { x: { min: 0, max: 1 }, y: { min: 0, max: 1 } } }} />
      </div>
      <p className="text-xs text-gray-500 mt-1">Accuracy as coverage increases from most confident predictions to all predictions.</p>
    </div>
  )
}



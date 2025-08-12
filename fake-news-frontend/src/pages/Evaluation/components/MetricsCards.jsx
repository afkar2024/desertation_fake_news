export default function MetricsCards({ metrics = {} }) {
  const items = [
    { key: 'accuracy', label: 'Accuracy', help: 'Overall fraction of correct predictions.' },
    { key: 'f1', label: 'F1', help: 'Harmonic mean of precision and recall.' },
    { key: 'precision', label: 'Precision', help: 'Among predicted fake, how many were actually fake.' },
    { key: 'recall', label: 'Recall', help: 'Among actual fake, how many were correctly identified.' },
  ]
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
      {items.map(({ key, label, help }) => (
        <div key={key} className="border rounded bg-white p-4">
          <div className="text-xs text-gray-500">{label}</div>
          <div className="text-lg font-semibold">{metrics[key] != null ? Number(metrics[key]).toFixed(3) : '—'}</div>
          <div className="text-[11px] text-gray-500 mt-1 leading-4">{help}</div>
        </div>
      ))}
    </div>
  )
}



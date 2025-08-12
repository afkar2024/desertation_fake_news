export default function FairnessTable({ groups }) {
  if (!groups) return null
  const entries = Object.entries(groups)
  return (
    <div className="bg-white border rounded p-4">
      <div className="text-sm font-semibold mb-2">Fairness (Group Accuracies)</div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead><tr><th className="text-left p-2">Group</th><th className="text-left p-2">Accuracy</th><th className="text-left p-2">N</th></tr></thead>
          <tbody>
            {entries.map(([group, v]) => (
              <tr key={group} className="border-t"><td className="p-2">{group}</td><td className="p-2">{v.accuracy}</td><td className="p-2">{v.n}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-gray-500 mt-1">Per-group accuracy; large gaps can indicate bias.</p>
    </div>
  )
}



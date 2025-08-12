export default function ConfusionMatrix({ matrix }) {
  if (!Array.isArray(matrix) || matrix.length !== 2) return null
  const [row0, row1] = matrix
  return (
    <div>
      <div className="text-sm text-gray-500 mb-2">Confusion Matrix (rows: actual, cols: predicted)</div>
      <div className="grid grid-cols-2 w-64 text-center border rounded overflow-hidden">
        {[...row0, ...row1].map((v, i) => (
          <div key={i} className={`p-3 ${i === 0 || i === 3 ? 'bg-green-50' : 'bg-red-50'} border`}>{v}</div>
        ))}
      </div>
    </div>
  )
}



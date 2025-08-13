export default function SignificancePanel({ b, c, chi2, p }) {
  if (b == null && c == null && chi2 == null && p == null) return null
  const significant = typeof p === 'number' ? p < 0.05 : false
  return (
    <div className="bg-white border rounded p-4">
      <div className="text-sm font-semibold mb-2">Statistical Significance (McNemar's Test)</div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <div className="border rounded p-3 bg-gray-50">
          <div className="text-gray-500 text-xs">b (Model correct, Baseline wrong)</div>
          <div className="text-lg font-semibold">{b ?? '-'}</div>
        </div>
        <div className="border rounded p-3 bg-gray-50">
          <div className="text-gray-500 text-xs">c (Model wrong, Baseline correct)</div>
          <div className="text-lg font-semibold">{c ?? '-'}</div>
        </div>
        <div className="border rounded p-3 bg-gray-50">
          <div className="text-gray-500 text-xs">Chi-square (χ²)</div>
          <div className="text-lg font-semibold">{chi2 ?? '-'}</div>
        </div>
        <div className={`border rounded p-3 ${significant ? 'bg-green-50' : 'bg-yellow-50'}`}> 
          <div className="text-gray-500 text-xs">p-value</div>
          <div className="text-lg font-semibold">{p ?? '-'}</div>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-2">McNemar's test assesses whether the difference between two classifiers on the same dataset is statistically significant (α = 0.05).</p>
    </div>
  )
}



export default function CalibrationSummary({ brier, ece, rocAuc, prAuc }) {
  if (brier == null && ece == null && rocAuc == null && prAuc == null) return null
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      {brier != null && (
        <div className="bg-white border rounded p-4">
          <div className="text-sm font-semibold mb-1">Brier Score</div>
          <div className="text-2xl font-semibold">{brier}</div>
          <div className="text-xs text-gray-500 mt-1">Mean squared error between predicted probabilities and true labels (lower is better).</div>
        </div>
      )}
      {ece != null && (
        <div className="bg-white border rounded p-4">
          <div className="text-sm font-semibold mb-1">ECE</div>
          <div className="text-2xl font-semibold">{ece}</div>
          <div className="text-xs text-gray-500 mt-1">Expected Calibration Error. Aggregated gap between confidence and accuracy (lower is better).</div>
        </div>
      )}
      {rocAuc != null && (
        <div className="bg-white border rounded p-4">
          <div className="text-sm font-semibold mb-1">ROC-AUC</div>
          <div className="text-2xl font-semibold">{rocAuc}</div>
          <div className="text-xs text-gray-500 mt-1">Area under ROC curve; probability a random positive ranks above a random negative.</div>
        </div>
      )}
      {prAuc != null && (
        <div className="bg-white border rounded p-4">
          <div className="text-sm font-semibold mb-1">PR-AUC</div>
          <div className="text-2xl font-semibold">{prAuc}</div>
          <div className="text-xs text-gray-500 mt-1">Area under Precision-Recall curve; emphasizes performance on the positive class.</div>
        </div>
      )}
    </div>
  )
}



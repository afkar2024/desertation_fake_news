import { useMemo, useState } from 'react'
import { predictText, predictUrl, predictBatch, predictExplain } from '../services/api'
import BaseChart from '../components/charts/BaseChart'
import { useNotifyStore } from '../stores/notifyStore'

export default function Analysis() {
  const [text, setText] = useState('')
  const [url, setUrl] = useState('')
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [batchText, setBatchText] = useState('')
  const [batchSummary, setBatchSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const push = useNotifyStore(s => s.push)
  const [includeExplanation, setIncludeExplanation] = useState(true)

  const analyzeText = async () => {
    setLoading(true)
    setError('')
    try {
      const data = includeExplanation
        ? await predictExplain({ text, top_tokens: 20 })
        : await predictText(text)
      setResult(data)
      setHistory((h) => [{ type: 'text', input: text.slice(0, 140), result: data, ts: Date.now() }, ...h].slice(0, 20))
      push({ title: 'Analyzed text', message: 'Prediction complete' })
    } catch (e) {
      setError(e.message || 'predict_failed')
      push({ title: 'Analysis failed', message: e.message || 'predict_failed', variant: 'error' })
    } finally {
      setLoading(false)
    }
  }

  const analyzeUrl = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await predictUrl(url)
      setResult(data)
      setHistory((h) => [{ type: 'url', input: url, result: data, ts: Date.now() }, ...h].slice(0, 20))
      push({ title: 'Analyzed URL', message: 'Prediction complete' })
    } catch (e) {
      setError(e.message || 'predict_url_failed')
      push({ title: 'URL analysis failed', message: e.message || 'predict_url_failed', variant: 'error' })
    } finally {
      setLoading(false)
    }
  }

  const analyzeBatch = async () => {
    const items = batchText
      .split(/\r?\n/)
      .map(s => s.trim())
      .filter(Boolean)
    if (items.length === 0) return
    setLoading(true)
    setError('')
    try {
      const data = await predictBatch(items)
      const n = Array.isArray(data) ? data.length : 0
      const avgFake = n ? (data.reduce((acc, r) => acc + (r?.prob_fake ?? 0), 0) / n) : 0
      const numFake = n ? data.filter(r => (r?.prob_fake ?? 0) > 0.5).length : 0
      const summary = { count: n, avgProbFake: avgFake, numPredFake: numFake }
      setBatchSummary(summary)
      push({ title: 'Batch analyzed', message: `${n} items processed` })
    } catch (e) {
      setError(e.message || 'predict_batch_failed')
      push({ title: 'Batch analysis failed', message: e.message || 'predict_batch_failed', variant: 'error' })
    } finally {
      setLoading(false)
    }
  }

  const onUploadFile = async (file) => {
    if (!file) return
    const text = await file.text()
    // naive parse: try JSON array, else use lines
    try {
      const parsed = JSON.parse(text)
      if (Array.isArray(parsed)) {
        setBatchText(parsed.map(String).join('\n'))
        return
      }
    } catch {}
    setBatchText(text)
  }

  // Resolve probabilities from any shape the backend returns
  const probFake = (((result?.probabilities?.fake) ?? (result?.explanation?.probabilities?.fake) ?? (result?.prob_fake)) ?? 0)
  const probReal = (((result?.probabilities?.real) ?? (result?.explanation?.probabilities?.real) ?? (result?.prob_real)) ?? 0)
  const labelText = (result?.label) ?? ((typeof result?.prediction !== 'undefined' && result?.prediction !== null)
    ? String(result.prediction)
    : ((probFake > 0.5) ? 'fake' : 'real'))

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Article Analysis</h1>
      <Guide />
      {error && <div className="p-3 text-sm rounded bg-red-50 text-red-700">{error}</div>}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm text-gray-700">Analyze Text</label>
            <textarea className="w-full border rounded p-2 h-40" value={text} onChange={e => setText(e.target.value)} />
            <label className="flex items-center gap-2 text-sm text-gray-600">
              <input type="checkbox" className="accent-blue-600" checked={includeExplanation} onChange={e => setIncludeExplanation(e.target.checked)} />
              Include explanation (SHAP)
            </label>
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !text.trim()} onClick={analyzeText}>
              {loading ? 'Analyzing…' : 'Analyze Text'}
            </button>
          </div>
          <div className="space-y-2">
            <label className="text-sm text-gray-700">Analyze URL</label>
            <input className="w-full border rounded p-2" placeholder="https://…" value={url} onChange={e => setUrl(e.target.value)} />
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !url.trim()} onClick={analyzeUrl}>
              {loading ? 'Analyzing…' : 'Analyze URL'}
            </button>
          </div>
          <div className="space-y-2">
            <label className="text-sm text-gray-700">Batch (one per line or upload .txt/.csv/.json)</label>
            <textarea className="w-full border rounded p-2 h-32" placeholder={"Enter one text per line…"} value={batchText} onChange={e => setBatchText(e.target.value)} />
            <div className="flex items-center gap-2">
              <input type="file" accept=".txt,.csv,.json" onChange={e => onUploadFile(e.target.files?.[0])} />
              <button className="px-3 py-2 rounded bg-slate-700 text-white disabled:opacity-50" disabled={loading || !batchText.trim()} onClick={analyzeBatch}>
                {loading ? 'Analyzing…' : 'Analyze Batch'}
              </button>
            </div>
            {batchSummary && (
              <div className="text-xs text-gray-600">
                Count: {batchSummary.count} • Avg P(fake): {batchSummary.avgProbFake.toFixed(3)} • Predicted Fake: {batchSummary.numPredFake}
              </div>
            )}
          </div>
        </div>
        <div>
          {result ? (
            <div className="p-4 border rounded bg-white space-y-3">
              <div className="text-sm text-gray-500">Prediction</div>
              <div className="text-lg font-semibold">{labelText}</div>
              <div className="text-sm">P(fake): {probFake.toFixed(3)} | P(real): {probReal.toFixed(3)}</div>
              {result?.uncertainty?.entropy != null && (
                <div className="text-sm text-gray-600">Entropy: {result.uncertainty.entropy.toFixed(3)}</div>
              )}
              <div className="h-48">
                <BaseChart
                  type="bar"
                  data={{
                    labels: ['Real', 'Fake'],
                    datasets: [{
                      label: 'Probability',
                      data: [probReal, probFake],
                      backgroundColor: ['rgba(34,197,94,0.7)','rgba(239,68,68,0.7)']
                    }]
                  }}
                  options={{
                    scales: { y: { min: 0, max: 1 } },
                    plugins: { legend: { display: false } }
                  }}
                />
              </div>
              {result?.explanation?.tokens?.length > 0 && (
                <div className="mt-3">
                  <div className="text-sm text-gray-500 mb-1">Top Tokens (SHAP)</div>
                  <div className="flex flex-wrap gap-1">
                    {result.explanation.tokens.map((tok, i) => {
                      const val = result.explanation.shap_values?.[i] ?? 0
                      const abs = Math.min(Math.abs(val), 1.0)
                      const bg = val > 0 ? `rgba(239,68,68,${0.15 + abs*0.6})` : `rgba(34,197,94,${0.15 + abs*0.6})`
                      return <span key={i} className="px-2 py-1 rounded text-xs" style={{ background: bg }}>{tok}</span>
                    })}
                  </div>
                  <div className="text-[11px] text-gray-500 mt-1">Red increases fake probability; green increases real probability. Intensity reflects magnitude.</div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-gray-500">No result yet</div>
          )}
          {history.length > 0 && (
            <div className="mt-6">
              <div className="text-sm font-medium mb-2">History</div>
              <div className="space-y-2">
                {history.map((item, idx) => (
                  <div key={idx} className="border rounded p-2 text-sm flex items-center justify-between">
                    <div className="truncate mr-4">
                      <span className="text-gray-500 mr-2">[{item.type}]</span>
                      <span className="truncate inline-block max-w-[420px] align-middle">{item.input}</span>
                    </div>
                    <div className="text-xs text-gray-600">P(fake): {(item.result?.prob_fake ?? 0).toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


function Guide() {
  return (
    <div className="bg-white border rounded p-4 text-sm text-gray-700">
      <div className="font-semibold mb-1">How to use the Analysis page</div>
      <ul className="list-disc pl-5 space-y-1">
        <li><span className="font-medium">Analyze Text</span>: paste any article text and click “Analyze Text”. Optionally enable “Include explanation (SHAP)” to see top influential tokens.</li>
        <li><span className="font-medium">Analyze URL</span>: paste a news article URL; the backend fetches content and classifies it.</li>
        <li><span className="font-medium">Batch Analysis</span>: enter one text per line or upload a .txt/.csv/.json; we’ll classify each and show a quick summary.</li>
        <li><span className="font-medium">Interpreting results</span>:
          <ul className="list-disc pl-5 mt-1">
            <li><span className="font-medium">P(fake)/P(real)</span> are the model’s predicted probabilities.</li>
            <li><span className="font-medium">Entropy</span> quantifies prediction uncertainty (higher = more uncertain).</li>
            <li><span className="font-medium">SHAP tokens</span>: red tokens push towards “fake”, green towards “real”. Intensity indicates impact magnitude.</li>
          </ul>
        </li>
        <li><span className="font-medium">Tips</span>: shorter inputs run faster; for URLs, ensure the page is publicly accessible.</li>
      </ul>
    </div>
  )
}



import { useState } from 'react'
import { Tab } from '@headlessui/react'
import { predictExplain, predictCounterfactual } from '../services/api'
import { useNotifyStore } from '../stores/notifyStore'
import BaseChart from '../components/charts/BaseChart'

export default function Explainability() {
  const [text, setText] = useState('')
  const [useLime, setUseLime] = useState(false)
  const [useAttention, setUseAttention] = useState(false)
  const [explanation, setExplanation] = useState(null)
  const [counterfactual, setCounterfactual] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const push = useNotifyStore(s => s.push)

  const explain = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await predictExplain({ text, use_lime: useLime, use_attention: useAttention })
      setExplanation(data)
      push({ title: 'Explanation ready', message: (useLime ? 'LIME' : (useAttention ? 'Attention' : 'SHAP')) + ' generated' })
    } catch (e) {
      setError(e.message || 'explain_failed')
      push({ title: 'Explain failed', message: e.message || 'explain_failed', variant: 'error' })
    } finally {
      setLoading(false)
    }
  }

  const generateCounterfactual = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await predictCounterfactual({ text })
      setCounterfactual(data)
      push({ title: 'Counterfactual generated' })
    } catch (e) {
      setError(e.message || 'counterfactual_failed')
      push({ title: 'Counterfactual failed', message: e.message || 'counterfactual_failed', variant: 'error' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Explainability</h1>
      {error && <div className="p-3 text-sm rounded bg-red-50 text-red-700">{error}</div>}
      <textarea className="w-full border rounded p-2 h-40" value={text} onChange={e => setText(e.target.value)} />

      <Tab.Group>
        <Tab.List className="flex space-x-1 rounded bg-gray-100 p-1 w-max">
          {['SHAP','LIME','Attention','Counterfactual'].map((t) => (
            <Tab key={t} className={({ selected }) => `px-3 py-1.5 text-sm rounded ${selected ? 'bg-white shadow' : 'hover:bg-white/60'}`}>{t}</Tab>
          ))}
        </Tab.List>
        <Tab.Panels className="mt-4">
          <Tab.Panel>
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !text.trim()} onClick={() => { setUseLime(false); setUseAttention(false); explain() }}>
              {loading ? 'Explaining…' : 'Explain with SHAP'}
            </button>
            {explanation && (
              <div className="p-4 border rounded bg-white space-y-4 mt-3">
                <div className="text-sm text-gray-500">SHAP Explanation</div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="border rounded p-3 bg-gray-50">
                    <div className="text-xs text-gray-500">Prediction</div>
                    <div className="text-lg font-semibold capitalize">{explanation?.prediction}</div>
                  </div>
                  <div className="border rounded p-3 bg-gray-50">
                    <div className="text-xs text-gray-500">Confidence</div>
                    <div className="text-lg font-semibold">{(explanation?.confidence ?? 0).toFixed?.(3)}</div>
                  </div>
                  <div className="border rounded p-3 bg-gray-50">
                    <div className="text-xs text-gray-500">Probabilities</div>
                    <div className="text-xs">Real: {(explanation?.probabilities?.real ?? 0).toFixed?.(3)} | Fake: {(explanation?.probabilities?.fake ?? 0).toFixed?.(3)}</div>
                  </div>
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Top Tokens by |SHAP|</div>
                    <div className="flex flex-wrap gap-1">
                      {(explanation?.explanation?.tokens || []).map((t, i) => (
                        <span key={i} className="px-2 py-1 rounded text-xs" style={{ backgroundColor: (explanation?.explanation?.shap_values?.[i] || 0) > 0 ? 'rgba(244,67,54,0.15)' : 'rgba(16,185,129,0.15)'}}>{t}</span>
                      ))}
                    </div>
                  </div>
                  <div className="h-48">
                    <BaseChart type="bar" data={{
                      labels: (explanation?.explanation?.tokens || []).slice(0, 20),
                      datasets: [{ label: 'SHAP', data: (explanation?.explanation?.shap_values || []).slice(0, 20), backgroundColor: (explanation?.explanation?.shap_values || []).slice(0, 20).map(v => v > 0 ? 'rgba(244,67,54,0.8)' : 'rgba(16,185,129,0.8)') }]
                    }} options={{ indexAxis: 'y', plugins: { legend: { display: false } } }} />
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="border rounded p-3">
                    <div className="text-xs text-gray-500 mb-2">Top Positive Contributors (towards Fake)</div>
                    <ul className="text-sm space-y-1">
                      {(() => {
                        const toks = explanation?.explanation?.tokens || []
                        const vals = explanation?.explanation?.shap_values || []
                        const pairs = toks.map((t, i) => ({ t, v: vals[i] || 0 }))
                        return pairs.filter(x => x.v > 0).sort((a,b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 10).map((x, i) => (
                          <li key={i} className="flex justify-between"><span>"{x.t}"</span><span>{x.v.toFixed?.(4)}</span></li>
                        ))
                      })()}
                    </ul>
                  </div>
                  <div className="border rounded p-3">
                    <div className="text-xs text-gray-500 mb-2">Top Negative Contributors (towards Real)</div>
                    <ul className="text-sm space-y-1">
                      {(() => {
                        const toks = explanation?.explanation?.tokens || []
                        const vals = explanation?.explanation?.shap_values || []
                        const pairs = toks.map((t, i) => ({ t, v: vals[i] || 0 }))
                        return pairs.filter(x => x.v < 0).sort((a,b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 10).map((x, i) => (
                          <li key={i} className="flex justify-between"><span>"{x.t}"</span><span>{x.v.toFixed?.(4)}</span></li>
                        ))
                      })()}
                    </ul>
                  </div>
                </div>
                <div className="text-xs text-gray-500">Red bars increase fake probability; green bars decrease it. Base value: {(explanation?.explanation?.base_value ?? 0).toFixed?.(4)}.</div>
              </div>
            )}
          </Tab.Panel>
          <Tab.Panel>
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !text.trim()} onClick={() => { setUseLime(true); setUseAttention(false); explain() }}>
              {loading ? 'Explaining…' : 'Explain with LIME'}
            </button>
            {explanation && (
              <div className="p-4 border rounded bg-white space-y-4 mt-3">
                <div className="text-sm text-gray-500">LIME Explanation</div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div className="h-56">
                    <BaseChart type="bar" data={{
                      labels: (explanation?.explanation?.features || []).slice(0, 20),
                      datasets: [{ label: 'Weight', data: (explanation?.explanation?.weights || []).slice(0, 20), backgroundColor: '#3b82f6' }]
                    }} options={{ indexAxis: 'y', plugins: { legend: { display: false } } }} />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="border rounded p-3">
                      <div className="text-xs text-gray-500 mb-2">Top Positive Features</div>
                      <ul className="text-sm space-y-1">
                        {(() => {
                          const feats = explanation?.explanation?.features || []
                          const w = explanation?.explanation?.weights || []
                          const pairs = feats.map((f, i) => ({ f, v: w[i] || 0 }))
                          return pairs.filter(x => x.v > 0).sort((a,b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 8).map((x, i) => (
                            <li key={i} className="flex justify-between"><span>{x.f}</span><span>{x.v.toFixed?.(4)}</span></li>
                          ))
                        })()}
                      </ul>
                    </div>
                    <div className="border rounded p-3">
                      <div className="text-xs text-gray-500 mb-2">Top Negative Features</div>
                      <ul className="text-sm space-y-1">
                        {(() => {
                          const feats = explanation?.explanation?.features || []
                          const w = explanation?.explanation?.weights || []
                          const pairs = feats.map((f, i) => ({ f, v: w[i] || 0 }))
                          return pairs.filter(x => x.v < 0).sort((a,b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 8).map((x, i) => (
                            <li key={i} className="flex justify-between"><span>{x.f}</span><span>{x.v.toFixed?.(4)}</span></li>
                          ))
                        })()}
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="text-xs text-gray-500">Positive weights push towards fake, negative towards real.</div>
              </div>
            )}
          </Tab.Panel>
          <Tab.Panel>
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !text.trim()} onClick={() => { setUseLime(false); setUseAttention(true); explain() }}>
              {loading ? 'Explaining…' : 'Explain with Attention'}
            </button>
            {explanation && (
              <div className="p-4 border rounded bg-white space-y-4 mt-3">
                <div className="text-sm text-gray-500">Attention Weights</div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div className="flex flex-wrap gap-1">
                    {(explanation?.explanation?.attention_tokens || []).slice(0, 60).map((t, i) => (
                      <span key={i} className="px-2 py-1 rounded text-xs" style={{ backgroundColor: `rgba(59,130,246,${(explanation?.explanation?.attention_weights?.[i] || 0) * 0.7 + 0.1})` }}>{t}</span>
                    ))}
                  </div>
                  <div className="h-48">
                    <BaseChart type="bar" data={{
                      labels: (explanation?.explanation?.attention_tokens || []).slice(0, 20),
                      datasets: [{ label: 'Attention', data: (explanation?.explanation?.attention_weights || []).slice(0, 20), backgroundColor: 'rgba(59,130,246,0.8)' }]
                    }} options={{ indexAxis: 'y', plugins: { legend: { display: false } } }} />
                  </div>
                </div>
                <div className="text-xs text-gray-500">Darker tokens received higher attention from the [CLS] token. Bar chart shows top weights.</div>
              </div>
            )}
          </Tab.Panel>
          <Tab.Panel>
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !text.trim()} onClick={generateCounterfactual}>
              {loading ? 'Generating…' : 'Generate Counterfactual'}
            </button>
            {counterfactual && (
              <div className="p-4 border rounded bg-white space-y-4 mt-3">
                <div className="text-sm text-gray-500">Counterfactual Output</div>
                <div className="text-xs text-gray-500">Base text preview:</div>
                <div className="text-xs p-2 bg-gray-50 border rounded max-h-28 overflow-auto">{counterfactual?.base_text}</div>
                <div className="space-y-2">
                  {(counterfactual?.counterfactuals || []).map((c, i) => (
                    <div key={i} className="border rounded p-3 bg-gray-50">
                      <div className="flex items-center justify-between text-xs text-gray-500 mb-2">
                        <div>Variant #{i + 1}</div>
                        <div>Δ P(fake): {c.delta_fake_probability?.toFixed?.(4)}</div>
                      </div>
                      <div className="text-sm mb-2">{c.text}</div>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                        <div className="border rounded p-2 bg-white">
                          <div className="text-gray-500">Prediction</div>
                          <div className="font-semibold capitalize">{c.prediction === 1 ? 'fake' : 'real'}</div>
                        </div>
                        <div className="border rounded p-2 bg-white">
                          <div className="text-gray-500">Confidence</div>
                          <div className="font-semibold">{(c.confidence ?? 0).toFixed?.(3)}</div>
                        </div>
                        <div className="border rounded p-2 bg-white">
                          <div className="text-gray-500">Probabilities</div>
                          <div>Real: {(c?.probabilities?.real ?? 0).toFixed?.(3)} | Fake: {(c?.probabilities?.fake ?? 0).toFixed?.(3)}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="text-xs text-gray-500">Each variant removes influential tokens to observe prediction shifts.</div>
              </div>
            )}
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  )
}



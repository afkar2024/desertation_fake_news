import { useState } from 'react'
import { Tab } from '@headlessui/react'
import { predictExplain, predictCounterfactual } from '../services/api'
import { useNotifyStore } from '../stores/notifyStore'

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
              <div className="p-4 border rounded bg-white space-y-3 mt-3">
                <div className="text-sm text-gray-500">SHAP Explanation</div>
                <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(explanation, null, 2)}</pre>
              </div>
            )}
          </Tab.Panel>
          <Tab.Panel>
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !text.trim()} onClick={() => { setUseLime(true); setUseAttention(false); explain() }}>
              {loading ? 'Explaining…' : 'Explain with LIME'}
            </button>
            {explanation && (
              <div className="p-4 border rounded bg-white space-y-3 mt-3">
                <div className="text-sm text-gray-500">LIME Explanation</div>
                <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(explanation, null, 2)}</pre>
              </div>
            )}
          </Tab.Panel>
          <Tab.Panel>
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !text.trim()} onClick={() => { setUseLime(false); setUseAttention(true); explain() }}>
              {loading ? 'Explaining…' : 'Explain with Attention'}
            </button>
            {explanation && (
              <div className="p-4 border rounded bg-white space-y-3 mt-3">
                <div className="text-sm text-gray-500">Attention Weights</div>
                <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(explanation, null, 2)}</pre>
              </div>
            )}
          </Tab.Panel>
          <Tab.Panel>
            <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={loading || !text.trim()} onClick={generateCounterfactual}>
              {loading ? 'Generating…' : 'Generate Counterfactual'}
            </button>
            {counterfactual && (
              <div className="p-4 border rounded bg-white space-y-3 mt-3">
                <div className="text-sm text-gray-500">Counterfactual Output</div>
                <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(counterfactual, null, 2)}</pre>
              </div>
            )}
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  )
}



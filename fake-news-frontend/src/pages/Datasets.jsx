import { useEffect, useState } from 'react'
import { listDatasets, getDatasetSample, runFullPipeline, runCrossDomainEvaluation, runDatasetEvaluation } from '../services/api'
import BaseChart from '../components/charts/BaseChart'
import { useNotifyStore } from '../stores/notifyStore'

export default function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [selected, setSelected] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const push = useNotifyStore(s => s.push)
  const [sample, setSample] = useState([])
  const [sampleInfo, setSampleInfo] = useState({ totalRecords: null, sampleSize: 0 })
  const [pipelineResult, setPipelineResult] = useState(null)
  const [evaluationResult, setEvaluationResult] = useState(null)
  const [crossDomainResult, setCrossDomainResult] = useState(null)

  useEffect(() => {
    const load = async () => {
      try {
        const data = await listDatasets()
        // Support various shapes: array of objects with name, array of strings, or object map
        let items = []
        if (Array.isArray(data)) {
          items = data
        } else if (data && typeof data === 'object') {
          // Convert object map to array of { name }
          const keys = Object.keys(data)
          items = keys.map(k => ({ name: k, ...(typeof data[k] === 'object' ? data[k] : {}) }))
        }
        setDatasets(items)
      } catch (e) {
        push({ title: 'Failed to load datasets', message: e.message || 'request_failed', variant: 'error' })
        setDatasets([])
      }
    }
    load()
  }, [push])

  useEffect(() => {
    const loadSample = async () => {
      if (!selected) { setSample([]); return }
      try {
        const data = await getDatasetSample(selected, 5)
        const rows = Array.isArray(data?.sample) ? data.sample : []
        setSample(rows)
        setSampleInfo({
          totalRecords: typeof data?.total_records === 'number' ? data.total_records : (rows?.length || 0),
          sampleSize: typeof data?.sample_size === 'number' ? data.sample_size : (rows?.length || 0)
        })
      } catch {
        setSample([])
        setSampleInfo({ totalRecords: null, sampleSize: 0 })
      }
    }
    loadSample()
  }, [selected])

  const onRunFullPipeline = async () => {
    if (!selected) return
    setLoading(true)
    setMessage('')
    try {
      const res = await runFullPipeline(selected)
      setPipelineResult(res)
      const msg = `Full pipeline completed for ${selected}`
      setMessage(msg)
      push({ title: 'Pipeline', message: msg })
    } catch (e) {
      const msg = e.message || 'pipeline_failed'
      setMessage(msg)
      push({ title: 'Pipeline failed', message: msg, variant: 'error' })
    } finally {
      setLoading(false)
    }
  }

  const onRunCrossDomain = async () => {
    setLoading(true)
    setMessage('')
    try {
      const res = await runCrossDomainEvaluation()
      setCrossDomainResult(res)
      const msg = 'Cross-domain evaluation completed'
      setMessage(msg)
      push({ title: 'Cross-domain', message: msg })
    } catch (e) {
      const msg = e.message || 'cross_domain_failed'
      setMessage(msg)
      push({ title: 'Cross-domain failed', message: msg, variant: 'error' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Datasets</h1>
      <div className="flex items-center gap-3">
        <select className="border rounded p-2" value={selected} onChange={e => setSelected(e.target.value)}>
          <option value="">Select dataset…</option>
          {datasets.map((d, idx) => {
            const name = typeof d === 'string' ? d : (d.name || d.dataset || `dataset_${idx}`)
            return (
              <option key={name} value={name}>{name}</option>
            )
          })}
        </select>
        <button className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50" disabled={!selected || loading} onClick={onRunFullPipeline}>
          {loading ? 'Running…' : 'Run Full Pipeline'}
        </button>
        <button className="px-3 py-2 rounded bg-slate-700 text-white disabled:opacity-50" disabled={loading} onClick={onRunCrossDomain}>
          {loading ? 'Running…' : 'Run Cross-Domain Evaluation'}
        </button>
        <button className="px-3 py-2 rounded bg-emerald-600 text-white disabled:opacity-50" disabled={!selected || loading} onClick={async () => {
          try {
            setLoading(true)
            const res = await runDatasetEvaluation(selected, { save_report: false, limit: 500 })
            setEvaluationResult(res)
            push({ title: 'Evaluation', message: `Evaluation complete for ${selected}` })
          } catch (e) {
            push({ title: 'Evaluation failed', message: e.message || 'evaluate_failed', variant: 'error' })
          } finally {
            setLoading(false)
          }
        }}>
          {loading ? 'Running…' : 'Run Evaluation'}
        </button>
      </div>
      {message && <div className="text-sm text-gray-600">{message}</div>}

      {selected && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <div className="text-lg font-semibold text-gray-800">
              Sample Data ({selected}) — {sampleInfo.sampleSize} shown
            </div>
            <div className="text-sm text-gray-500">
              Total: {sampleInfo.totalRecords ?? 'N/A'} records
            </div>
          </div>

          {/* Glossary and Route Overview */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-4">
              <div className="text-sm font-semibold text-gray-800 mb-2">Glossary</div>
              <ul className="text-xs text-gray-600 space-y-1 leading-5">
                <li><span className="font-medium text-gray-800">Label</span>: Truthfulness class (true, mostly-true, half-true, barely-true, false, pants-on-fire).</li>
                <li><span className="font-medium text-gray-800">Subject</span>: Topic tags extracted from the dataset (comma-separated).</li>
                <li><span className="font-medium text-gray-800">Context</span>: Source or medium where the statement appeared.</li>
                <li><span className="font-medium text-gray-800">Split</span>: Dataset partition (train/valid/test).</li>
              </ul>
            </div>
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-4">
              <div className="text-sm font-semibold text-gray-800 mb-2">API Routes (This Page)</div>
              <ul className="text-xs text-gray-600 space-y-1 leading-5">
                <li><code className="bg-gray-100 px-1.5 py-0.5 rounded">GET /datasets</code> — list available datasets.</li>
                <li><code className="bg-gray-100 px-1.5 py-0.5 rounded">GET /datasets/{'{name}'}/sample?size=K</code> — fetch a preview sample.</li>
                <li><code className="bg-gray-100 px-1.5 py-0.5 rounded">POST /datasets/full-pipeline/{'{name}'}</code> — run preprocessing pipeline.</li>
                <li><code className="bg-gray-100 px-1.5 py-0.5 rounded">POST /datasets/evaluate/{'{name}'}</code> — evaluate model on dataset.</li>
                <li><code className="bg-gray-100 px-1.5 py-0.5 rounded">POST /datasets/evaluate/cross-domain</code> — cross-dataset evaluation.</li>
              </ul>
            </div>
          </div>
          
          {sample.length > 0 ? (
            <div className="bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gradient-to-r from-blue-50 to-indigo-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-b border-gray-200">
                        Statement
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-b border-gray-200">
                        Label
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-b border-gray-200">
                        Speaker
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-b border-gray-200">
                        Subject
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-b border-gray-200">
                        Context
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-b border-gray-200">
                        Details
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-100">
                    {sample.map((item, idx) => (
                      <tr key={idx} className="hover:bg-gray-50 transition-colors duration-150">
                        <td className="px-4 py-4 text-sm text-gray-900 max-w-xs">
                          <div className="break-words leading-relaxed">
                            {item.statement}
                          </div>
                        </td>
                        <td className="px-4 py-4 text-sm">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            // Handle both string labels (LIAR) and label_text (ISOT)
                            (item.label_text === 'real' || item.label === 'true') ? 'bg-green-100 text-green-800' :
                            item.label === 'mostly-true' ? 'bg-blue-100 text-blue-800' :
                            item.label === 'half-true' ? 'bg-yellow-100 text-yellow-800' :
                            item.label === 'barely-true' ? 'bg-orange-100 text-orange-800' :
                            (item.label_text === 'fake' || item.label === 'false') ? 'bg-red-100 text-red-800' :
                            item.label === 'pants-on-fire' ? 'bg-red-200 text-red-900' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {/* Use label_text if available (ISOT), otherwise use label with replace (LIAR) */}
                            {item.label_text || (typeof item.label === 'string' ? item.label.replace('-', ' ') : String(item.label))}
                          </span>
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-900">
                          <div className="font-medium">{item.speaker || 'N/A'}</div>
                          {item.job_title && (
                            <div className="text-xs text-gray-500">{item.job_title}</div>
                          )}
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-900">
                          <div className="flex flex-wrap gap-1">
                            {item.subject?.split(',').map((subj, subjIdx) => (
                              <span key={subjIdx} className="inline-flex items-center px-2 py-1 rounded-md text-xs bg-indigo-50 text-indigo-700 border border-indigo-200">
                                {subj.trim()}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-600">
                          <div className="italic text-xs">
                            {item.context || 'N/A'}
                          </div>
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-600">
                          <div className="space-y-1 text-xs">
                            {item.state_info && (
                              <div className="flex items-center gap-1">
                                <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                                {item.state_info}
                              </div>
                            )}
                            {item.party_affiliation && item.party_affiliation !== 'none' && (
                              <div className="flex items-center gap-1">
                                <span className={`w-2 h-2 rounded-full ${
                                  item.party_affiliation === 'republican' ? 'bg-red-400' :
                                  item.party_affiliation === 'democrat' ? 'bg-blue-400' :
                                  'bg-gray-400'
                                }`}></span>
                                {item.party_affiliation}
                              </div>
                            )}
                            {item.split && (
                              <div className="text-gray-400">
                                Split: {item.split}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <div className="text-lg font-medium">No sample data available</div>
              <div className="text-sm">Select a different dataset or try again later</div>
            </div>
          )}

          {/* Pipeline Summary */}
          {pipelineResult && (
            <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-white rounded-lg shadow border border-gray-200 p-4">
                <div className="text-sm font-semibold mb-2">Pipeline Summary</div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="text-gray-500">Original Records</div>
                    <div className="text-lg font-semibold">{pipelineResult.original_records}</div>
                  </div>
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="text-gray-500">Processed Records</div>
                    <div className="text-lg font-semibold">{pipelineResult.processed_records}</div>
                  </div>
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="text-gray-500">Final Records</div>
                    <div className="text-lg font-semibold">{pipelineResult.final_records}</div>
                  </div>
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="text-gray-500">Features Added</div>
                    <div className="text-lg font-semibold">{pipelineResult.features_added}</div>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg shadow border border-gray-200 p-4">
                <div className="text-sm font-semibold mb-2">Record Counts</div>
                <div className="h-48">
                  <BaseChart
                    type="bar"
                    data={{
                      labels: ['Original', 'Processed', 'Final'],
                      datasets: [{
                        label: 'Records',
                        data: [pipelineResult.original_records, pipelineResult.processed_records, pipelineResult.final_records],
                        backgroundColor: ['#93c5fd', '#a7f3d0', '#fcd34d']
                      }]
                    }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Evaluation Summary */}
          {evaluationResult && (
            <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="bg-white rounded-lg shadow border border-gray-200 p-4">
                <div className="text-sm font-semibold mb-2">Evaluation Metrics</div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="p-3 bg-gray-50 rounded"><div className="text-gray-500">Accuracy</div><div className="text-lg font-semibold">{evaluationResult.accuracy?.toFixed(4)}</div></div>
                  <div className="p-3 bg-gray-50 rounded"><div className="text-gray-500">Precision</div><div className="text-lg font-semibold">{evaluationResult.precision?.toFixed(4)}</div></div>
                  <div className="p-3 bg-gray-50 rounded"><div className="text-gray-500">Recall</div><div className="text-lg font-semibold">{evaluationResult.recall?.toFixed(4)}</div></div>
                  <div className="p-3 bg-gray-50 rounded"><div className="text-gray-500">F1</div><div className="text-lg font-semibold">{evaluationResult.f1?.toFixed(4)}</div></div>
                </div>
              </div>
              <div className="bg-white rounded-lg shadow border border-gray-200 p-4 lg:col-span-2">
                <div className="text-sm font-semibold mb-2">Model vs Baseline (Accuracy)</div>
                <div className="h-48">
                  <BaseChart
                    type="bar"
                    data={{
                      labels: ['Model', 'Baseline'],
                      datasets: [{
                        label: 'Accuracy',
                        data: [evaluationResult.accuracy || 0, evaluationResult.extra_metrics?.baseline_accuracy || 0],
                        backgroundColor: ['#60a5fa', '#f87171']
                      }]
                    }}
                    options={{ scales: { y: { min: 0, max: 1 } } }}
                  />
                </div>
              </div>
              </div>
          )}

          {/* Cross-Domain Summary */}
          {crossDomainResult?.evaluations?.length > 0 && (
            <div className="mt-8 bg-white rounded-lg shadow border border-gray-200 p-4">
              <div className="text-sm font-semibold mb-3">Cross-Domain Evaluation</div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Dataset</th>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Size</th>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Accuracy</th>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Precision</th>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Recall</th>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">F1</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-100">
                    {crossDomainResult.evaluations.map((ev, i) => (
                      <tr key={i} className="hover:bg-gray-50">
                        <td className="px-4 py-2 text-sm">{ev.dataset}</td>
                        <td className="px-4 py-2 text-sm">{ev.size}</td>
                        <td className="px-4 py-2 text-sm">{ev.accuracy ?? '—'}</td>
                        <td className="px-4 py-2 text-sm">{ev.precision ?? '—'}</td>
                        <td className="px-4 py-2 text-sm">{ev.recall ?? '—'}</td>
                        <td className="px-4 py-2 text-sm">{ev.f1 ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="h-56 mt-4">
                <BaseChart
                  type="bar"
                  data={{
                    labels: crossDomainResult.evaluations.map(e => e.dataset),
                    datasets: [{ label: 'Accuracy', data: crossDomainResult.evaluations.map(e => e.accuracy || 0), backgroundColor: '#34d399' }]
                  }}
                  options={{ scales: { y: { min: 0, max: 1 } } }}
                />
          </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}



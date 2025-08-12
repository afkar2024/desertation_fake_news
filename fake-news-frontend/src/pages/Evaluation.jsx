import { useEffect, useMemo, useState } from 'react'
import { listDatasets, runDatasetEvaluation, listReports } from '../services/api'
import BaseChart from '../components/charts/BaseChart'
import { Tab } from '@headlessui/react'
import clsx from 'clsx'
import MetricsCards from './Evaluation/components/MetricsCards'
import CurvesPanel from './Evaluation/components/CurvesPanel'
import ConfusionMatrix from './Evaluation/components/ConfusionMatrix'
import BaselinesPanel from './Evaluation/components/BaselinesPanel'
import AbstentionPanel from './Evaluation/components/AbstentionPanel'
import FairnessTable from './Evaluation/components/FairnessTable'
import TemporalStability from './Evaluation/components/TemporalStability'
import Button from '../components/ui/Button'
import { useNotifyStore } from '../stores/notifyStore'

export default function Evaluation() {
  const [datasets, setDatasets] = useState([])
  const [selected, setSelected] = useState('')
  const [loading, setLoading] = useState(false)
  const [evalData, setEvalData] = useState(null)
  const [error, setError] = useState('')
  const push = useNotifyStore(s => s.push)
  const [reports, setReports] = useState([])
  const [selectedReport, setSelectedReport] = useState(null)
  const [saveReport, setSaveReport] = useState(false)

  useEffect(() => {
    listDatasets().then(setDatasets).catch(() => {})
    listReports().then((r) => setReports(r?.items || [])).catch(() => {})
  }, [])

  const runEval = async () => {
    if (!selected) return
    setLoading(true)
    setError('')
    try {
      const data = await runDatasetEvaluation(selected, { compare_traditional: true, save_report: saveReport, limit: 1000, abstention_curve: true, explainability_quality: true, mc_dropout_samples: 30 })
      setEvalData(data)
      if (saveReport) {
        // refresh report list
        listReports().then((r) => setReports(r?.items || [])).catch(() => {})
      }
      push({ title: 'Evaluation complete', message: selected })
    } catch (e) {
      setError(e.message || 'evaluate_failed')
      push({ title: 'Evaluation failed', message: e.message || 'evaluate_failed', variant: 'error' })
    } finally {
      setLoading(false)
    }
  }

  const metrics = evalData ? { accuracy: evalData.accuracy, f1: evalData.f1, precision: evalData.precision, recall: evalData.recall } : {}
  const extras = evalData?.extra_metrics || {}
  const confusion = extras?.confusion_matrix || null
  const calibration = extras && (extras.brier_score != null || extras.ece != null || extras.reliability_bins) ? { bins: extras.reliability_bins || [] } : null
  const roc = extras?.roc_curve || null
  const pr = extras?.pr_curve || null
  const baselineAcc = extras?.baseline_accuracy
  const traditionalAcc = extras?.traditional_accuracy
  const abstention = extras?.coverage_accuracy || []
  const fairness = extras?.fairness_groups || null
  const temporal = extras?.temporal_periods || null
  const miMean = extras?.mutual_information_mean
  const expQuality = extras?.explainability_quality_topk_delta_mean

  const rocChart = useMemo(() => {
    if (!roc) return null
    const points = Array.isArray(roc.points) ? roc.points : []
    const auc = roc.auc || null
    return {
      data: {
        datasets: [
          {
            label: 'ROC',
            data: points.map(p => ({ x: p.fpr, y: p.tpr })),
            borderColor: 'rgb(59,130,246)',
            backgroundColor: 'rgba(59,130,246,0.1)',
            fill: true,
            tension: 0.2,
            pointRadius: 0
          },
          {
            label: 'Random',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: 'rgba(156,163,175,0.8)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        plugins: { legend: { display: false }, title: { display: !!auc, text: auc ? `AUC: ${auc.toFixed?.(3) ?? auc}` : '' } },
        scales: { x: { title: { display: true, text: 'FPR' }, min: 0, max: 1 }, y: { title: { display: true, text: 'TPR' }, min: 0, max: 1 } }
      }
    }
  }, [roc])

  const prChart = useMemo(() => {
    if (!pr) return null
    const points = Array.isArray(pr.points) ? pr.points : []
    const auc = pr.auc || null
    return {
      data: {
        datasets: [
          {
            label: 'PR',
            data: points.map(p => ({ x: p.recall ?? p.r, y: p.precision ?? p.p })),
            borderColor: 'rgb(16,185,129)',
            backgroundColor: 'rgba(16,185,129,0.1)',
            fill: true,
            tension: 0.2,
            pointRadius: 0
          }
        ]
      },
      options: {
        plugins: { legend: { display: false }, title: { display: !!auc, text: auc ? `PR-AUC: ${auc.toFixed?.(3) ?? auc}` : '' } },
        scales: { x: { title: { display: true, text: 'Recall' }, min: 0, max: 1 }, y: { title: { display: true, text: 'Precision' }, min: 0, max: 1 } }
      }
    }
  }, [pr])

  const calibChart = useMemo(() => {
    if (!calibration) return null
    const bins = calibration.bins || []
    const labels = bins.map((b, i) => (b?.low != null && b?.high != null) ? `${b.low.toFixed?.(2)}-${b.high.toFixed?.(2)}` : String(i + 1))
    const accs = bins.map((b) => b?.accuracy ?? 0)
    return {
      data: {
        labels,
        datasets: [
          { label: 'Accuracy', data: accs, borderColor: 'rgb(34,197,94)', backgroundColor: 'rgba(34,197,94,0.1)', tension: 0.2, fill: true, pointRadius: 0 }
        ]
      },
      options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1 } } }
    }
  }, [calibration])

  const reliabilityChart = useMemo(() => {
    if (!calibration) return null
    const bins = calibration.bins || []
    const xs = bins.map((b, i) => b?.avg_conf ?? ((b?.low + b?.high) / 2) ?? (i + 1) / bins.length)
    const ys = bins.map((b) => b?.accuracy ?? 0)
    return {
      data: {
        datasets: [
          { label: 'Perfect', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: 'rgba(107,114,128,0.9)', borderDash: [4,4], fill: false, pointRadius: 0 },
          { label: 'Reliability', data: xs.map((x, i) => ({ x, y: ys[i] })), borderColor: 'rgb(59,130,246)', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.2, pointRadius: 2 }
        ]
      },
      options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Confidence' }, min: 0, max: 1 }, y: { title: { display: true, text: 'Observed Accuracy' }, min: 0, max: 1 } } }
    }
  }, [calibration])

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Evaluation</h1>
      {error && <div className="p-3 text-sm rounded bg-red-50 text-red-700">{error}</div>}

      <div className="flex gap-3 items-center">
        <select className="border rounded p-2" value={selected} onChange={e => setSelected(e.target.value)}>
          <option value="">Select dataset…</option>
          {(datasets || []).map((d) => (
            <option key={d.name || d} value={d.name || d}>{d.name || d}</option>
          ))}
        </select>
        <label className="flex items-center gap-2 text-sm text-gray-600">
          <input type="checkbox" className="accent-blue-600" checked={saveReport} onChange={e => setSaveReport(e.target.checked)} />
          Save this evaluation
        </label>
        <Button disabled={!selected || loading} onClick={runEval}>{loading ? 'Running…' : 'Run Evaluation'}</Button>
      </div>

      {reports?.length > 0 && (
        <div className="mt-4">
          <div className="text-sm text-gray-500 mb-2">Recent JSON Evaluation Reports</div>
          <div className="flex flex-wrap gap-2">
            {reports.slice(0, 6).map((r) => (
              <Button key={r.id} variant="ghost" onClick={async () => {
                try {
                  const item = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'}/reports/${r.id}`).then(res => res.json())
                  setSelectedReport(item)
                } catch {}
              }}>{r.report_type} • {r.dataset} • {new Date(r.created_at).toLocaleString()}</Button>
            ))}
          </div>
          {selectedReport && (
            <div className="mt-3 border rounded p-3 bg-white">
              <div className="text-sm text-gray-500 mb-2">Report JSON</div>
              <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(selectedReport.payload, null, 2)}</pre>
            </div>
          )}
        </div>
      )}

      <Tab.Group>
        <Tab.List className="flex space-x-2 rounded bg-gray-100 p-1">
          {['Metrics','Curves','Baselines','Fairness','Temporal','Uncertainty & XAI','Reports'].map((label) => (
            <Tab key={label} className={({ selected }) => clsx('px-3 py-1.5 rounded text-sm font-medium', selected ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900')}>{label}</Tab>
          ))}
        </Tab.List>
        <Tab.Panels className="mt-4 space-y-4">
          <Tab.Panel>
            {metrics && (metrics.accuracy || metrics.f1 || metrics.precision || metrics.recall) && (
              <MetricsCards metrics={metrics} />
            )}
            {confusion && Array.isArray(confusion) && confusion.length === 2 && (
              <div className="mt-4">
                <div className="text-sm text-gray-500 mb-2">Confusion Matrix</div>
                <div className="grid grid-cols-2 w-64 text-center border rounded overflow-hidden">
                  {confusion.flat().map((v, i) => (
                    <div key={i} className={`p-3 ${i === 0 || i === 3 ? 'bg-green-50' : 'bg-red-50'} border`}>{v}</div>
                  ))}
                </div>
              </div>
            )}
          </Tab.Panel>
          <Tab.Panel>
            <CurvesPanel roc={extras?.roc_curve || null} pr={extras?.pr_curve || null} calibration={{ bins: extras?.reliability_bins || [] }} />
          </Tab.Panel>
          <Tab.Panel>
            <BaselinesPanel modelAcc={metrics.accuracy} baselineAcc={baselineAcc} traditionalAcc={traditionalAcc} />
          </Tab.Panel>
          <Tab.Panel>
            <FairnessTable groups={fairness} />
          </Tab.Panel>
          <Tab.Panel>
            <TemporalStability periods={temporal} />
          </Tab.Panel>
          <Tab.Panel>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {miMean != null && (
                <div className="bg-white border rounded p-4">
                  <div className="text-sm font-semibold mb-2">MC Dropout (Mutual Information)</div>
                  <div className="text-xs text-gray-500 mb-2">Higher values indicate higher epistemic uncertainty.</div>
                  <div className="text-2xl font-semibold">{miMean?.toFixed?.(4)}</div>
                </div>
              )}
              {expQuality != null && (
                <div className="bg-white border rounded p-4">
                  <div className="text-sm font-semibold mb-2">Explanation Quality (Deletion)</div>
                  <div className="text-xs text-gray-500 mb-2">Average change in probability when removing top SHAP tokens; larger indicates stronger explanations.</div>
                  <div className="text-2xl font-semibold">{expQuality?.toFixed?.(4)}</div>
                </div>
              )}
            </div>
            <div className="mt-4">
              <AbstentionPanel curve={abstention} />
            </div>
          </Tab.Panel>
          <Tab.Panel>
            {reports?.length > 0 ? (
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2">
                  {reports.slice(0, 10).map((r) => (
                    <Button key={r.id} variant="ghost" onClick={async () => {
                      try {
                        const item = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'}/reports/${r.id}`).then(res => res.json())
                        setSelectedReport(item)
                      } catch {}
                    }}>{r.report_type} • {r.dataset} • {new Date(r.created_at).toLocaleString()}</Button>
                  ))}
                </div>
                {selectedReport && (
                  <div className="border rounded p-4 bg-white">
                    <div className="text-sm text-gray-500 mb-2">Selected Report</div>
                    <div className="text-xs text-gray-500 mb-1">Dataset: {selectedReport.dataset} | Type: {selectedReport.report_type} | Created: {new Date(selectedReport.created_at).toLocaleString()}</div>
                    <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(selectedReport.payload, null, 2)}</pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-sm text-gray-500">No saved reports yet. Enable "Save this evaluation" and run an evaluation.</div>
            )}
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  )
}

// Subcomponents moved into ./components



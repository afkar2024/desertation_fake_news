import { useEffect, useMemo, useState } from 'react'
import { listDatasets, runDatasetEvaluation, listReports, getReport } from '../services/api'
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
import SignificancePanel from './Evaluation/components/SignificancePanel'
import CalibrationSummary from './Evaluation/components/CalibrationSummary'
import Glossary from './Evaluation/components/Glossary'
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
  const mcnemarB = extras?.mcnemar_b
  const mcnemarC = extras?.mcnemar_c
  const mcnemarChi2 = extras?.mcnemar_chi2
  const mcnemarP = extras?.mcnemar_p
  const rocAuc = extras?.roc_curve?.auc ?? extras?.roc_auc
  const prAuc = extras?.pr_curve?.auc ?? extras?.pr_auc
  const abstention = extras?.coverage_accuracy || []
  const fairness = extras?.fairness_groups || null
  const temporal = extras?.temporal_periods || null
  const miMean = extras?.mutual_information_mean
  const miStd = extras?.mutual_information_std
  const mcSamples = extras?.mc_dropout_samples
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
          <div className="text-sm text-gray-500 mb-2">Recent Reports</div>
          <div className="flex flex-wrap gap-2">
            {reports.slice(0, 6).map((r) => (
              <Button key={r.id} variant="ghost" onClick={async () => {
                try {
                  const item = await getReport(r.id)
                  setSelectedReport(item)
                  if (item?.payload) {
                    setEvalData(item.payload)
                  }
                } catch {}
              }}>{r.report_type} • {r.dataset} • {new Date(r.created_at).toLocaleString()}</Button>
            ))}
          </div>
          {selectedReport && (
            <div className="mt-3 border rounded p-3 bg-white">
              <div className="text-sm text-gray-500 mb-2">Selected Report</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                <div className="border rounded p-3 bg-gray-50">
                  <div className="text-gray-500">Dataset</div>
                  <div className="font-semibold">{selectedReport.dataset}</div>
                </div>
                <div className="border rounded p-3 bg-gray-50">
                  <div className="text-gray-500">Type</div>
                  <div className="font-semibold capitalize">{selectedReport.report_type}</div>
                </div>
                <div className="border rounded p-3 bg-gray-50">
                  <div className="text-gray-500">Created</div>
                  <div className="font-semibold">{new Date(selectedReport.created_at).toLocaleString()}</div>
                </div>
                <div className="border rounded p-3 bg-gray-50">
                  <div className="text-gray-500">Trace ID</div>
                  <div className="font-mono truncate" title={selectedReport?.payload?.trace_id || ''}>{selectedReport?.payload?.trace_id || '-'}</div>
                </div>
              </div>
              {selectedReport?.report_type === 'cross_domain' && Array.isArray(selectedReport?.payload?.evaluations) && (
                <div className="mt-3 overflow-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-600">
                        <th className="p-2">Dataset</th>
                        <th className="p-2">Samples</th>
                        <th className="p-2">Accuracy</th>
                        <th className="p-2">F1</th>
                        <th className="p-2">Baseline Acc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedReport.payload.evaluations.map((ev, i) => (
                        <tr key={i} className="border-t">
                          <td className="p-2">{ev.dataset}</td>
                          <td className="p-2">{ev.size}</td>
                          <td className="p-2">{ev.accuracy ?? '-'}</td>
                          <td className="p-2">{ev.f1 ?? '-'}</td>
                          <td className="p-2">{ev.baseline_accuracy ?? '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
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
            <div className="mt-4">
              <CalibrationSummary brier={extras?.brier_score} ece={extras?.ece} rocAuc={rocAuc} prAuc={prAuc} />
            </div>
            {(mcnemarB != null || mcnemarC != null || mcnemarChi2 != null || mcnemarP != null) && (
              <div className="mt-4">
                <SignificancePanel b={mcnemarB} c={mcnemarC} chi2={mcnemarChi2} p={mcnemarP} />
              </div>
            )}
            {(baselineAcc != null || traditionalAcc != null || extras?.baseline_precision != null || extras?.traditional_precision != null) && (
              <div className="mt-4 bg-white border rounded p-4">
                <div className="text-sm font-semibold mb-2">Baseline & Traditional Details</div>
                <div className="overflow-auto">
                  <table className="min-w-[520px] text-sm">
                    <thead>
                      <tr className="text-left text-gray-600">
                        <th className="p-2">Model</th>
                        <th className="p-2">Accuracy</th>
                        <th className="p-2">Precision</th>
                        <th className="p-2">Recall</th>
                        <th className="p-2">F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-t">
                        <td className="p-2">This Model</td>
                        <td className="p-2">{metrics.accuracy ?? '-'}</td>
                        <td className="p-2">{metrics.precision ?? '-'}</td>
                        <td className="p-2">{metrics.recall ?? '-'}</td>
                        <td className="p-2">{metrics.f1 ?? '-'}</td>
                      </tr>
                      {(baselineAcc != null || extras?.baseline_precision != null) && (
                        <tr className="border-t">
                          <td className="p-2">Heuristic Baseline</td>
                          <td className="p-2">{baselineAcc ?? '-'}</td>
                          <td className="p-2">{extras?.baseline_precision ?? '-'}</td>
                          <td className="p-2">{extras?.baseline_recall ?? '-'}</td>
                          <td className="p-2">{extras?.baseline_f1 ?? '-'}</td>
                        </tr>
                      )}
                      {(traditionalAcc != null || extras?.traditional_precision != null) && (
                        <tr className="border-t">
                          <td className="p-2">Traditional (TF-IDF + LR)</td>
                          <td className="p-2">{traditionalAcc ?? '-'}</td>
                          <td className="p-2">{extras?.traditional_precision ?? '-'}</td>
                          <td className="p-2">{extras?.traditional_recall ?? '-'}</td>
                          <td className="p-2">{extras?.traditional_f1 ?? '-'}</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
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
              {miStd != null && (
                <div className="bg-white border rounded p-4">
                  <div className="text-sm font-semibold mb-2">MC Dropout (MI Std)</div>
                  <div className="text-xs text-gray-500 mb-2">Dispersion of mutual information over samples.</div>
                  <div className="text-2xl font-semibold">{miStd?.toFixed?.(4)}</div>
                </div>
              )}
              {mcSamples != null && (
                <div className="bg-white border rounded p-4">
                  <div className="text-sm font-semibold mb-2">MC Samples</div>
                  <div className="text-xs text-gray-500 mb-2">Number of stochastic forward passes.</div>
                  <div className="text-2xl font-semibold">{mcSamples}</div>
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
            {Array.isArray(abstention) && abstention.length > 0 && (
              <div className="mt-3 bg-white border rounded p-4 overflow-auto">
                <div className="text-sm font-semibold mb-2">Coverage-Accuracy Table</div>
                <table className="min-w-[360px] text-sm">
                  <thead>
                    <tr className="text-left text-gray-600">
                      <th className="p-2">Coverage</th>
                      <th className="p-2">Accuracy</th>
                      <th className="p-2">N</th>
                    </tr>
                  </thead>
                  <tbody>
                    {abstention.map((row, i) => (
                      <tr key={i} className="border-t">
                        <td className="p-2">{row.coverage}</td>
                        <td className="p-2">{row.accuracy}</td>
                        <td className="p-2">{row.n}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
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
                    <div className="text-xs text-gray-500 mb-3">Dataset: {selectedReport.dataset} | Type: {selectedReport.report_type} | Created: {new Date(selectedReport.created_at).toLocaleString()}</div>
                    {selectedReport?.report_type === 'evaluation' && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">Total Evaluated</div>
                          <div className="text-lg font-semibold">{selectedReport?.payload?.total_evaluated ?? '-'}</div>
                        </div>
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">Accuracy</div>
                          <div className="text-lg font-semibold">{selectedReport?.payload?.accuracy ?? '-'}</div>
                        </div>
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">F1</div>
                          <div className="text-lg font-semibold">{selectedReport?.payload?.f1 ?? '-'}</div>
                        </div>
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">Trace ID</div>
                          <div className="text-xs font-mono truncate" title={selectedReport?.payload?.trace_id || ''}>{selectedReport?.payload?.trace_id || '-'}</div>
                        </div>
                      </div>
                    )}
                    {selectedReport?.report_type === 'cross_domain' && Array.isArray(selectedReport?.payload?.evaluations) && (
                      <div className="overflow-auto">
                        <table className="min-w-full text-sm">
                          <thead>
                            <tr className="text-left text-gray-600">
                              <th className="p-2">Dataset</th>
                              <th className="p-2">Samples</th>
                              <th className="p-2">Accuracy</th>
                              <th className="p-2">F1</th>
                              <th className="p-2">Baseline Acc</th>
                            </tr>
                          </thead>
                          <tbody>
                            {selectedReport.payload.evaluations.map((ev, i) => (
                              <tr key={i} className="border-t">
                                <td className="p-2">{ev.dataset}</td>
                                <td className="p-2">{ev.size}</td>
                                <td className="p-2">{ev.accuracy ?? '-'}</td>
                                <td className="p-2">{ev.f1 ?? '-'}</td>
                                <td className="p-2">{ev.baseline_accuracy ?? '-'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {selectedReport?.report_type === 'full_pipeline' && selectedReport?.payload && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">Original Records</div>
                          <div className="text-lg font-semibold">{selectedReport?.payload?.original_records ?? '-'}</div>
                        </div>
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">Processed Records</div>
                          <div className="text-lg font-semibold">{selectedReport?.payload?.processed_records ?? '-'}</div>
                        </div>
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">Final Records</div>
                          <div className="text-lg font-semibold">{selectedReport?.payload?.final_records ?? '-'}</div>
                        </div>
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">Features Added</div>
                          <div className="text-lg font-semibold">{selectedReport?.payload?.features_added ?? '-'}</div>
                        </div>
                        <div className="border rounded p-3 bg-gray-50">
                          <div className="text-gray-500 text-xs">Balance Strategy</div>
                          <div className="text-lg font-semibold">{selectedReport?.payload?.balance_strategy ?? '-'}</div>
                        </div>
                        <div className="sm:col-span-2 lg:col-span-3 border rounded p-3">
                          <div className="text-gray-500 text-xs mb-1">Files</div>
                          <div className="text-xs font-mono break-all">{JSON.stringify(selectedReport?.payload?.files_generated || {}, null, 2)}</div>
                        </div>
                      </div>
                    )}
                    {selectedReport?.report_type === 'evaluation' && selectedReport?.payload && (
                      <div className="mt-3">
                        {selectedReport?.payload?.extra_metrics?.shap_samples_json && (
                          <div className="text-xs text-gray-500">SHAP JSON: {selectedReport.payload.extra_metrics.shap_samples_json}</div>
                        )}
                        {selectedReport?.payload?.extra_metrics?.shap_samples_markdown && (
                          <div className="text-xs text-gray-500">SHAP Markdown: {selectedReport.payload.extra_metrics.shap_samples_markdown}</div>
                        )}
                      </div>
                    )}
                  </div>
                )}
                {selectedReport && (
                  <div className="mt-3">
                    <Glossary items={[
                      { term: 'Accuracy', desc: 'Fraction of correct predictions over all predictions.' },
                      { term: 'Precision', desc: 'Among predicted positives, how many are truly positive.' },
                      { term: 'Recall', desc: 'Among true positives, how many are correctly identified.' },
                      { term: 'F1', desc: 'Harmonic mean of Precision and Recall.' },
                      { term: 'ROC-AUC', desc: 'Area under the ROC curve (higher is better).' },
                      { term: 'PR-AUC', desc: 'Area under the Precision-Recall curve (higher is better).' },
                      { term: 'Brier Score', desc: 'Mean squared error of predicted probabilities (lower is better).' },
                      { term: 'ECE', desc: 'Expected Calibration Error (lower is better).' },
                      { term: 'Coverage-Accuracy', desc: 'Accuracy measured at different coverage thresholds based on model confidence.' },
                      { term: "McNemar's Test", desc: 'Statistical test comparing two classifiers on paired data; p < 0.05 indicates significance.' },
                      { term: 'Mutual Information', desc: 'Epistemic uncertainty estimate via MC Dropout.' }
                    ]} />
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



import { useEffect, useState } from 'react'
import { getHealth, getModelInfo, getAnalyticsSummary } from '../services/api'
import Button from '../components/ui/Button'
import { Card, CardContent } from '../components/ui/Card'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import BaseChart from '../components/charts/BaseChart'
import { useAppStore } from '../stores/appStore'

export default function Dashboard() {
  const [health, setHealth] = useState('unknown')
  const [model, setModel] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const setDashboard = useAppStore(s => s.setDashboard)

  useEffect(() => {
    Promise.all([
      getHealth(),
      getModelInfo(),
      getAnalyticsSummary()
    ]).then(([h, m, a]) => {
      setHealth(h?.status || 'ok')
      setModel(m)
      setAnalytics(a)
      setDashboard({ health: h, model: m, analytics: a })
      setLoading(false)
    }).catch((e) => {
      setError(e.message || 'request_failed')
      setLoading(false)
    })
  }, [setDashboard])

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Dashboard</h1>

      {error && (
        <div className="p-3 text-sm rounded bg-red-50 text-red-700">Error loading: {error}</div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card>
          <CardContent>
            <div className="text-xs text-gray-500">API Health</div>
            <div className="text-lg font-semibold flex items-center gap-2">{loading ? <LoadingSpinner /> : health}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <div className="text-xs text-gray-500">Model</div>
            <div className="text-lg font-semibold">{loading ? '…' : (model?.model_name || '—')}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <div className="text-xs text-gray-500">Predictions</div>
            <div className="text-lg font-semibold">{loading ? '…' : (analytics?.prediction_count ?? '—')}</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardContent>
          <div className="text-xs text-gray-500">Drift Signal</div>
          <div className="text-lg font-semibold">{loading ? '…' : (analytics?.drift_ks_p_value != null ? analytics.drift_ks_p_value.toFixed(3) : '—')}</div>
          {!loading && analytics?.drift_ks_p_value != null && (
            <div className="text-xs mt-1 text-gray-600">{analytics.drift_ks_p_value < 0.05 ? 'Potential drift detected (p < 0.05)' : 'No significant drift detected'}</div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card>
          <CardContent>
            <div className="text-xs text-gray-500">Avg P(fake)</div>
            <div className="text-lg font-semibold">{loading ? '…' : (analytics?.avg_prob_fake != null ? analytics.avg_prob_fake.toFixed(3) : '—')}</div>
            {!loading && Array.isArray(analytics?.avg_prob_fake_history) && (
              <div className="h-24 mt-2">
                <BaseChart
                  type="line"
                  data={{
                    labels: analytics.avg_prob_fake_history.map((_, i) => i + 1),
                    datasets: [{
                      label: 'Avg P(fake) (recent)',
                      data: analytics.avg_prob_fake_history,
                      borderColor: 'rgba(239,68,68,0.9)',
                      backgroundColor: 'rgba(239,68,68,0.1)',
                      tension: 0.3,
                      fill: true,
                      pointRadius: 0
                    }]
                  }}
                  options={{
                    plugins: { legend: { display: false } },
                    scales: { x: { display: false }, y: { min: 0, max: 1, ticks: { display: false } } }
                  }}
                />
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <div className="text-xs text-gray-500">Avg P(real)</div>
            <div className="text-lg font-semibold">{loading ? '…' : (analytics?.avg_prob_real != null ? analytics.avg_prob_real.toFixed(3) : '—')}</div>
            {!loading && Array.isArray(analytics?.avg_prob_real_history) && (
              <div className="h-24 mt-2">
                <BaseChart
                  type="line"
                  data={{
                    labels: analytics.avg_prob_real_history.map((_, i) => i + 1),
                    datasets: [{
                      label: 'Avg P(real) (recent)',
                      data: analytics.avg_prob_real_history,
                      borderColor: 'rgba(34,197,94,0.9)',
                      backgroundColor: 'rgba(34,197,94,0.1)',
                      tension: 0.3,
                      fill: true,
                      pointRadius: 0
                    }]
                  }}
                  options={{
                    plugins: { legend: { display: false } },
                    scales: { x: { display: false }, y: { min: 0, max: 1, ticks: { display: false } } }
                  }}
                />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="flex gap-3">
        <Button variant="primary" onClick={() => window.location.assign('/datasets')}>Run Evaluation</Button>
        <Button variant="secondary" onClick={() => window.location.assign('/datasets')}>Run Cross-Domain</Button>
      </div>
    </div>
  )
}



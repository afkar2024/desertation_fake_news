import { useEffect, useState } from 'react'
import { getHealth, getModelInfo, getAnalyticsSummary, listReports, getReport } from '../services/api'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import EmptyState from '../components/ui/EmptyState'
import SystemStatus from '../components/dashboard/SystemStatus'
import EvaluationSummary from '../components/dashboard/EvaluationSummary'
import RecentReports from '../components/dashboard/RecentReports'
import { useAppStore } from '../stores/appStore'

export default function Dashboard() {
  const [health, setHealth] = useState('unknown')
  const [model, setModel] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [reports, setReports] = useState([])
  const [latestEvaluation, setLatestEvaluation] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const setDashboard = useAppStore(s => s.setDashboard)

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setLoading(true)
        
        // Load system data
        const [h, m, a] = await Promise.all([
          getHealth(),
          getModelInfo(),
          getAnalyticsSummary()
        ])
        
        setHealth(h)
        setModel(m)
        setAnalytics(a)
        setDashboard({ health: h, model: m, analytics: a })
        
        // Load reports
        try {
          const reportsData = await listReports()
          setReports(reportsData.items || [])
          
          // Find latest evaluation report
          const evaluationReports = reportsData.items?.filter(r => r.report_type === 'evaluation') || []
          
          if (evaluationReports.length > 0) {
            const latestReport = evaluationReports[0]
            
            const reportDetails = await getReport(latestReport.id)
            
            // Transform the data for the EvaluationSummary component
            const transformedEvaluationData = {
              metrics: {
                accuracy: reportDetails.payload?.accuracy || 0,
                precision: reportDetails.payload?.precision || 0,
                recall: reportDetails.payload?.recall || 0,
                f1: reportDetails.payload?.f1 || 0,
                roc_auc: reportDetails.payload?.extra_metrics?.roc_auc || 0,
                brier_score: reportDetails.payload?.extra_metrics?.brier_score || 0,
                ece: reportDetails.payload?.extra_metrics?.ece || 0
              },
              dataset: reportDetails.dataset,
              created_at: reportDetails.created_at,
              size: reportDetails.payload?.total_evaluated || 0,
              extra_metrics: reportDetails.payload?.extra_metrics || {}
            }
            
            setLatestEvaluation(transformedEvaluationData)
          }
        } catch (reportError) {
          console.warn('Failed to load reports:', reportError)
          setReports([])
        }
        
        setLoading(false)
      } catch (e) {
        console.error('Dashboard loading error:', e)
        setError(e.message || 'Failed to load dashboard data')
        setLoading(false)
      }
    }

    loadDashboardData()
  }, [setDashboard])

  const handleViewReport = async (reportId) => {
    try {
      const report = await getReport(reportId)
      if (report) {
        // Navigate to evaluation page with report data
        window.location.href = `/evaluation?report=${reportId}`
      }
    } catch (error) {
      console.error('Failed to load report:', error)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <LoadingSpinner size="large" />
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <div className="p-4 text-sm rounded-lg bg-red-50 text-red-700 border border-red-200">
          Error loading dashboard: {error}
        </div>
      </div>
    )
  }

  const hasEvaluationData = latestEvaluation && reports.length > 0
  
  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-2">
            {hasEvaluationData 
              ? 'Overview of your fake news detection system and evaluation results'
              : 'Get started with evaluating your fake news detection model'
            }
          </p>
        </div>
        
        {hasEvaluationData && (
          <div className="flex space-x-3">
            <button
              onClick={() => window.location.href = '/datasets'}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Run New Evaluation
            </button>
            <button
              onClick={() => window.location.href = '/evaluation'}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              View All Reports
            </button>
          </div>
        )}
      </div>

      {/* System Status - Always show */}
      <SystemStatus 
        health={health} 
        model={model} 
        analytics={analytics} 
      />

      {!hasEvaluationData ? (
        <EmptyState
          title="No Evaluation Data Available"
          description="You haven't run any evaluations yet. Get started by running the full pipeline on a dataset to see comprehensive metrics and analysis."
          actionText="Run Evaluation Pipeline"
          actionHref="/datasets"
          icon={
            <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
            </svg>
          }
        />
      ) : (
        <>
          <EvaluationSummary evaluationData={latestEvaluation} />
          <RecentReports reports={reports} onViewReport={handleViewReport} />
        </>
      )}
    </div>
  )
}



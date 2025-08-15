import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import BaseChart from '../charts/BaseChart'
import ProgressRing from '../ui/ProgressRing'
import MetricCard from '../ui/MetricCard'
import { Card, CardContent } from '../ui/Card'

const EvaluationSummary = ({ evaluationData, className = '' }) => {
  const [isVisible, setIsVisible] = useState(false)
  
  useEffect(() => {
    setIsVisible(true)
  }, [])

  if (!evaluationData) return null

  const { metrics, dataset, created_at, size } = evaluationData
  const { accuracy, precision, recall, f1 } = metrics || {}

  // Calculate color based on performance
  const getPerformanceColor = (value) => {
    if (value >= 0.8) return 'green'
    if (value >= 0.6) return 'blue'
    if (value >= 0.4) return 'orange'
    return 'red'
  }

  // Prepare reliability data for chart
  const reliabilityData = evaluationData.payload?.reliability_bins
  const coverageData = evaluationData.payload?.coverage_accuracy

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
      transition={{ duration: 0.5 }}
      className={`space-y-6 ${className}`}
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Evaluation Summary</h2>
            <p className="text-blue-100">
              Dataset: <span className="font-semibold">{dataset}</span> • 
              Samples: <span className="font-semibold">{size}</span> • 
              Date: <span className="font-semibold">{new Date(created_at).toLocaleDateString()}</span>
            </p>
          </div>
          <div className="text-right">
            <div className="text-4xl font-bold">{Math.round((accuracy || 0) * 100)}%</div>
            <div className="text-blue-100">Overall Accuracy</div>
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title="Accuracy"
          value={`${Math.round((accuracy || 0) * 100)}%`}
          subtitle="Correct predictions"
          color={getPerformanceColor(accuracy)}
          icon={
            <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          }
        />
        <MetricCard
          title="Precision"
          value={`${Math.round((precision || 0) * 100)}%`}
          subtitle="True positives / predicted positives"
          color={getPerformanceColor(precision)}
          icon={
            <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
            </svg>
          }
        />
        <MetricCard
          title="Recall"
          value={`${Math.round((recall || 0) * 100)}%`}
          subtitle="True positives / actual positives"
          color={getPerformanceColor(recall)}
          icon={
            <svg className="w-5 h-5 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
            </svg>
          }
        />
        <MetricCard
          title="F1 Score"
          value={`${Math.round((f1 || 0) * 100)}%`}
          subtitle="Harmonic mean of precision & recall"
          color={getPerformanceColor(f1)}
          icon={
            <svg className="w-5 h-5 text-orange-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
            </svg>
          }
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reliability Chart */}
        {reliabilityData && (
          <Card>
            <CardContent>
              <h3 className="text-lg font-semibold mb-4">Reliability Analysis</h3>
              <div className="h-64">
                <BaseChart
                  type="bar"
                  data={{
                    labels: reliabilityData.map(bin => `Bin ${bin.bin}`),
                    datasets: [{
                      label: 'Accuracy',
                      data: reliabilityData.map(bin => bin.accuracy || 0),
                      backgroundColor: 'rgba(59, 130, 246, 0.8)',
                      borderColor: 'rgba(59, 130, 246, 1)',
                      borderWidth: 1
                    }]
                  }}
                  options={{
                    plugins: { legend: { display: false } },
                    scales: {
                      y: { 
                        beginAtZero: true, 
                        max: 1,
                        ticks: { callback: value => `${(value * 100).toFixed(0)}%` }
                      }
                    }
                  }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Confidence vs Accuracy across different probability bins
              </p>
            </CardContent>
          </Card>
        )}

        {/* Coverage vs Accuracy Chart */}
        {coverageData && (
          <Card>
            <CardContent>
              <h3 className="text-lg font-semibold mb-4">Coverage vs Accuracy</h3>
              <div className="h-64">
                <BaseChart
                  type="line"
                  data={{
                    labels: coverageData.map(row => `${(row.coverage * 100).toFixed(0)}%`),
                    datasets: [{
                      label: 'Accuracy',
                      data: coverageData.map(row => row.accuracy),
                      borderColor: 'rgba(16, 185, 129, 1)',
                      backgroundColor: 'rgba(16, 185, 129, 0.1)',
                      tension: 0.4,
                      fill: true
                    }]
                  }}
                  options={{
                    plugins: { legend: { display: false } },
                    scales: {
                      y: { 
                        beginAtZero: true, 
                        max: 1,
                        ticks: { callback: value => `${(value * 100).toFixed(0)}%` }
                      }
                    }
                  }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Model performance as coverage increases with confidence threshold
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Additional Metrics */}
      {evaluationData.payload && (
        <Card>
          <CardContent>
            <h3 className="text-lg font-semibold mb-4">Additional Metrics</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {evaluationData.payload.roc_auc && (
                <div className="text-center">
                  <ProgressRing
                    progress={Math.round((evaluationData.payload.roc_auc || 0) * 100)}
                    color="#8B5CF6"
                    label="ROC-AUC"
                    subtitle="Discrimination"
                  />
                </div>
              )}
              {evaluationData.payload.brier_score && (
                <div className="text-center">
                  <ProgressRing
                    progress={Math.round((1 - (evaluationData.payload.brier_score || 0)) * 100)}
                    color="#EF4444"
                    label="Brier Score"
                    subtitle="Calibration"
                  />
                </div>
              )}
              {evaluationData.payload.ece && (
                <div className="text-center">
                  <ProgressRing
                    progress={Math.round((1 - (evaluationData.payload.ece || 0)) * 100)}
                    color="#F59E0B"
                    label="ECE"
                    subtitle="Calibration Error"
                  />
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </motion.div>
  )
}

export default EvaluationSummary

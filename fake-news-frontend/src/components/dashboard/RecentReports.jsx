import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import { Card, CardContent } from '../ui/Card'
import Button from '../ui/Button'

const RecentReports = ({ reports = [], onViewReport, className = '' }) => {
  const [isVisible, setIsVisible] = useState(false)
  
  useEffect(() => {
    setIsVisible(true)
  }, [])

  if (!reports || reports.length === 0) return null

  const getReportIcon = (reportType) => {
    switch (reportType) {
      case 'evaluation':
        return (
          <svg className="w-6 h-6 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
          </svg>
        )
      case 'cross_domain':
        return (
          <svg className="w-6 h-6 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
          </svg>
        )
      default:
        return (
          <svg className="w-6 h-6 text-gray-600" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
          </svg>
        )
    }
  }

  const getReportTypeLabel = (reportType) => {
    switch (reportType) {
      case 'evaluation':
        return 'Model Evaluation'
      case 'cross_domain':
        return 'Cross-Domain Analysis'
      case 'full_pipeline':
        return 'Full Pipeline'
      default:
        return reportType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
      transition={{ duration: 0.5 }}
      className={`space-y-4 ${className}`}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Recent Reports</h3>
        <Button 
          variant="secondary" 
          onClick={() => window.location.href = '/datasets'}
          size="sm"
        >
          Run New Evaluation
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {reports.slice(0, 6).map((report, index) => (
          <motion.div
            key={report.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={isVisible ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card className="hover:shadow-lg transition-all duration-200 cursor-pointer group">
              <CardContent className="p-4">
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0">
                    <div className="p-2 rounded-lg bg-gray-100 group-hover:bg-blue-50 transition-colors">
                      {getReportIcon(report.report_type)}
                    </div>
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <h4 className="text-sm font-medium text-gray-900 truncate">
                      {getReportTypeLabel(report.report_type)}
                    </h4>
                    <p className="text-xs text-gray-500 mt-1">
                      Dataset: <span className="font-medium">{report.dataset}</span>
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {formatDate(report.created_at)}
                    </p>
                  </div>
                </div>
                
                <div className="mt-3 flex justify-end">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => onViewReport(report.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    View Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {reports.length > 6 && (
        <div className="text-center pt-4">
          <Button 
            variant="secondary" 
            onClick={() => window.location.href = '/evaluation'}
          >
            View All Reports
          </Button>
        </div>
      )}
    </motion.div>
  )
}

export default RecentReports

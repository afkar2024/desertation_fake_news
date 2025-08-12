import { Chart as ChartJS } from 'chart.js/auto'
import { Chart } from 'react-chartjs-2'
import { forwardRef } from 'react'

const BaseChart = forwardRef(({ type = 'bar', data, options = {}, className = '', ...props }, ref) => {
  const defaultOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: { usePointStyle: true, padding: 20, font: { size: 12 } }
      },
      tooltip: {
        backgroundColor: 'rgba(0,0,0,0.8)',
        titleFont: { size: 14 },
        bodyFont: { size: 13 },
        cornerRadius: 8,
        displayColors: true
      }
    },
    ...options,
  }

  return (
    <div className={`relative ${className}`}>
      <Chart ref={ref} type={type} data={data} options={defaultOptions} {...props} />
    </div>
  )
})

export default BaseChart



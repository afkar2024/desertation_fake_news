import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

const StatsGrid = ({ 
  items, 
  columns = 3, 
  className = '',
  animationDelay = 0.1 
}) => {
  const [isVisible, setIsVisible] = useState(false)
  
  useEffect(() => {
    setIsVisible(true)
  }, [])

  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
    5: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5',
    6: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6'
  }

  return (
    <div className={`grid ${gridCols[columns]} gap-6 ${className}`}>
      {items.map((item, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ 
            duration: 0.5, 
            delay: animationDelay * index,
            ease: "easeOut"
          }}
          className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-6 border border-gray-100"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              {item.icon && (
                <div className={`p-2 rounded-lg ${item.iconBg || 'bg-blue-100'}`}>
                  {item.icon}
                </div>
              )}
              <div>
                <h3 className="text-sm font-medium text-gray-600 uppercase tracking-wide">
                  {item.title}
                </h3>
                {item.subtitle && (
                  <p className="text-xs text-gray-500 mt-1">{item.subtitle}</p>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-baseline space-x-2">
            <div className="text-2xl font-bold text-gray-900">
              {item.value}
            </div>
            
            {item.trend && (
              <div className={`text-sm font-medium ${
                item.trend > 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {item.trend > 0 ? '+' : ''}{item.trend}%
              </div>
            )}
          </div>
          
          {item.description && (
            <p className="text-xs text-gray-500 mt-2">{item.description}</p>
          )}
          
          {item.chart && (
            <div className="mt-4 h-16">
              {item.chart}
            </div>
          )}
        </motion.div>
      ))}
    </div>
  )
}

export default StatsGrid

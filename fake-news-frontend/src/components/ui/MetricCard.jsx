import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

const MetricCard = ({ 
  title, 
  value, 
  subtitle, 
  trend, 
  icon, 
  color = 'blue',
  size = 'default',
  className = '' 
}) => {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    setIsVisible(true)
  }, [])

  const colorVariants = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    red: 'from-red-500 to-red-600',
    purple: 'from-purple-500 to-purple-600',
    orange: 'from-orange-500 to-orange-600',
    indigo: 'from-indigo-500 to-indigo-600'
  }

  const sizeVariants = {
    small: 'p-4',
    default: 'p-6',
    large: 'p-8'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className={`relative overflow-hidden rounded-xl bg-white shadow-lg hover:shadow-xl transition-all duration-300 ${sizeVariants[size]} ${className}`}
    >
      {/* Background gradient */}
      <div className={`absolute inset-0 bg-gradient-to-br ${colorVariants[color]} opacity-5`} />
      
      {/* Content */}
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            {icon && (
              <div className={`p-2 rounded-lg bg-gradient-to-br ${colorVariants[color]} bg-opacity-10`}>
                {icon}
              </div>
            )}
            <div>
              <h3 className="text-sm font-medium text-gray-600 uppercase tracking-wide">
                {title}
              </h3>
              {subtitle && (
                <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
              )}
            </div>
          </div>
        </div>
        
        <div className="flex items-baseline space-x-2">
          <motion.div
            initial={{ scale: 0.8 }}
            animate={isVisible ? { scale: 1 } : { scale: 0.8 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-3xl font-bold text-gray-900"
          >
            {value}
          </motion.div>
          
          {trend && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={isVisible ? { opacity: 1, x: 0 } : { opacity: 0, x: -10 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className={`text-sm font-medium ${
                trend > 0 ? 'text-green-600' : 'text-red-600'
              }`}
            >
              {trend > 0 ? '+' : ''}{trend}%
            </motion.div>
          )}
        </div>
      </div>
      
      {/* Decorative elements */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-transparent to-current opacity-5 rounded-full -translate-y-16 translate-x-16" />
    </motion.div>
  )
}

export default MetricCard

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

const ProgressRing = ({ 
  progress, 
  size = 120, 
  strokeWidth = 8, 
  color = '#3B82F6',
  label,
  subtitle,
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(false)
  
  useEffect(() => {
    setIsVisible(true)
  }, [])

  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (progress / 100) * circumference

  return (
    <div className={`flex flex-col items-center ${className}`}>
      <div className="relative" style={{ width: size, height: size }}>
        {/* Background circle */}
        <svg
          width={size}
          height={size}
          className="transform -rotate-90"
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#E5E7EB"
            strokeWidth={strokeWidth}
            fill="transparent"
          />
          {/* Progress circle */}
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="transparent"
            strokeDasharray={strokeDasharray}
            initial={{ strokeDashoffset: circumference }}
            animate={isVisible ? { strokeDashoffset } : { strokeDashoffset: circumference }}
            transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
            strokeLinecap="round"
          />
        </svg>
        
        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={isVisible ? { scale: 1, opacity: 1 } : { scale: 0, opacity: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="text-center"
          >
            <div className="text-2xl font-bold text-gray-900">
              {Math.round(progress)}%
            </div>
            {subtitle && (
              <div className="text-xs text-gray-500 mt-1">{subtitle}</div>
            )}
          </motion.div>
        </div>
      </div>
      
      {label && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 10 }}
          transition={{ duration: 0.5, delay: 0.7 }}
          className="text-sm font-medium text-gray-700 mt-3 text-center"
        >
          {label}
        </motion.div>
      )}
    </div>
  )
}

export default ProgressRing

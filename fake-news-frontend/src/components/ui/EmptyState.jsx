import { motion } from 'framer-motion'
import Button from './Button'

const EmptyState = ({ 
  title = "No Evaluation Data Available",
  description = "You haven't run any evaluations yet. Get started by running the full pipeline on a dataset.",
  actionText = "Run Evaluation Pipeline",
  actionHref = "/datasets",
  icon,
  className = ''
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className={`text-center py-16 px-6 ${className}`}
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="mx-auto max-w-md"
      >
        {icon && (
          <div className="mx-auto flex items-center justify-center h-24 w-24 rounded-full bg-gray-100 mb-6">
            <div className="h-12 w-12 text-gray-400">
              {icon}
            </div>
          </div>
        )}
        
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          {title}
        </h3>
        
        <p className="text-sm text-gray-500 mb-6">
          {description}
        </p>
        
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Button 
            variant="primary" 
            onClick={() => window.location.href = actionHref}
            className="w-full sm:w-auto"
          >
            {actionText}
          </Button>
        </motion.div>
      </motion.div>
    </motion.div>
  )
}

export default EmptyState

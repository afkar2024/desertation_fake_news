export default function LoadingSpinner({ size = 'default', className = '' }) {
  const sizes = {
    sm: 'h-4 w-4',
    default: 'h-5 w-5',
    lg: 'h-8 w-8',
    xl: 'h-12 w-12'
  }
  
  return (
    <div 
      className={`inline-block ${sizes[size]} animate-spin rounded-full border-2 border-current border-t-transparent text-blue-600 align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite] ${className}`} 
      role="status" 
      aria-label="loading" 
    />
  )
}



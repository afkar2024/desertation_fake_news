export default function Button({ 
  children, 
  className = '', 
  variant = 'primary', 
  size = 'default',
  disabled, 
  ...props 
}) {
  const base = 'inline-flex items-center justify-center rounded font-medium transition focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
    secondary: 'bg-gray-600 text-white hover:bg-gray-700 focus:ring-2 focus:ring-gray-500 focus:ring-offset-2',
    ghost: 'bg-transparent hover:bg-gray-100 text-gray-700 hover:text-gray-900'
  }
  
  const sizes = {
    sm: 'px-2.5 py-1.5 text-xs',
    default: 'px-3 py-2 text-sm',
    lg: 'px-4 py-2.5 text-base',
    xl: 'px-6 py-3 text-lg'
  }
  
  return (
    <button 
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`} 
      disabled={disabled} 
      {...props}
    >
      {children}
    </button>
  )
}



export default function Button({ children, className = '', variant = 'primary', disabled, ...props }) {
  const base = 'inline-flex items-center justify-center rounded px-3 py-2 text-sm font-medium transition focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed'
  const variants = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-slate-700 text-white hover:bg-slate-800',
    ghost: 'bg-transparent hover:bg-gray-100'
  }
  return (
    <button className={`${base} ${variants[variant]} ${className}`} disabled={disabled} {...props}>
      {children}
    </button>
  )
}



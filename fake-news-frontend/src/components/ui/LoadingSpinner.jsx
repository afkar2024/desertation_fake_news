export default function LoadingSpinner({ className = '' }) {
  return (
    <div className={`inline-block h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent text-blue-600 align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite] ${className}`} role="status" aria-label="loading" />
  )
}



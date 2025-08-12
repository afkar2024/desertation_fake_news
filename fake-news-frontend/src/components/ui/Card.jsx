export function Card({ children, className = '' }) {
  return <div className={`border rounded bg-white ${className}`}>{children}</div>
}

export function CardHeader({ children, className = '' }) {
  return <div className={`px-4 py-3 border-b ${className}`}>{children}</div>
}

export function CardContent({ children, className = '' }) {
  return <div className={`px-4 py-3 ${className}`}>{children}</div>
}



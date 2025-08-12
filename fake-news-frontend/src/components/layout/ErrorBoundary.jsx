import { Component } from 'react'

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    if (this.props.onError) this.props.onError(error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="max-w-2xl mx-auto p-4 mt-6 border rounded bg-red-50 text-red-800">
          <div className="font-semibold mb-1">Something went wrong.</div>
          <pre className="text-xs whitespace-pre-wrap">{String(this.state.error)}</pre>
        </div>
      )
    }
    return this.props.children
  }
}



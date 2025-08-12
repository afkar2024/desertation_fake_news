import { Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import './App.css'
import { ErrorBoundary } from './components/layout/ErrorBoundary'
import Toaster from './components/ui/Toaster'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Analysis = lazy(() => import('./pages/Analysis'))
const Explainability = lazy(() => import('./pages/Explainability'))
const Datasets = lazy(() => import('./pages/Datasets'))
const Evaluation = lazy(() => import('./pages/Evaluation'))

function App() {
  return (
    <Router>
      <div className="min-h-screen flex">
        <aside className="w-64 shrink-0 hidden md:block border-r bg-white">
          <div className="p-4 font-semibold">Fake News Detection</div>
          <nav className="flex flex-col gap-1 p-2 text-sm">
            <NavLink className={({isActive}) => `px-3 py-2 rounded ${isActive ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-50'}`} to="/">Dashboard</NavLink>
            <NavLink className={({isActive}) => `px-3 py-2 rounded ${isActive ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-50'}`} to="/analysis">Analysis</NavLink>
            <NavLink className={({isActive}) => `px-3 py-2 rounded ${isActive ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-50'}`} to="/explainability">Explainability</NavLink>
            <NavLink className={({isActive}) => `px-3 py-2 rounded ${isActive ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-50'}`} to="/datasets">Datasets</NavLink>
            <NavLink className={({isActive}) => `px-3 py-2 rounded ${isActive ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-50'}`} to="/evaluation">Evaluation</NavLink>
          </nav>
        </aside>
        <main className="flex-1">
          <header className="border-b bg-white">
            <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
              <div className="font-semibold">System Console</div>
              <div className="flex items-center gap-3 text-xs text-gray-500">
                <button onClick={() => document.documentElement.classList.toggle('dark')} className="px-2 py-1 border rounded">Theme</button>
                <span>v0.1.0</span>
              </div>
            </div>
          </header>
          <div className="max-w-6xl mx-auto px-4 py-6">
            <ErrorBoundary>
              <Suspense fallback={<div>Loading…</div>}>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/analysis" element={<Analysis />} />
                  <Route path="/explainability" element={<Explainability />} />
                  <Route path="/datasets" element={<Datasets />} />
                  <Route path="/evaluation" element={<Evaluation />} />
                </Routes>
              </Suspense>
            </ErrorBoundary>
          </div>
        </main>
        <Toaster />
      </div>
    </Router>
  )
}

export default App

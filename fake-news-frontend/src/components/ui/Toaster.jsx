import { useNotifyStore } from '../../stores/notifyStore'

export default function Toaster() {
  const toasts = useNotifyStore(s => s.toasts)
  const remove = useNotifyStore(s => s.remove)
  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2">
      {toasts.map(t => (
        <div key={t.id} className={`min-w-64 max-w-sm p-3 rounded shadow text-sm border bg-white ${t.variant === 'error' ? 'border-red-200' : 'border-gray-200'}`}>
          <div className="flex items-start gap-2">
            <div className="font-medium">{t.title}</div>
            <button onClick={() => remove(t.id)} className="ml-auto text-gray-400 hover:text-gray-600">✕</button>
          </div>
          {t.message && <div className="mt-1 text-gray-600">{t.message}</div>}
        </div>
      ))}
    </div>
  )
}



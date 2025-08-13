export default function Glossary({ items = [] }) {
  if (!Array.isArray(items) || items.length === 0) return null
  return (
    <div className="bg-white border rounded p-4">
      <div className="text-sm font-semibold mb-2">Glossary</div>
      <ul className="text-sm text-gray-700 list-disc ml-5 space-y-1">
        {items.map((it, idx) => (
          <li key={idx}><span className="font-semibold">{it.term}:</span> {it.desc}</li>
        ))}
      </ul>
    </div>
  )
}



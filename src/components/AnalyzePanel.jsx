import { useState } from 'react'

function AnalyzePanel() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim()) return

    setLoading(true)
    setResult(null)

    try {
      const apiUrl = import.meta.env.PROD ? '/analyze' : 'http://127.0.0.1:8000/analyze'
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      if (!res.ok) throw new Error(`Error ${res.status}`)
      const data = await res.json()
      
      setTimeout(() => {
        setResult(data)
        setLoading(false)
      }, 300)
    } catch (err) {
      console.error('Error:', err)
      setResult({
        sentiment: 'Error analizando la frase',
        score: 0,
        emoji: '⚠️',
      })
      setLoading(false)
    }
  }

  const handleInputChange = (e) => {
    setText(e.target.value)
    if (result) setResult(null)
  }

  const getResultClass = (sentiment) => {
    if (sentiment === 'positivo') return 'pos'
    if (sentiment === 'negativo') return 'neg'
    return 'neu'
  }

  return (
    <section id="analizar" className="panel">
      <h2>Prueba la demo</h2>
      <p className="subtitle">
        Escribe una frase y detecta si es <strong>positiva</strong>, <strong>negativa</strong> o <strong>neutral</strong>.
      </p>

      <form onSubmit={handleSubmit} className={loading ? 'loading' : ''}>
        <label htmlFor="text">Frase a analizar</label>
        <textarea
          id="text"
          name="text"
          rows="4"
          placeholder="Ej: excelente atención, muy rápido y eficiente..."
          value={text}
          onChange={handleInputChange}
          required
        />
        <button type="submit">
          <span className="btn-text">{loading ? 'Analizando...' : 'Analizar Sentimiento'}</span>
        </button>
      </form>

      {result && (
        <div className={`result ${getResultClass(result.sentiment)}`}>
          <div className="badge">{result.emoji}</div>
          <div className="result-content">
            <div className="sentiment">{result.sentiment}</div>
            <div className="score">Score: {result.score}</div>
          </div>
        </div>
      )}
    </section>
  )
}

export default AnalyzePanel


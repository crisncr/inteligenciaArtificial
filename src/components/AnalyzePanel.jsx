import { useState, useEffect } from 'react'
import { analysesAPI } from '../utils/api'

function AnalyzePanel({ onAnalyze, reanalyzeText, user, freeAnalysesLeft, onLimitReached }) {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)

  // Escuchar cambios en reanalyzeText
  useEffect(() => {
    if (reanalyzeText) {
      setText(reanalyzeText)
      setResult(null)
    }
  }, [reanalyzeText])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim()) return

    // Verificar límite de análisis gratuitos
    if ((!user || user.plan === 'free') && freeAnalysesLeft <= 0) {
      if (onLimitReached) {
        onLimitReached()
      }
      return
    }

    setLoading(true)
    setResult(null)

    try {
      let data
      
      if (user) {
        // Usuario autenticado: usar API real que guarda en BD
        // analyze_sentiment() ahora SOLO usa red neuronal LSTM
        const analysis = await analysesAPI.create(text)
        data = {
          sentiment: analysis.sentiment,
          score: analysis.score,
          emoji: analysis.emoji
        }
      } else {
        // Usuario no autenticado: usar endpoint público
        // El endpoint /analyze usa red neuronal LSTM
        const apiUrl = import.meta.env.PROD ? '/analyze' : 'http://127.0.0.1:8000/analyze'
        const res = await fetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })

        if (!res.ok) throw new Error(`Error ${res.status}`)
        data = await res.json()
      }
      
      setTimeout(() => {
        setResult({ ...data, timestamp: new Date().toISOString() })
        setLoading(false)
        if (onAnalyze) {
          onAnalyze({ ...data, text, timestamp: new Date().toISOString() })
        }
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

  const handleClear = () => {
    setText('')
    setResult(null)
    setCopied(false)
  }

  const handleCopyResult = async () => {
    if (!result) return
    const resultText = `Sentimiento: ${result.sentiment}\nScore: ${result.score}\nTexto: ${text}`
    try {
      await navigator.clipboard.writeText(resultText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Error al copiar:', err)
    }
  }

  const getResultClass = (sentiment) => {
    if (sentiment === 'positivo') return 'pos'
    if (sentiment === 'negativo') return 'neg'
    return 'neu'
  }

  return (
    <section id="analizar" className="panel panel-enhanced">
      <div className="panel-header-enhanced">
        <div className="panel-title-group">
          <h2>Prueba la demo</h2>
          <p className="subtitle">
            Escribe una frase y detecta si es <strong>positiva</strong>, <strong>negativa</strong> o <strong>neutral</strong> usando <strong>Red Neuronal LSTM</strong>.
          </p>
        </div>
      </div>

      <div className="panel-content-wrapper">

      {(!user || user.plan === 'free') && freeAnalysesLeft <= 0 && (
        <div className="analysis-limit-warning">
          {!user ? (
            <p>⚠️ Has alcanzado el límite de <strong>3 análisis gratuitos</strong>. <strong>Inicia sesión</strong> o <strong>regístrate</strong> para seleccionar un plan.</p>
          ) : (
            <p>⚠️ Has alcanzado el límite de <strong>3 análisis gratuitos</strong>. Selecciona un plan para continuar.</p>
          )}
        </div>
      )}

      {(!user || user.plan === 'free') && freeAnalysesLeft > 0 && (
        <div className="analysis-counter-enhanced">
          <div className="counter-icon">📊</div>
          <div className="counter-content">
            <span className="counter-label">Análisis gratuitos restantes</span>
            <span className="counter-value"><strong>{freeAnalysesLeft}</strong> de 3</span>
          </div>
        </div>
      )}

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
        <div className="form-actions">
          <button 
            type="submit" 
            disabled={(!user || user.plan === 'free') && freeAnalysesLeft <= 0}
            title="Análisis con Red Neuronal LSTM"
          >
            <span className="btn-text">{loading ? 'Analizando con IA...' : 'Analizar con Red Neuronal'}</span>
          </button>
          {text && (
            <button type="button" className="btn--ghost btn--small" onClick={handleClear}>
              Limpiar
            </button>
          )}
        </div>
      </form>

      {result && (
        <div className={`result ${getResultClass(result.sentiment)}`}>
          <div className="badge">{result.emoji}</div>
          <div className="result-content">
            <div className="sentiment">{result.sentiment}</div>
            <div className="score">Score: {result.score}</div>
            <div className="result-actions">
              <button 
                type="button" 
                className="btn--small btn--ghost" 
                onClick={handleCopyResult}
                title="Copiar resultado"
              >
                {copied ? '✓ Copiado' : '📋 Copiar'}
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </section>
  )
}

export default AnalyzePanel


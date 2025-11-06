import { useState, useEffect } from 'react'
import { analysesAPI } from '../utils/api'

function ManualAnalysis({ user, onAnalyze, freeAnalysesLeft, onLimitReached }) {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)
  const [analysesToday, setAnalysesToday] = useState(0)

  // Cargar contador de análisis del día
  useEffect(() => {
    const loadAnalysesCount = async () => {
      if (user && user.plan === 'free') {
        try {
          const analyses = await analysesAPI.getAll()
          const today = new Date()
          today.setHours(0, 0, 0, 0)
          const todayAnalyses = analyses.filter(a => {
            const analysisDate = new Date(a.created_at)
            analysisDate.setHours(0, 0, 0, 0)
            return analysisDate.getTime() === today.getTime() && a.source === 'manual'
          })
          setAnalysesToday(todayAnalyses.length)
        } catch (err) {
          console.error('Error al cargar análisis del día:', err)
        }
      }
    }
    loadAnalysesCount()
  }, [user, result])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim()) return

    // Verificar límite de análisis gratuitos (10 por día para plan gratuito)
    if (user && user.plan === 'free') {
      if (analysesToday >= 10) {
        if (onLimitReached) {
          onLimitReached()
        }
        return
      }
    }

    setLoading(true)
    setResult(null)

    try {
      // Usuario autenticado: usar API real que guarda en BD con source='manual'
      const analysis = await analysesAPI.create(text)
      const data = {
        sentiment: analysis.sentiment,
        score: analysis.score,
        emoji: analysis.emoji,
        source: 'manual'
      }
      
      setTimeout(() => {
        setResult({ ...data, timestamp: new Date().toISOString() })
        setLoading(false)
        setAnalysesToday(prev => prev + 1)
        
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

  const analysesLeft = user && user.plan === 'free' ? Math.max(0, 10 - analysesToday) : Infinity

  return (
    <section className="dashboard-section">
      <div className="section-header">
        <h1>Análisis de Sentimientos</h1>
        <p className="section-subtitle">
          Analiza el sentimiento de cualquier texto directamente desde aquí
        </p>
      </div>

      {user && user.plan === 'free' && analysesToday >= 10 && (
        <div className="analysis-limit-warning">
          <p>⚠️ Has alcanzado el límite de <strong>10 análisis por día</strong> del plan gratuito. Selecciona un plan Pro o Enterprise para análisis ilimitados.</p>
        </div>
      )}

      {user && user.plan === 'free' && analysesToday < 10 && (
        <div className="analysis-counter-enhanced">
          <div className="counter-icon">📊</div>
          <div className="counter-content">
            <span className="counter-label">Análisis restantes hoy</span>
            <span className="counter-value"><strong>{analysesLeft}</strong> de 10</span>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className={loading ? 'loading' : ''}>
        <label htmlFor="text">Texto a analizar</label>
        <textarea
          id="text"
          name="text"
          rows="6"
          placeholder="Ej: excelente atención, muy rápido y eficiente. El servicio fue increíble..."
          value={text}
          onChange={handleInputChange}
          required
        />
        <div className="form-actions">
          <button 
            type="submit" 
            disabled={user && user.plan === 'free' && analysesToday >= 10}
            className="btn"
          >
            <span className="btn-text">{loading ? 'Analizando...' : 'Analizar Sentimiento'}</span>
          </button>
          {text && (
            <button type="button" className="btn btn--ghost" onClick={handleClear}>
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
                className="btn btn--ghost btn--small" 
                onClick={handleCopyResult}
                title="Copiar resultado"
              >
                {copied ? '✓ Copiado' : '📋 Copiar'}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

export default ManualAnalysis


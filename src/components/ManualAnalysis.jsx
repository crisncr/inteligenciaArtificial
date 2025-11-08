import { useState, useEffect } from 'react'
import { analysesAPI, datasetsAPI } from '../utils/api'

function ManualAnalysis({ user, onAnalyze, freeAnalysesLeft, onLimitReached }) {
  const [activeTab, setActiveTab] = useState('individual') // 'individual', 'dataset', 'search'
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)
  const [analysesToday, setAnalysesToday] = useState(0)
  
  // Estados para pestaña de Dataset
  const [datasetFile, setDatasetFile] = useState(null)
  const [datasetResults, setDatasetResults] = useState(null)
  const [datasetLoading, setDatasetLoading] = useState(false)
  const [datasetTexts, setDatasetTexts] = useState([])
  
  // Estados para pestaña de Búsqueda
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])

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

  // Funciones para Dataset
  const handleDatasetUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    setDatasetFile(file)
    setDatasetLoading(true)
    
    try {
      const result = await datasetsAPI.upload(file)
      setDatasetTexts(result.texts || [])
      setDatasetResults({ total: result.total, message: result.message })
    } catch (err) {
      console.error('Error al cargar dataset:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setDatasetLoading(false)
    }
  }

  const handleAnalyzeDataset = async () => {
    if (datasetTexts.length === 0) {
      alert('Primero carga un dataset')
      return
    }
    
    setDatasetLoading(true)
    try {
      const results = await datasetsAPI.analyzeBatch(datasetTexts)
      setDatasetResults(results)
    } catch (err) {
      console.error('Error al analizar dataset:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setDatasetLoading(false)
    }
  }

  // Funciones para Búsqueda
  const handleSearch = async () => {
    if (!searchQuery.trim() || datasetTexts.length === 0) {
      alert('Primero carga un dataset y luego busca')
      return
    }
    
    try {
      const results = await datasetsAPI.search(searchQuery, datasetTexts)
      setSearchResults(results.results || [])
    } catch (err) {
      console.error('Error al buscar:', err)
      alert(`Error: ${err.message}`)
    }
  }

  const analysesLeft = user && user.plan === 'free' ? Math.max(0, 10 - analysesToday) : Infinity

  return (
    <section className="dashboard-section">
      <div className="section-header">
        <h1>Análisis de Sentimientos</h1>
        <p className="section-subtitle">
          Parte 1: Clasificación de texto con Red Neuronal - Carga datos, limpia texto y clasifica como positivo/negativo
        </p>
      </div>

      {/* Pestañas */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
        <button
          onClick={() => setActiveTab('individual')}
          style={{
            padding: '10px 20px',
            background: activeTab === 'individual' ? 'var(--primary)' : 'transparent',
            border: 'none',
            color: 'var(--text)',
            cursor: 'pointer',
            borderBottom: activeTab === 'individual' ? '2px solid var(--primary)' : '2px solid transparent'
          }}
        >
          Análisis Individual
        </button>
        <button
          onClick={() => setActiveTab('dataset')}
          style={{
            padding: '10px 20px',
            background: activeTab === 'dataset' ? 'var(--primary)' : 'transparent',
            border: 'none',
            color: 'var(--text)',
            cursor: 'pointer',
            borderBottom: activeTab === 'dataset' ? '2px solid var(--primary)' : '2px solid transparent'
          }}
        >
          Cargar Dataset
        </button>
        <button
          onClick={() => setActiveTab('search')}
          style={{
            padding: '10px 20px',
            background: activeTab === 'search' ? 'var(--primary)' : 'transparent',
            border: 'none',
            color: 'var(--text)',
            cursor: 'pointer',
            borderBottom: activeTab === 'search' ? '2px solid var(--primary)' : '2px solid transparent'
          }}
        >
          Búsqueda de Texto
        </button>
      </div>

      {/* Pestaña: Análisis Individual */}
      {activeTab === 'individual' && (
        <>

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
        </>
      )}

      {/* Pestaña: Cargar Dataset */}
      {activeTab === 'dataset' && (
        <div>
          <form className="api-form" style={{ marginBottom: '20px' }}>
            <div className="form-field">
              <label htmlFor="dataset-file">Cargar Dataset (CSV/JSON)</label>
              <input
                type="file"
                id="dataset-file"
                accept=".csv,.json"
                onChange={handleDatasetUpload}
                className="form-input"
              />
              <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
                {user?.plan === 'free' ? 'Máximo 100 comentarios' : 'Ilimitado'}
              </small>
            </div>
          </form>

          {datasetResults && datasetResults.total > 0 && (
            <div className="stats-grid" style={{ marginBottom: '20px' }}>
              <div className="stat-card">
                <div className="stat-value">{datasetResults.total}</div>
                <div className="stat-label">Comentarios Cargados</div>
              </div>
            </div>
          )}

          {datasetTexts.length > 0 && (
            <div className="form-actions">
              <button className="btn" onClick={handleAnalyzeDataset} disabled={datasetLoading}>
                {datasetLoading ? 'Analizando...' : 'Analizar Dataset con Red Neuronal'}
              </button>
            </div>
          )}

          {datasetResults && datasetResults.summary && (
            <div className="stats-chart" style={{ marginTop: '20px' }}>
              <h3>Resultados del Análisis</h3>
              <div className="chart-container">
                <div className="chart-item">
                  <div className="chart-label">
                    <span className="chart-emoji">🟢</span>
                    <span>Positivo</span>
                    <span className="chart-percent">{datasetResults.summary.positive_percent}%</span>
                  </div>
                  <div className="chart-bar">
                    <div 
                      className="chart-bar-fill chart-bar-pos" 
                      style={{ width: `${datasetResults.summary.positive_percent}%` }}
                    ></div>
                  </div>
                  <div className="chart-count">{datasetResults.summary.positive} de {datasetResults.total}</div>
                </div>

                <div className="chart-item">
                  <div className="chart-label">
                    <span className="chart-emoji">🔴</span>
                    <span>Negativo</span>
                    <span className="chart-percent">{datasetResults.summary.negative_percent}%</span>
                  </div>
                  <div className="chart-bar">
                    <div 
                      className="chart-bar-fill chart-bar-neg" 
                      style={{ width: `${datasetResults.summary.negative_percent}%` }}
                    ></div>
                  </div>
                  <div className="chart-count">{datasetResults.summary.negative} de {datasetResults.total}</div>
                </div>

                <div className="chart-item">
                  <div className="chart-label">
                    <span className="chart-emoji">🟡</span>
                    <span>Neutral</span>
                    <span className="chart-percent">{datasetResults.summary.neutral_percent}%</span>
                  </div>
                  <div className="chart-bar">
                    <div 
                      className="chart-bar-fill chart-bar-neu" 
                      style={{ width: `${datasetResults.summary.neutral_percent}%` }}
                    ></div>
                  </div>
                  <div className="chart-count">{datasetResults.summary.neutral} de {datasetResults.total}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Pestaña: Búsqueda de Texto */}
      {activeTab === 'search' && (
        <div>
          {datasetTexts.length === 0 ? (
            <div className="stats-empty">
              <p>Primero carga un dataset en la pestaña "Cargar Dataset"</p>
            </div>
          ) : (
            <>
              <form className="api-form" onSubmit={(e) => { e.preventDefault(); handleSearch(); }}>
                <div className="form-field">
                  <label htmlFor="search-query">Buscar en comentarios</label>
                  <input
                    type="text"
                    id="search-query"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Ej: atención, servicio, producto..."
                    className="form-input"
                  />
                </div>
                <div className="form-actions">
                  <button type="submit" className="btn">
                    Buscar
                  </button>
                </div>
              </form>

              {searchResults.length > 0 && (
                <div className="history-list" style={{ marginTop: '20px' }}>
                  <h3>Resultados de búsqueda ({searchResults.length})</h3>
                  {searchResults.map((item, index) => (
                    <div key={index} className="history-item">
                      <div className="history-text">{item.text}</div>
                      <div className="history-item-header">
                        <span>Coincidencias: {item.matches}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Explicación Técnica - Parte 1 */}
      <div className="message" style={{ marginTop: '30px', background: 'rgba(110, 139, 255, 0.1)', padding: '20px', borderRadius: '8px' }}>
        <h3 style={{ marginTop: 0 }}>Explicación Técnica - Parte 1</h3>
        <p><strong>Tipo de Aprendizaje:</strong> Supervisado</p>
        <p><strong>Algoritmo:</strong> Red Neuronal (LSTM)</p>
        <p><strong>Justificación:</strong> Las redes neuronales LSTM capturan el contexto y las relaciones semánticas en el texto, superando métodos basados en diccionarios.</p>
        <p><strong>Proceso de IA:</strong></p>
        <ol>
          <li>Recolección de datos (comentarios de redes sociales, blogs, etc.)</li>
          <li>Limpieza de texto (eliminar URLs, caracteres especiales, normalización)</li>
          <li>Tokenización (convertir texto a secuencias numéricas)</li>
          <li>Entrenamiento del modelo (ajustar pesos de la red neuronal)</li>
          <li>Clasificación (positivo/negativo/neutral)</li>
        </ol>
      </div>
    </section>
  )
}

export default ManualAnalysis


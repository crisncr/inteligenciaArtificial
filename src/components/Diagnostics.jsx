import { useState, useEffect } from 'react'
import { analysesAPI } from '../utils/api'

function Diagnostics({ user, history, onReanalyze }) {
  const [diagnostics, setDiagnostics] = useState([])
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState('all')
  const [apiFilter, setApiFilter] = useState('all')

  useEffect(() => {
    // Filtrar historial para mostrar solo diagnósticos de API externa
    if (history && history.length > 0) {
      const apiAnalyses = history.filter(a => a.source === 'api_external')
      setDiagnostics(apiAnalyses)
    } else {
      setDiagnostics([])
    }
  }, [history])

  const filteredDiagnostics = diagnostics.filter(item => {
    // Filtrar por sentimiento
    if (filter !== 'all') {
      if (filter === 'positivo' && item.sentiment !== 'positivo') return false
      if (filter === 'negativo' && item.sentiment !== 'negativo') return false
      if (filter === 'neutral' && item.sentiment !== 'neutral' && item.sentiment !== 'moderado/neutral') return false
    }
    return true
  })

  const formatDate = (timestamp) => {
    if (!timestamp) return 'Fecha no disponible'
    const date = new Date(timestamp)
    return new Intl.DateTimeFormat('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date)
  }

  const getResultClass = (sentiment) => {
    if (sentiment === 'positivo') return 'pos'
    if (sentiment === 'negativo') return 'neg'
    return 'neu'
  }

  const stats = {
    total: filteredDiagnostics.length,
    positive: filteredDiagnostics.filter(d => d.sentiment === 'positivo').length,
    negative: filteredDiagnostics.filter(d => d.sentiment === 'negativo').length,
    neutral: filteredDiagnostics.filter(d => d.sentiment === 'neutral' || d.sentiment === 'moderado/neutral').length
  }

  return (
    <section className="diagnostics-panel">
      <h2>Diagnósticos de API Externa</h2>
      <p className="subtitle">
        Historial de análisis obtenidos de APIs externas
      </p>

      {loading ? (
        <div className="loading">Cargando diagnósticos...</div>
      ) : (
        <>
          <div className="diagnostics-stats">
            <div className="stat-card">
              <div className="stat-value">{stats.total}</div>
              <div className="stat-label">Total</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.positive}</div>
              <div className="stat-label">Positivos</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.negative}</div>
              <div className="stat-label">Negativos</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.neutral}</div>
              <div className="stat-label">Neutrales</div>
            </div>
          </div>

          <div className="diagnostics-filters">
            <div className="filter-group">
              <label>Filtrar por sentimiento:</label>
              <div className="filter-buttons">
                <button
                  className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
                  onClick={() => setFilter('all')}
                >
                  Todos
                </button>
                <button
                  className={`filter-btn ${filter === 'positivo' ? 'active' : ''}`}
                  onClick={() => setFilter('positivo')}
                >
                  🟢 Positivos
                </button>
                <button
                  className={`filter-btn ${filter === 'negativo' ? 'active' : ''}`}
                  onClick={() => setFilter('negativo')}
                >
                  🔴 Negativos
                </button>
                <button
                  className={`filter-btn ${filter === 'neutral' ? 'active' : ''}`}
                  onClick={() => setFilter('neutral')}
                >
                  🟡 Neutrales
                </button>
              </div>
            </div>
          </div>

          {filteredDiagnostics.length === 0 ? (
            <div className="empty-state">
              <p>No hay diagnósticos {filter !== 'all' ? 'con el filtro seleccionado' : ''} aún.</p>
              <p>Configura una API externa y analiza comentarios para ver diagnósticos aquí.</p>
            </div>
          ) : (
            <div className="diagnostics-list">
              {filteredDiagnostics.map((item) => (
                <div key={item.id} className={`diagnostic-item ${getResultClass(item.sentiment)}`}>
                  <div className="diagnostic-header">
                    <span className="diagnostic-emoji">{item.emoji}</span>
                    <span className="diagnostic-sentiment">{item.sentiment}</span>
                    <span className="diagnostic-score">Score: {item.score}</span>
                    <span className="diagnostic-date">{formatDate(item.created_at)}</span>
                  </div>
                  <div className="diagnostic-text">{item.text}</div>
                  <div className="diagnostic-source">
                    <span>Fuente: API Externa</span>
                  </div>
                  {onReanalyze && (
                    <button 
                      className="btn--ghost btn--small" 
                      onClick={() => onReanalyze(item.text)}
                      title="Re-analizar este texto"
                    >
                      🔄 Re-analizar
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  )
}

export default Diagnostics


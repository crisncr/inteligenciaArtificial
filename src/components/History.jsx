import { useState, useEffect } from 'react'

function History({ history, onReanalyze, onClearHistory, filter, onFilterChange }) {
  const [expanded, setExpanded] = useState(false)
  const [currentFilter, setCurrentFilter] = useState(filter || 'all')

  useEffect(() => {
    if (onFilterChange) {
      onFilterChange(currentFilter)
    }
  }, [currentFilter, onFilterChange])

  const filteredHistory = currentFilter === 'all' 
    ? history 
    : history.filter(item => {
        if (currentFilter === 'positivo') return item.sentiment === 'positivo'
        if (currentFilter === 'negativo') return item.sentiment === 'negativo'
        if (currentFilter === 'neutral') return item.sentiment === 'neutral' || item.sentiment === 'moderado/neutral'
        return true
      })

  if (!history || history.length === 0) {
    return (
      <section className="history-panel">
        <h3>Historial de Análisis de API Externa</h3>
        <p className="subtitle">No hay análisis de API externa aún. Configura una API externa en la sección "API Externa" para comenzar a analizar comentarios automáticamente con Red Neuronal LSTM.</p>
      </section>
    )
  }

  const formatDate = (timestamp) => {
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

  const displayHistory = expanded ? filteredHistory : filteredHistory.slice(0, 10)

  const filters = [
    { id: 'all', label: 'Todos', icon: '📋' },
    { id: 'positivo', label: 'Positivos', icon: '🟢' },
    { id: 'negativo', label: 'Negativos', icon: '🔴' },
    { id: 'neutral', label: 'Neutrales', icon: '🟡' }
  ]

  return (
    <section className="history-panel">
      <div className="history-header">
        <h3>Historial de Análisis de API Externa</h3>
        <p className="subtitle" style={{ fontSize: '0.9rem', marginTop: '5px', opacity: 0.8 }}>
          Todos los análisis fueron realizados con Red Neuronal LSTM
        </p>
        <div className="history-actions">
          <div className="history-filters">
            {filters.map((f) => (
              <button
                key={f.id}
                className={`filter-btn ${currentFilter === f.id ? 'active' : ''}`}
                onClick={() => setCurrentFilter(f.id)}
              >
                <span>{f.icon}</span>
                <span>{f.label}</span>
              </button>
            ))}
          </div>
          {filteredHistory.length > 10 && (
            <button 
              className="btn--ghost btn--small" 
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? 'Ver menos' : `Ver todos (${filteredHistory.length})`}
            </button>
          )}
          <button 
            className="btn--ghost btn--small" 
            onClick={onClearHistory}
            title="Limpiar historial"
          >
            🗑️ Limpiar
          </button>
        </div>
      </div>
      {filteredHistory.length === 0 ? (
        <div className="history-empty">
          <p>No hay análisis {currentFilter !== 'all' ? filters.find(f => f.id === currentFilter)?.label.toLowerCase() : ''} aún</p>
        </div>
      ) : (
        <div className="history-list">
          {displayHistory.map((item, index) => (
          <div key={index} className={`history-item ${getResultClass(item.sentiment)}`}>
            <div className="history-item-header">
              <span className="history-emoji">{item.emoji}</span>
              <span className="history-sentiment">{item.sentiment}</span>
              <span className="history-score">Score: {item.score}</span>
              <span className="history-date">{formatDate(item.timestamp)}</span>
            </div>
            <div className="history-text">{item.text}</div>
            <button 
              className="btn--ghost btn--small" 
              onClick={() => onReanalyze(item.text)}
              title="Re-analizar este texto"
            >
              🔄 Re-analizar
            </button>
          </div>
        ))}
        </div>
      )}
    </section>
  )
}

export default History



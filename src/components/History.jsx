import { useState, useEffect } from 'react'

function History({ history, onReanalyze, onClearHistory }) {
  const [expanded, setExpanded] = useState(false)

  if (!history || history.length === 0) {
    return null
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

  const displayHistory = expanded ? history : history.slice(0, 3)

  return (
    <section className="history-panel">
      <div className="history-header">
        <h3>Historial de Análisis</h3>
        <div className="history-actions">
          {history.length > 3 && (
            <button 
              className="btn--ghost btn--small" 
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? 'Ver menos' : `Ver todos (${history.length})`}
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
    </section>
  )
}

export default History



function Stats({ history }) {
  if (!history || history.length === 0) {
    return (
      <section id="stats" className="stats-panel">
        <h2>Estadísticas de API Externa</h2>
        <p className="subtitle">No hay análisis de API externa aún. Configura una API externa en la sección "API Externa" para comenzar a analizar comentarios automáticamente.</p>
        <div className="stats-empty">
          <div className="stats-icon">📊</div>
          <p>No hay análisis de API externa aún</p>
        </div>
      </section>
    )
  }

  const total = history.length
  const positive = history.filter(h => h.sentiment === 'positivo').length
  const negative = history.filter(h => h.sentiment === 'negativo').length
  const neutral = history.filter(h => h.sentiment === 'moderado/neutral' || h.sentiment === 'neutral').length

  const positivePercent = ((positive / total) * 100).toFixed(1)
  const negativePercent = ((negative / total) * 100).toFixed(1)
  const neutralPercent = ((neutral / total) * 100).toFixed(1)

  const avgScore = history.reduce((sum, h) => sum + parseFloat(h.score || 0), 0) / total

  return (
    <section id="stats" className="stats-panel">
      <h2>Estadísticas de API Externa</h2>
      <p className="subtitle">Resumen de tus análisis de API externa</p>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{total}</div>
          <div className="stat-label">Total de Análisis</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{avgScore.toFixed(2)}</div>
          <div className="stat-label">Score Promedio</div>
        </div>
      </div>

      <div className="stats-chart">
        <h3>Distribución de Sentimientos</h3>
        <div className="chart-container">
          <div className="chart-item">
            <div className="chart-label">
              <span className="chart-emoji">🟢</span>
              <span>Positivo</span>
              <span className="chart-percent">{positivePercent}%</span>
            </div>
            <div className="chart-bar">
              <div 
                className="chart-bar-fill chart-bar-pos" 
                style={{ width: `${positivePercent}%` }}
              ></div>
            </div>
            <div className="chart-count">{positive} de {total}</div>
          </div>

          <div className="chart-item">
            <div className="chart-label">
              <span className="chart-emoji">🔴</span>
              <span>Negativo</span>
              <span className="chart-percent">{negativePercent}%</span>
            </div>
            <div className="chart-bar">
              <div 
                className="chart-bar-fill chart-bar-neg" 
                style={{ width: `${negativePercent}%` }}
              ></div>
            </div>
            <div className="chart-count">{negative} de {total}</div>
          </div>

          <div className="chart-item">
            <div className="chart-label">
              <span className="chart-emoji">🟡</span>
              <span>Neutral</span>
              <span className="chart-percent">{neutralPercent}%</span>
            </div>
            <div className="chart-bar">
              <div 
                className="chart-bar-fill chart-bar-neu" 
                style={{ width: `${neutralPercent}%` }}
              ></div>
            </div>
            <div className="chart-count">{neutral} de {total}</div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Stats



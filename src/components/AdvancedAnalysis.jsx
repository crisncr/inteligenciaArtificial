function AdvancedAnalysis({ user }) {
  return (
    <section className="advanced-analysis-panel">
      <h2>Análisis Avanzado</h2>
      <p className="subtitle">
        Funciones avanzadas de análisis de sentimientos disponibles solo en Enterprise
      </p>
      
      <div className="feature-grid">
        <div className="feature-card">
          <div className="feature-icon">🎯</div>
          <h3>Análisis Multi-idioma</h3>
          <p>Analiza sentimientos en múltiples idiomas con precisión mejorada</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🧠</div>
          <h3>Machine Learning Avanzado</h3>
          <p>Modelos de ML personalizados para tu industria específica</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">⚡</div>
          <h3>Procesamiento en Tiempo Real</h3>
          <p>Análisis instantáneo de grandes volúmenes de datos</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🔬</div>
          <h3>Análisis de Emociones</h3>
          <p>Detecta emociones específicas: alegría, tristeza, ira, miedo, sorpresa</p>
        </div>
      </div>
    </section>
  )
}

export default AdvancedAnalysis


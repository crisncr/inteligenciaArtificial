function Reports({ user }) {
  return (
    <section className="reports-panel">
      <h2>Reportes Personalizados</h2>
      <p className="subtitle">
        Genera reportes detallados de tus análisis de sentimientos
      </p>

      <div className="reports-grid">
        <div className="report-card">
          <div className="report-icon">📊</div>
          <h3>Reporte Diario</h3>
          <p>Resumen diario de todos tus análisis con gráficos y estadísticas</p>
          <button className="btn btn--ghost">Generar Reporte</button>
        </div>

        <div className="report-card">
          <div className="report-icon">📈</div>
          <h3>Reporte Semanal</h3>
          <p>Análisis de tendencias semanales con comparativas y insights</p>
          <button className="btn btn--ghost">Generar Reporte</button>
        </div>

        <div className="report-card">
          <div className="report-icon">📉</div>
          <h3>Reporte Mensual</h3>
          <p>Reporte completo mensual con análisis profundo y recomendaciones</p>
          <button className="btn btn--ghost">Generar Reporte</button>
        </div>

        <div className="report-card">
          <div className="report-icon">🎯</div>
          <h3>Reporte Personalizado</h3>
          <p>Crea tu propio reporte con los parámetros que necesites</p>
          <button className="btn">Crear Reporte</button>
        </div>
      </div>
    </section>
  )
}

export default Reports


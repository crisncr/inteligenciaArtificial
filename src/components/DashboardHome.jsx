function DashboardHome({ user }) {
  return (
    <section className="dashboard-home">
      <div className="welcome-section">
        <div className="welcome-content">
          <h1>¡Bienvenido, {user?.name || 'Usuario'}! 👋</h1>
          <p className="welcome-subtitle">
            Gestiona tus análisis de sentimientos desde APIs externas y obtén insights valiosos
          </p>
        </div>
        <div className="welcome-image">
          <div className="image-placeholder">
            <svg width="200" height="200" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="100" cy="100" r="80" fill="url(#gradient1)" opacity="0.3"/>
              <circle cx="100" cy="100" r="50" fill="url(#gradient2)" opacity="0.5"/>
              <path d="M70 100 L90 120 L130 80" stroke="var(--primary)" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"/>
              <defs>
                <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="var(--primary)" />
                  <stop offset="100%" stopColor="var(--secondary)" />
                </linearGradient>
                <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="var(--secondary)" />
                  <stop offset="100%" stopColor="var(--primary)" />
                </linearGradient>
              </defs>
            </svg>
          </div>
        </div>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">🔌</div>
          <h3>API Externa</h3>
          <p>Conecta tu API externa para obtener comentarios automáticamente y analizarlos en tiempo real</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>Estadísticas</h3>
          <p>Visualiza métricas detalladas de tus análisis de sentimientos con gráficos y resúmenes</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📋</div>
          <h3>Historial</h3>
          <p>Revisa todos tus diagnósticos de API externa con filtros avanzados por sentimiento</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🔍</div>
          <h3>Diagnósticos</h3>
          <p>Analiza comentarios de tus APIs externas y obtén insights sobre el sentimiento de tus usuarios</p>
        </div>
      </div>

      <div className="quick-stats">
        <div className="stat-card">
          <div className="stat-icon">🚀</div>
          <div className="stat-content">
            <h3>Comienza Ahora</h3>
            <p>Configura tu primera API externa en la sección "API Externa" para empezar a analizar comentarios automáticamente</p>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">📈</div>
          <div className="stat-content">
            <h3>Monitorea Resultados</h3>
            <p>Revisa tus estadísticas y diagnósticos para entender mejor el sentimiento de tus usuarios</p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default DashboardHome


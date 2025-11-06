function Integrations({ user }) {
  const integrations = [
    {
      name: 'Slack',
      icon: '💬',
      description: 'Recibe notificaciones de análisis en tiempo real',
      status: 'available'
    },
    {
      name: 'Zapier',
      icon: '⚡',
      description: 'Conecta con más de 5000 aplicaciones',
      status: 'available'
    },
    {
      name: 'Webhooks',
      icon: '🔗',
      description: 'Recibe eventos de análisis mediante webhooks',
      status: 'available'
    },
    {
      name: 'API REST',
      icon: '🌐',
      description: 'Integración completa mediante API REST',
      status: 'available'
    }
  ]

  return (
    <section className="integrations-panel">
      <h2>Integraciones</h2>
      <p className="subtitle">
        Conecta Sentimetría con tus herramientas favoritas
      </p>

      <div className="integrations-grid">
        {integrations.map((integration, index) => (
          <div key={index} className="integration-card">
            <div className="integration-header">
              <div className="integration-icon">{integration.icon}</div>
              <div className="integration-status">
                <span className={`status-badge ${integration.status}`}>
                  {integration.status === 'available' ? 'Disponible' : 'Próximamente'}
                </span>
              </div>
            </div>
            <h3>{integration.name}</h3>
            <p>{integration.description}</p>
            <button className="btn btn--ghost">
              {integration.status === 'available' ? 'Configurar' : 'Próximamente'}
            </button>
          </div>
        ))}
      </div>
    </section>
  )
}

export default Integrations


import { useState } from 'react'

function Integrations({ user }) {
  const [activeTab, setActiveTab] = useState('outgoing')

  // Integraciones salientes: para que otros consuman nuestra API
  const outgoingIntegrations = [
    {
      name: 'API REST',
      icon: '🌐',
      description: 'Nuestra API REST para que otras aplicaciones consuman análisis de sentimientos',
      type: 'api_rest',
      status: 'available',
      config: {
        baseUrl: 'https://inteligenciaartificial-1-2ljl.onrender.com/api',
        endpoints: ['/analyses', '/analyses/stats/summary'],
        auth: 'Bearer Token'
      }
    },
    {
      name: 'Webhooks',
      icon: '🔗',
      description: 'Recibe eventos de análisis mediante webhooks en tiempo real',
      type: 'webhooks',
      status: 'available',
      config: {
        events: ['analysis.created', 'analysis.completed', 'batch.analyzed']
      }
    }
  ]

  // Integraciones entrantes: para consumir APIs externas
  const incomingIntegrations = [
    {
      name: 'API Externa',
      icon: '🔌',
      description: 'Conecta APIs externas para consumir comentarios y analizarlos automáticamente',
      type: 'external_api',
      status: 'available',
      config: {
        purpose: 'Consumir comentarios de APIs externas y analizarlos'
      }
    },
    {
      name: 'Slack',
      icon: '💬',
      description: 'Recibe notificaciones de análisis en tiempo real en Slack',
      type: 'slack',
      status: 'available',
      config: {
        channels: ['#sentiment-analysis', '#notifications']
      }
    },
    {
      name: 'Zapier',
      icon: '⚡',
      description: 'Conecta con más de 5000 aplicaciones mediante Zapier',
      type: 'zapier',
      status: 'available',
      config: {
        triggers: ['New Analysis', 'Batch Completed']
      }
    }
  ]

  const handleConfigure = (integration) => {
    if (integration.type === 'api_rest') {
      // Mostrar información de la API REST
      alert(`API REST Configurada\n\nBase URL: ${integration.config.baseUrl}\nEndpoints: ${integration.config.endpoints.join(', ')}\nAutenticación: ${integration.config.auth}\n\nUsa tu token de autenticación para consumir la API.`)
    } else if (integration.type === 'webhooks') {
      // Configurar webhooks
      const webhookUrl = prompt('Ingresa la URL del webhook donde recibirás los eventos:')
      if (webhookUrl) {
        alert(`Webhook configurado: ${webhookUrl}\n\nEventos disponibles: ${integration.config.events.join(', ')}`)
      }
    } else if (integration.type === 'external_api') {
      // Redirigir a la sección de API Externa
      alert('Redirigiendo a la sección de API Externa para configurar...')
      // Aquí podrías usar un router para navegar
    } else if (integration.type === 'slack') {
      // Configurar Slack
      const slackWebhook = prompt('Ingresa la URL del webhook de Slack:')
      if (slackWebhook) {
        alert(`Slack configurado. Recibirás notificaciones en los canales configurados.`)
      }
    } else if (integration.type === 'zapier') {
      // Abrir configuración de Zapier
      alert('Redirigiendo a Zapier para configurar la integración...')
    }
  }

  return (
    <section className="integrations-panel">
      <div className="panel-header">
        <div>
          <h2>Integraciones</h2>
          <p className="subtitle">
            Conecta Sentimetría con tus herramientas favoritas
          </p>
        </div>
      </div>

      <div className="integrations-tabs">
        <button
          className={`tab-button ${activeTab === 'outgoing' ? 'active' : ''}`}
          onClick={() => setActiveTab('outgoing')}
        >
          Salientes (Nuestra API)
        </button>
        <button
          className={`tab-button ${activeTab === 'incoming' ? 'active' : ''}`}
          onClick={() => setActiveTab('incoming')}
        >
          Entrantes (APIs Externas)
        </button>
      </div>

      <div className="integrations-content">
        {activeTab === 'outgoing' && (
          <div className="integrations-grid">
            {outgoingIntegrations.map((integration, index) => (
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
                {integration.type === 'api_rest' && (
                  <div className="integration-details">
                    <p><strong>Base URL:</strong> {integration.config.baseUrl}</p>
                    <p><strong>Autenticación:</strong> {integration.config.auth}</p>
                  </div>
                )}
                <button
                  className="btn btn--ghost"
                  onClick={() => handleConfigure(integration)}
                >
                  {integration.status === 'available' ? 'Configurar' : 'Próximamente'}
                </button>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'incoming' && (
          <div className="integrations-grid">
            {incomingIntegrations.map((integration, index) => (
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
                <button
                  className="btn btn--ghost"
                  onClick={() => handleConfigure(integration)}
                >
                  {integration.status === 'available' ? 'Configurar' : 'Próximamente'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="integrations-info">
        <h3>Diferencia entre Integraciones</h3>
        <div className="info-grid">
          <div className="info-card">
            <h4>🔌 API Externa (Entrante)</h4>
            <p>Consume comentarios de APIs externas y los analiza automáticamente. Los resultados se guardan en tu historial.</p>
          </div>
          <div className="info-card">
            <h4>🌐 API REST (Saliente)</h4>
            <p>Nuestra API que otras aplicaciones pueden consumir para obtener análisis de sentimientos. Tú proporcionas el servicio.</p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Integrations

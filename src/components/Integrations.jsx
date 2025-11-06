import { useState } from 'react'

function Integrations({ user, onSectionChange }) {
  const [activeTab, setActiveTab] = useState('outgoing')
  const [configuringIntegration, setConfiguringIntegration] = useState(null)
  const [configData, setConfigData] = useState({})
  const [savedConfigs, setSavedConfigs] = useState({})

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
      description: 'Envía eventos de análisis mediante webhooks en tiempo real a URLs externas',
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
    if (integration.type === 'external_api') {
      // Redirigir a la sección de API Externa
      if (onSectionChange) {
        onSectionChange('api-externa')
      }
      return
    }

    // Inicializar datos de configuración según el tipo
    const initialData = savedConfigs[integration.type] || {}
    
    if (integration.type === 'api_rest') {
      setConfigData({
        baseUrl: initialData.baseUrl || integration.config.baseUrl,
        endpoints: initialData.endpoints || integration.config.endpoints.join(', '),
        auth: initialData.auth || integration.config.auth,
        token: initialData.token || ''
      })
    } else if (integration.type === 'webhooks') {
      setConfigData({
        webhookUrl: initialData.webhookUrl || '',
        events: initialData.events || integration.config.events,
        secret: initialData.secret || ''
      })
    } else if (integration.type === 'slack') {
      setConfigData({
        webhookUrl: initialData.webhookUrl || '',
        channels: initialData.channels || integration.config.channels.join(', '),
        events: initialData.events || ['analysis.created', 'analysis.completed']
      })
    } else if (integration.type === 'zapier') {
      setConfigData({
        apiKey: initialData.apiKey || '',
        triggers: initialData.triggers || integration.config.triggers,
        webhookUrl: initialData.webhookUrl || ''
      })
    }

    setConfiguringIntegration(integration)
  }

  const handleSaveConfig = (e) => {
    e.preventDefault()
    
    if (!configuringIntegration) return

    // Guardar configuración
    setSavedConfigs(prev => ({
      ...prev,
      [configuringIntegration.type]: { ...configData }
    }))

    // Cerrar formulario
    setConfiguringIntegration(null)
    setConfigData({})
    
    // Mostrar mensaje de éxito
    alert(`Configuración de ${configuringIntegration.name} guardada correctamente.`)
  }

  const handleCancelConfig = () => {
    setConfiguringIntegration(null)
    setConfigData({})
  }

  const renderConfigForm = () => {
    if (!configuringIntegration) return null

    const { type, name } = configuringIntegration

    if (type === 'api_rest') {
      return (
        <form onSubmit={handleSaveConfig} className="integration-config-form">
          <h3>Configurar API REST</h3>
          <div className="form-group">
            <label htmlFor="baseUrl">Base URL</label>
            <input
              type="url"
              id="baseUrl"
              value={configData.baseUrl || ''}
              onChange={(e) => setConfigData({ ...configData, baseUrl: e.target.value })}
              required
              placeholder="https://api.ejemplo.com"
            />
          </div>
          <div className="form-group">
            <label htmlFor="endpoints">Endpoints (separados por comas)</label>
            <input
              type="text"
              id="endpoints"
              value={configData.endpoints || ''}
              onChange={(e) => setConfigData({ ...configData, endpoints: e.target.value })}
              required
              placeholder="/analyses, /analyses/stats/summary"
            />
          </div>
          <div className="form-group">
            <label htmlFor="auth">Tipo de Autenticación</label>
            <select
              id="auth"
              value={configData.auth || 'Bearer Token'}
              onChange={(e) => setConfigData({ ...configData, auth: e.target.value })}
            >
              <option value="Bearer Token">Bearer Token</option>
              <option value="API Key">API Key</option>
              <option value="Basic Auth">Basic Auth</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="token">Token/API Key (opcional, para documentación)</label>
            <input
              type="password"
              id="token"
              value={configData.token || ''}
              onChange={(e) => setConfigData({ ...configData, token: e.target.value })}
              placeholder="Ingresa tu token para documentación"
            />
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn--primary">Guardar Configuración</button>
            <button type="button" className="btn btn--ghost" onClick={handleCancelConfig}>Cancelar</button>
          </div>
        </form>
      )
    }

    if (type === 'webhooks') {
      return (
        <form onSubmit={handleSaveConfig} className="integration-config-form">
          <h3>Configurar Webhooks</h3>
          <div className="form-group">
            <label htmlFor="webhookUrl">URL del Webhook</label>
            <input
              type="url"
              id="webhookUrl"
              value={configData.webhookUrl || ''}
              onChange={(e) => setConfigData({ ...configData, webhookUrl: e.target.value })}
              required
              placeholder="https://tu-app.com/webhook"
            />
            <small>Esta URL recibirá eventos cuando se realicen análisis</small>
          </div>
          <div className="form-group">
            <label>Eventos a Enviar</label>
            <div className="checkbox-group">
              {outgoingIntegrations.find(i => i.type === 'webhooks')?.config.events.map(event => (
                <label key={event} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={configData.events?.includes(event) || false}
                    onChange={(e) => {
                      const events = configData.events || []
                      if (e.target.checked) {
                        setConfigData({ ...configData, events: [...events, event] })
                      } else {
                        setConfigData({ ...configData, events: events.filter(e => e !== event) })
                      }
                    }}
                  />
                  <span>{event}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="form-group">
            <label htmlFor="secret">Secret (opcional, para validación)</label>
            <input
              type="password"
              id="secret"
              value={configData.secret || ''}
              onChange={(e) => setConfigData({ ...configData, secret: e.target.value })}
              placeholder="Secret para validar webhooks"
            />
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn--primary">Guardar Configuración</button>
            <button type="button" className="btn btn--ghost" onClick={handleCancelConfig}>Cancelar</button>
          </div>
        </form>
      )
    }

    if (type === 'slack') {
      return (
        <form onSubmit={handleSaveConfig} className="integration-config-form">
          <h3>Configurar Slack</h3>
          <div className="form-group">
            <label htmlFor="slackWebhookUrl">Webhook URL de Slack</label>
            <input
              type="url"
              id="slackWebhookUrl"
              value={configData.webhookUrl || ''}
              onChange={(e) => setConfigData({ ...configData, webhookUrl: e.target.value })}
              required
              placeholder="https://hooks.slack.com/services/..."
            />
            <small>Obtén tu webhook URL desde la configuración de Slack Apps</small>
          </div>
          <div className="form-group">
            <label htmlFor="channels">Canales (separados por comas)</label>
            <input
              type="text"
              id="channels"
              value={configData.channels || ''}
              onChange={(e) => setConfigData({ ...configData, channels: e.target.value })}
              placeholder="#sentiment-analysis, #notifications"
            />
          </div>
          <div className="form-group">
            <label>Eventos a Notificar</label>
            <div className="checkbox-group">
              {['analysis.created', 'analysis.completed', 'batch.analyzed'].map(event => (
                <label key={event} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={configData.events?.includes(event) || false}
                    onChange={(e) => {
                      const events = configData.events || []
                      if (e.target.checked) {
                        setConfigData({ ...configData, events: [...events, event] })
                      } else {
                        setConfigData({ ...configData, events: events.filter(e => e !== event) })
                      }
                    }}
                  />
                  <span>{event}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn--primary">Guardar Configuración</button>
            <button type="button" className="btn btn--ghost" onClick={handleCancelConfig}>Cancelar</button>
          </div>
        </form>
      )
    }

    if (type === 'zapier') {
      return (
        <form onSubmit={handleSaveConfig} className="integration-config-form">
          <h3>Configurar Zapier</h3>
          <div className="form-group">
            <label htmlFor="zapierApiKey">API Key de Zapier</label>
            <input
              type="password"
              id="zapierApiKey"
              value={configData.apiKey || ''}
              onChange={(e) => setConfigData({ ...configData, apiKey: e.target.value })}
              required
              placeholder="Tu API Key de Zapier"
            />
            <small>Obtén tu API Key desde tu cuenta de Zapier</small>
          </div>
          <div className="form-group">
            <label htmlFor="zapierWebhookUrl">Webhook URL de Zapier (opcional)</label>
            <input
              type="url"
              id="zapierWebhookUrl"
              value={configData.webhookUrl || ''}
              onChange={(e) => setConfigData({ ...configData, webhookUrl: e.target.value })}
              placeholder="https://hooks.zapier.com/hooks/catch/..."
            />
          </div>
          <div className="form-group">
            <label>Triggers Disponibles</label>
            <div className="checkbox-group">
              {incomingIntegrations.find(i => i.type === 'zapier')?.config.triggers.map(trigger => (
                <label key={trigger} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={configData.triggers?.includes(trigger) || false}
                    onChange={(e) => {
                      const triggers = configData.triggers || []
                      if (e.target.checked) {
                        setConfigData({ ...configData, triggers: [...triggers, trigger] })
                      } else {
                        setConfigData({ ...configData, triggers: triggers.filter(t => t !== trigger) })
                      }
                    }}
                  />
                  <span>{trigger}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn--primary">Guardar Configuración</button>
            <button type="button" className="btn btn--ghost" onClick={handleCancelConfig}>Cancelar</button>
          </div>
        </form>
      )
    }

    return null
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
          onClick={() => {
            setActiveTab('outgoing')
            setConfiguringIntegration(null)
          }}
        >
          Salientes (Nuestra API)
        </button>
        <button
          className={`tab-button ${activeTab === 'incoming' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('incoming')
            setConfiguringIntegration(null)
          }}
        >
          Entrantes (APIs Externas)
        </button>
      </div>

      {configuringIntegration ? (
        <div className="integration-config-container">
          {renderConfigForm()}
        </div>
      ) : (
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
                  {savedConfigs[integration.type] && (
                    <div className="integration-saved-badge">
                      ✓ Configurado
                    </div>
                  )}
                  <button
                    className="btn btn--ghost"
                    onClick={() => handleConfigure(integration)}
                  >
                    {savedConfigs[integration.type] ? 'Editar Configuración' : 'Configurar'}
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
                  {savedConfigs[integration.type] && (
                    <div className="integration-saved-badge">
                      ✓ Configurado
                    </div>
                  )}
                  <button
                    className="btn btn--ghost"
                    onClick={() => handleConfigure(integration)}
                  >
                    {savedConfigs[integration.type] ? 'Editar Configuración' : 'Configurar'}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

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

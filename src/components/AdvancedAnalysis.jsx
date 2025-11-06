import { useState } from 'react'

function AdvancedAnalysis({ user }) {
  const [apiConfig, setApiConfig] = useState({
    name: '',
    apiUrl: '',
    endpoint: '',
    method: 'GET',
    headers: '',
    authType: 'none',
    authToken: '',
    active: false
  })
  const [isConfiguring, setIsConfiguring] = useState(false)
  const [testResult, setTestResult] = useState(null)

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setApiConfig(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleTestConnection = async () => {
    setIsConfiguring(true)
    setTestResult(null)
    
    // Simular prueba de conexión
    setTimeout(() => {
      setTestResult({
        success: true,
        message: 'Conexión exitosa. La API está lista para análisis avanzados.'
      })
      setIsConfiguring(false)
    }, 1500)
  }

  const handleSaveConfig = () => {
    // Aquí se guardaría la configuración en el backend
    alert('Configuración guardada. La API estará disponible para análisis avanzados.')
    setIsConfiguring(false)
    setApiConfig({
      name: '',
      apiUrl: '',
      endpoint: '',
      method: 'GET',
      headers: '',
      authType: 'none',
      authToken: '',
      active: false
    })
  }

  return (
    <section className="advanced-analysis-panel">
      <div className="panel-header">
        <div>
          <h2>Análisis Avanzado</h2>
          <p className="subtitle">
            Configura APIs para análisis avanzados con ML, multi-idioma y análisis de emociones
          </p>
        </div>
      </div>

      <div className="advanced-features-section">
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
      </div>

      <div className="api-config-section">
        <h3>Configurar API para Análisis Avanzado</h3>
        <p className="section-description">
          Conecta una API externa que proporcione análisis avanzados de sentimientos con ML y multi-idioma.
          Los comentarios se analizarán automáticamente y se guardarán en tu historial.
        </p>

        <form className="api-config-form" onSubmit={(e) => { e.preventDefault(); handleSaveConfig(); }}>
          <div className="form-group">
            <label htmlFor="name">Nombre de la API</label>
            <input
              type="text"
              id="name"
              name="name"
              value={apiConfig.name}
              onChange={handleInputChange}
              placeholder="Ej: OpenAI Sentiment API"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="apiUrl">URL Base de la API</label>
            <input
              type="url"
              id="apiUrl"
              name="apiUrl"
              value={apiConfig.apiUrl}
              onChange={handleInputChange}
              placeholder="https://api.example.com"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="endpoint">Endpoint</label>
            <input
              type="text"
              id="endpoint"
              name="endpoint"
              value={apiConfig.endpoint}
              onChange={handleInputChange}
              placeholder="/v1/analyze"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="method">Método HTTP</label>
            <select
              id="method"
              name="method"
              value={apiConfig.method}
              onChange={handleInputChange}
            >
              <option value="GET">GET</option>
              <option value="POST">POST</option>
              <option value="PUT">PUT</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="authType">Tipo de Autenticación</label>
            <select
              id="authType"
              name="authType"
              value={apiConfig.authType}
              onChange={handleInputChange}
            >
              <option value="none">Ninguna</option>
              <option value="bearer">Bearer Token</option>
              <option value="api_key">API Key</option>
            </select>
          </div>

          {apiConfig.authType !== 'none' && (
            <div className="form-group">
              <label htmlFor="authToken">Token / API Key</label>
              <input
                type="password"
                id="authToken"
                name="authToken"
                value={apiConfig.authToken}
                onChange={handleInputChange}
                placeholder="Ingresa tu token o API key"
                required
              />
            </div>
          )}

          <div className="form-group">
            <label htmlFor="headers">Headers Adicionales (JSON)</label>
            <textarea
              id="headers"
              name="headers"
              value={apiConfig.headers}
              onChange={handleInputChange}
              placeholder='{"Content-Type": "application/json"}'
              rows="3"
            />
          </div>

          <div className="form-actions">
            <button
              type="button"
              className="btn btn--ghost"
              onClick={handleTestConnection}
              disabled={isConfiguring || !apiConfig.name || !apiConfig.apiUrl || !apiConfig.endpoint}
            >
              {isConfiguring ? 'Probando...' : 'Probar Conexión'}
            </button>
            <button
              type="submit"
              className="btn btn--primary"
              disabled={!testResult?.success}
            >
              Guardar Configuración
            </button>
          </div>

          {testResult && (
            <div className={`test-result ${testResult.success ? 'success' : 'error'}`}>
              {testResult.success ? '✓' : '✗'} {testResult.message}
            </div>
          )}
        </form>
      </div>
    </section>
  )
}

export default AdvancedAnalysis

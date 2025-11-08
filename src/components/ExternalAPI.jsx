import { useState, useEffect } from 'react'
import { externalAPI } from '../utils/api'

function ExternalAPI({ user, onAnalyze }) {
  const [apis, setApis] = useState([])
  const [loading, setLoading] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [editingApi, setEditingApi] = useState(null)
  const [message, setMessage] = useState({ type: '', text: '' })
  
  const [formData, setFormData] = useState({
    name: '',
    api_url: '',
    endpoint: '',
    method: 'GET',
    headers: '',
    auth_type: '',
    auth_token: '',
    active: true
  })

  useEffect(() => {
    loadAPIs()
  }, [])

  const loadAPIs = async () => {
    try {
      const data = await externalAPI.getAll()
      setApis(data)
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Error al cargar APIs' })
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMessage({ type: '', text: '' })

    try {
      const headers = formData.headers ? JSON.parse(formData.headers) : null
      
      const apiData = {
        name: formData.name,
        api_url: formData.api_url,
        endpoint: formData.endpoint,
        method: formData.method,
        headers: headers,
        auth_type: formData.auth_type || null,
        auth_token: formData.auth_token || null,
        active: formData.active
      }

      if (editingApi) {
        await externalAPI.update(editingApi.id, apiData)
        setMessage({ type: 'success', text: 'API actualizada correctamente' })
      } else {
        await externalAPI.create(apiData)
        setMessage({ type: 'success', text: 'API creada correctamente' })
      }

      setShowForm(false)
      setEditingApi(null)
      setFormData({
        name: '',
        api_url: '',
        endpoint: '',
        method: 'GET',
        headers: '',
        auth_type: '',
        auth_token: '',
        active: true
      })
      loadAPIs()
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Error al guardar API' })
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = (api) => {
    setEditingApi(api)
    setFormData({
      name: api.name,
      api_url: api.api_url,
      endpoint: api.endpoint,
      method: api.method,
      headers: api.headers ? JSON.stringify(api.headers, null, 2) : '',
      auth_type: api.auth_type || '',
      auth_token: api.auth_token || '',
      active: api.active
    })
    setShowForm(true)
  }

  const handleDelete = async (id) => {
    if (!window.confirm('¿Estás seguro de que quieres eliminar esta API?')) return

    try {
      await externalAPI.delete(id)
      setMessage({ type: 'success', text: 'API eliminada correctamente' })
      loadAPIs()
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Error al eliminar API' })
    }
  }

  const handleTest = async (id) => {
    setLoading(true)
    setMessage({ type: '', text: '' })

    try {
      const result = await externalAPI.test(id)
      if (result.success) {
        setMessage({ type: 'success', text: result.message })
      } else {
        setMessage({ type: 'error', text: result.message })
      }
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Error al probar API' })
    } finally {
      setLoading(false)
    }
  }

  const handleAnalyze = async (id) => {
    setLoading(true)
    setMessage({ type: '', text: '' })

    try {
      const result = await externalAPI.analyze(id)
      setMessage({ 
        type: 'success', 
        text: `✅ Análisis completado con Red Neuronal LSTM: ${result.analyses_created} comentarios analizados de ${result.comments_count} encontrados${result.errors ? '. Algunos errores: ' + result.errors.join(', ') : ''}` 
      })
      if (onAnalyze) {
        // Esperar un poco para que el backend procese
        setTimeout(() => {
          onAnalyze()
        }, 500)
      }
    } catch (err) {
      setMessage({ type: 'error', text: err.response?.data?.detail || err.message || 'Error al analizar comentarios' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="external-api-panel">
      <div className="panel-header">
        <h2>API Externa</h2>
        <p className="subtitle">
          Configura una API externa para obtener comentarios y analizarlos automáticamente con Red Neuronal LSTM
        </p>
        <button 
          className="btn" 
          onClick={() => {
            setShowForm(!showForm)
            setEditingApi(null)
            setFormData({
              name: '',
              api_url: '',
              endpoint: '',
              method: 'GET',
              headers: '',
              auth_type: '',
              auth_token: '',
              active: true
            })
          }}
        >
          {showForm ? 'Cancelar' : '+ Nueva API'}
        </button>
      </div>

      {message.text && (
        <div className={`message ${message.type}`}>
          {message.text}
        </div>
      )}

      {showForm && (
        <form onSubmit={handleSubmit} className="api-form">
          <div className="form-row">
            <div className="form-field">
              <label htmlFor="name">Nombre de la API</label>
              <input
                type="text"
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                required
                placeholder="Ej: API de Comentarios"
              />
            </div>
            <div className="form-field">
              <label htmlFor="api_url">URL Base</label>
              <input
                type="url"
                id="api_url"
                value={formData.api_url}
                onChange={(e) => setFormData({ ...formData, api_url: e.target.value })}
                required
                placeholder="https://api.ejemplo.com"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-field">
              <label htmlFor="endpoint">Endpoint</label>
              <input
                type="text"
                id="endpoint"
                value={formData.endpoint}
                onChange={(e) => setFormData({ ...formData, endpoint: e.target.value })}
                required
                placeholder="/comments o /api/comments"
              />
            </div>
            <div className="form-field">
              <label htmlFor="method">Método</label>
              <select
                id="method"
                value={formData.method}
                onChange={(e) => setFormData({ ...formData, method: e.target.value })}
              >
                <option value="GET">GET</option>
                <option value="POST">POST</option>
              </select>
            </div>
          </div>

          <div className="form-field">
            <label htmlFor="headers">Headers (JSON opcional)</label>
            <textarea
              id="headers"
              value={formData.headers}
              onChange={(e) => setFormData({ ...formData, headers: e.target.value })}
              placeholder='{"Content-Type": "application/json"}'
              rows="3"
            />
          </div>

          <div className="form-row">
            <div className="form-field">
              <label htmlFor="auth_type">Tipo de Autenticación</label>
              <select
                id="auth_type"
                value={formData.auth_type}
                onChange={(e) => setFormData({ ...formData, auth_type: e.target.value })}
              >
                <option value="">Ninguna</option>
                <option value="bearer">Bearer Token</option>
                <option value="api_key">API Key</option>
                <option value="basic">Basic Auth</option>
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="auth_token">Token/API Key</label>
              <input
                type="password"
                id="auth_token"
                value={formData.auth_token}
                onChange={(e) => setFormData({ ...formData, auth_token: e.target.value })}
                placeholder={formData.auth_type ? "Ingresa tu token" : ""}
                disabled={!formData.auth_type}
              />
            </div>
          </div>

          <div className="form-field">
            <label>
              <input
                type="checkbox"
                checked={formData.active}
                onChange={(e) => setFormData({ ...formData, active: e.target.checked })}
              />
              Activa
            </label>
          </div>

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Guardando...' : editingApi ? 'Actualizar' : 'Crear API'}
          </button>
        </form>
      )}

      <div className="apis-list">
        {apis.length === 0 ? (
          <div className="empty-state">
            <p>No hay APIs configuradas. Crea una nueva API para comenzar.</p>
          </div>
        ) : (
          apis.map((api) => (
            <div key={api.id} className={`api-card ${api.active ? '' : 'inactive'}`}>
              <div className="api-header">
                <h3>{api.name}</h3>
                <span className={`api-status ${api.active ? 'active' : 'inactive'}`}>
                  {api.active ? 'Activa' : 'Inactiva'}
                </span>
              </div>
              <div className="api-info">
                <p><strong>URL:</strong> {api.api_url}/{api.endpoint}</p>
                <p><strong>Método:</strong> {api.method}</p>
                {api.auth_type && <p><strong>Auth:</strong> {api.auth_type}</p>}
              </div>
              <div className="api-actions">
                <button 
                  className="btn--ghost btn--small" 
                  onClick={() => handleTest(api.id)}
                  disabled={loading}
                >
                  🧪 Probar
                </button>
                <button 
                  className="btn--ghost btn--small" 
                  onClick={() => handleAnalyze(api.id)}
                  disabled={loading || !api.active}
                  title="Analizar comentarios con Red Neuronal LSTM"
                >
                  📊 Analizar (Red Neuronal)
                </button>
                <button 
                  className="btn--ghost btn--small" 
                  onClick={() => handleEdit(api)}
                >
                  ✏️ Editar
                </button>
                <button 
                  className="btn--ghost btn--small" 
                  onClick={() => handleDelete(api.id)}
                >
                  🗑️ Eliminar
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  )
}

export default ExternalAPI


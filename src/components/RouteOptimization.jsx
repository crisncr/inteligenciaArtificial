import { useState } from 'react'
import { routeOptimizationAPI } from '../utils/api'

function RouteOptimization({ user }) {
  const [points, setPoints] = useState([])
  const [newPoint, setNewPoint] = useState({ name: '', lat: '', lng: '' })
  const [algorithm, setAlgorithm] = useState('astar')
  const [routeResult, setRouteResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleAddPoint = () => {
    if (!newPoint.name || !newPoint.lat || !newPoint.lng) {
      alert('Por favor completa todos los campos')
      return
    }

    const lat = parseFloat(newPoint.lat)
    const lng = parseFloat(newPoint.lng)

    if (isNaN(lat) || isNaN(lng)) {
      alert('Latitud y longitud deben ser números válidos')
      return
    }

    setPoints([...points, { name: newPoint.name, lat, lng }])
    setNewPoint({ name: '', lat: '', lng: '' })
  }

  const handleRemovePoint = (index) => {
    setPoints(points.filter((_, i) => i !== index))
  }

  const handleCalculateRoute = async () => {
    if (points.length < 2) {
      alert('Necesitas al menos 2 puntos para calcular una ruta')
      return
    }

    setLoading(true)
    setRouteResult(null)

    try {
      const result = await routeOptimizationAPI.optimize(points, algorithm, 0)
      setRouteResult(result)
    } catch (err) {
      console.error('Error al calcular ruta:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="dashboard-section">
      <div className="section-header">
        <h1>Optimización de Rutas</h1>
        <p className="section-subtitle">
          Parte 2: Algoritmos de búsqueda - Optimiza rutas de distribución minimizando distancia
        </p>
      </div>

      <form className="api-form" style={{ marginBottom: '20px' }}>
        <div className="form-row">
          <div className="form-field">
            <label htmlFor="point-name">Nombre del Punto</label>
            <input
              type="text"
              id="point-name"
              value={newPoint.name}
              onChange={(e) => setNewPoint({ ...newPoint, name: e.target.value })}
              placeholder="Ej: Almacén"
            />
          </div>
          <div className="form-field">
            <label htmlFor="point-lat">Latitud</label>
            <input
              type="number"
              id="point-lat"
              step="any"
              value={newPoint.lat}
              onChange={(e) => setNewPoint({ ...newPoint, lat: e.target.value })}
              placeholder="Ej: -12.0464"
            />
          </div>
          <div className="form-field">
            <label htmlFor="point-lng">Longitud</label>
            <input
              type="number"
              id="point-lng"
              step="any"
              value={newPoint.lng}
              onChange={(e) => setNewPoint({ ...newPoint, lng: e.target.value })}
              placeholder="Ej: -77.0428"
            />
          </div>
        </div>
        <button type="button" className="btn" onClick={handleAddPoint}>
          + Agregar Punto
        </button>
      </form>

      {points.length > 0 && (
        <div className="apis-list" style={{ marginBottom: '20px' }}>
          <h3>Puntos Agregados ({points.length})</h3>
          {points.map((point, index) => (
            <div key={index} className="api-card">
              <div className="api-header">
                <h3>{point.name}</h3>
                <button
                  type="button"
                  className="btn btn--ghost btn--small"
                  onClick={() => handleRemovePoint(index)}
                >
                  Eliminar
                </button>
              </div>
              <div className="api-info">
                <p><strong>Lat:</strong> {point.lat}</p>
                <p><strong>Lng:</strong> {point.lng}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="form-field" style={{ marginBottom: '20px' }}>
        <label htmlFor="algorithm">Algoritmo</label>
        <select
          id="algorithm"
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value)}
          className="form-input"
        >
          <option value="astar">A* (Recomendado)</option>
          <option value="dijkstra">Dijkstra</option>
          <option value="tsp">TSP (Traveling Salesman)</option>
        </select>
      </div>

      <button 
        className="btn" 
        onClick={handleCalculateRoute} 
        disabled={points.length < 2 || loading}
      >
        {loading ? 'Calculando...' : 'Calcular Ruta Óptima'}
      </button>

      {routeResult && (
        <div className="stats-panel" style={{ marginTop: '30px' }}>
          <h3>Ruta Óptima</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{routeResult.distance}</div>
              <div className="stat-label">Distancia Total</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{routeResult.route.length - 1}</div>
              <div className="stat-label">Puntos Visitados</div>
            </div>
          </div>
          
          <div className="history-list" style={{ marginTop: '20px' }}>
            <h3>Ruta Completa</h3>
            {routeResult.route.map((point, index) => (
              <div key={index} className="history-item">
                <div className="history-item-header">
                  <span><strong>{index + 1}.</strong> {point}</span>
                </div>
              </div>
            ))}
          </div>

          {routeResult.steps && routeResult.steps.length > 0 && (
            <div style={{ marginTop: '30px' }}>
              <h3>Pasos del Algoritmo - Selección de Nodos</h3>
              <div className="history-list">
                {routeResult.steps.map((step, index) => (
                  <div key={index} className="history-item">
                    <div className="history-item-header">
                      <span><strong>Paso {step.step}:</strong> Desde {step.current}</span>
                    </div>
                    <div className="history-text" style={{ marginTop: '10px' }}>
                      <p><strong>Puntos evaluados:</strong> {step.evaluated.join(', ')}</p>
                      <p><strong>Seleccionado:</strong> {step.selected}</p>
                      <p><strong>Distancia:</strong> {step.distance.toFixed(2)}</p>
                      <p><strong>Heurística:</strong> {step.heuristic_value.toFixed(2)}</p>
                      <p><strong>Razón:</strong> {step.reason}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Explicación Técnica - Parte 2 */}
      <div className="message" style={{ marginTop: '30px', background: 'rgba(110, 139, 255, 0.1)', padding: '20px', borderRadius: '8px' }}>
        <h3 style={{ marginTop: 0 }}>Explicación Técnica - Parte 2</h3>
        <p><strong>Algoritmo:</strong> A* (A estrella)</p>
        <p><strong>Tipo:</strong> Búsqueda heurística</p>
        <p><strong>Justificación:</strong> A* combina el costo real del camino con una heurística estimada, encontrando la ruta óptima de manera eficiente.</p>
        <p><strong>Proceso:</strong></p>
        <ol>
          <li>Crear grafo con puntos de entrega</li>
          <li>Aplicar heurística (distancia euclidiana)</li>
          <li>Seleccionar nodo con menor costo estimado</li>
          <li>Repetir hasta visitar todos los nodos</li>
          <li>Retornar ruta óptima</li>
        </ol>
        <p><strong>Selección de Nodos:</strong> En cada paso, el algoritmo evalúa todos los puntos no visitados, calcula la distancia desde el punto actual (heurística), y selecciona el punto más cercano. Esto minimiza la distancia total del recorrido.</p>
      </div>
    </section>
  )
}

export default RouteOptimization


import { useState, useEffect } from 'react'
import { routeOptimizationAPI } from '../utils/api'

function RouteOptimization({ user }) {
  const [points, setPoints] = useState([])
  const [newPoint, setNewPoint] = useState({ name: '', address: '' })
  const [algorithm, setAlgorithm] = useState('astar')
  const [routeResult, setRouteResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [savedRoutes, setSavedRoutes] = useState([])
  const [loadingRoutes, setLoadingRoutes] = useState(false)
  const [saveRouteName, setSaveRouteName] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)

  const handleAddPoint = () => {
    if (!newPoint.name || !newPoint.address) {
      alert('Por favor completa todos los campos (nombre y dirección)')
      return
    }

    setPoints([...points, { name: newPoint.name, address: newPoint.address }])
    setNewPoint({ name: '', address: '' })
  }

  const handleRemovePoint = (index) => {
    setPoints(points.filter((_, i) => i !== index))
  }

  // Cargar rutas guardadas al montar el componente
  useEffect(() => {
    loadSavedRoutes()
  }, [])

  const loadSavedRoutes = async () => {
    setLoadingRoutes(true)
    try {
      const routes = await routeOptimizationAPI.getRoutes()
      setSavedRoutes(routes)
    } catch (err) {
      console.error('Error al cargar rutas:', err)
    } finally {
      setLoadingRoutes(false)
    }
  }

  const handleCalculateRoute = async () => {
    if (points.length < 2) {
      alert('Necesitas al menos 2 puntos para calcular una ruta')
      return
    }

    setLoading(true)
    setRouteResult(null)

    try {
      const result = await routeOptimizationAPI.optimize(points, algorithm, 0, false, null)
      setRouteResult(result)
      setShowSaveDialog(true)
    } catch (err) {
      console.error('Error al calcular ruta:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleSaveRoute = async () => {
    if (!saveRouteName.trim()) {
      alert('Por favor ingresa un nombre para la ruta')
      return
    }

    setLoading(true)
    try {
      const result = await routeOptimizationAPI.optimize(points, algorithm, 0, true, saveRouteName)
      setRouteResult(result)
      setShowSaveDialog(false)
      setSaveRouteName('')
      await loadSavedRoutes()
      alert('Ruta guardada correctamente')
    } catch (err) {
      console.error('Error al guardar ruta:', err)
      alert(`Error al guardar ruta: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleLoadRoute = async (route) => {
    try {
      setLoading(true)
      const loadedRoute = await routeOptimizationAPI.getRoute(route.id)
      
      // Ordenar puntos por el campo 'order'
      const sortedPoints = [...loadedRoute.points].sort((a, b) => a.order - b.order)
      
      // Cargar puntos desde la ruta guardada
      const routePoints = sortedPoints.map(p => ({
        name: p.name,
        address: p.address
      }))
      setPoints(routePoints)
      setAlgorithm(loadedRoute.algorithm)
      
      // Construir resultado desde la ruta guardada (sin recalcular)
      const routeResultData = {
        route: sortedPoints.map(p => p.name),
        distance: loadedRoute.distance,
        algorithm: loadedRoute.algorithm,
        is_direct_route: sortedPoints.length === 2,
        points_info: sortedPoints.map(p => ({
          name: p.name,
          address: p.address,
          display_name: p.display_name || p.address,
          lat: p.lat,
          lng: p.lng
        })),
        steps: []  // Los pasos no se guardan, pero se pueden recalcular si es necesario
      }
      
      setRouteResult(routeResultData)
      setShowSaveDialog(false)
    } catch (err) {
      console.error('Error al cargar ruta:', err)
      alert(`Error al cargar ruta: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteRoute = async (routeId) => {
    if (!confirm('¿Estás seguro de que deseas eliminar esta ruta?')) {
      return
    }

    try {
      await routeOptimizationAPI.deleteRoute(routeId)
      await loadSavedRoutes()
      alert('Ruta eliminada correctamente')
    } catch (err) {
      console.error('Error al eliminar ruta:', err)
      alert(`Error al eliminar ruta: ${err.message}`)
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
              placeholder="Ej: Almacén Central"
            />
          </div>
          <div className="form-field" style={{ flex: 2 }}>
            <label htmlFor="point-address">Dirección</label>
            <input
              type="text"
              id="point-address"
              value={newPoint.address}
              onChange={(e) => setNewPoint({ ...newPoint, address: e.target.value })}
              placeholder="Ej: Pintor Gustavo Cabello Olguin 944, Rancagua, Chile"
            />
            <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
              Ingresa la dirección completa (calle, ciudad, país). Puedes usar comas con o sin espacios. Ej: "Calle, Ciudad, País" o "Calle,Ciudad,País"
            </small>
          </div>
        </div>
        <button type="button" className="btn" onClick={handleAddPoint}>
          + Agregar Punto
        </button>
      </form>

      {points.length > 0 && (
        <div className="apis-list" style={{ marginBottom: '20px' }}>
          <h3>Puntos Agregados ({points.length})</h3>
          {points.length === 2 && (
            <div className="message" style={{ marginBottom: '15px', background: 'rgba(110, 139, 255, 0.1)', padding: '10px', borderRadius: '8px', fontSize: '0.9em' }}>
              <p><strong>Nota:</strong> Con 2 puntos se calculará la ruta directa desde el punto de inicio hasta el punto de destino.</p>
            </div>
          )}
          {points.length > 2 && (
            <div className="message" style={{ marginBottom: '15px', background: 'rgba(110, 139, 255, 0.1)', padding: '10px', borderRadius: '8px', fontSize: '0.9em' }}>
              <p><strong>Nota:</strong> Con {points.length} puntos se optimizará el orden de visita para minimizar la distancia total (incluyendo retorno al inicio).</p>
            </div>
          )}
          {points.map((point, index) => (
            <div key={index} className="api-card">
              <div className="api-header">
                <h3>
                  {index === 0 ? '🚩 ' : index === points.length - 1 && points.length === 2 ? '🏁 ' : ''}
                  {point.name}
                  {index === 0 && points.length === 2 && <span style={{ fontSize: '0.8em', color: 'var(--text-secondary)', marginLeft: '10px' }}>(Inicio)</span>}
                  {index === 1 && points.length === 2 && <span style={{ fontSize: '0.8em', color: 'var(--text-secondary)', marginLeft: '10px' }}>(Destino)</span>}
                </h3>
                <button
                  type="button"
                  className="btn btn--ghost btn--small"
                  onClick={() => handleRemovePoint(index)}
                >
                  Eliminar
                </button>
              </div>
              <div className="api-info">
                <p><strong>Dirección:</strong> {point.address}</p>
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
        {loading ? 'Calculando ruta...' : points.length === 2 ? 'Calcular Ruta Directa' : 'Calcular Ruta Óptima'}
      </button>

      {/* Rutas Guardadas */}
      {savedRoutes.length > 0 && (
        <div className="apis-list" style={{ marginBottom: '20px', marginTop: '20px' }}>
          <h3>Rutas Guardadas ({savedRoutes.length})</h3>
          {savedRoutes.map((route) => (
            <div key={route.id} className="api-card">
              <div className="api-header">
                <h3>{route.name}</h3>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button
                    type="button"
                    className="btn btn--ghost btn--small"
                    onClick={() => handleLoadRoute(route)}
                  >
                    Cargar
                  </button>
                  <button
                    type="button"
                    className="btn btn--ghost btn--small"
                    onClick={() => handleDeleteRoute(route.id)}
                    style={{ color: 'var(--error)' }}
                  >
                    Eliminar
                  </button>
                </div>
              </div>
              <div className="api-info">
                <p><strong>Algoritmo:</strong> {route.algorithm}</p>
                <p><strong>Distancia:</strong> {route.distance?.toFixed(2) || 'N/A'} unidades</p>
                <p><strong>Puntos:</strong> {route.points?.length || 0}</p>
                <p><strong>Fecha:</strong> {new Date(route.created_at).toLocaleDateString()}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Diálogo para guardar ruta */}
      {showSaveDialog && routeResult && (
        <div className="message" style={{ marginTop: '20px', background: 'rgba(110, 139, 255, 0.1)', padding: '20px', borderRadius: '8px' }}>
          <h3 style={{ marginTop: 0 }}>¿Guardar esta ruta?</h3>
          <p style={{ marginBottom: '15px', color: 'var(--text-secondary)' }}>
            La ruta se guardará en la base de datos y podrás acceder a ella en cualquier momento.
          </p>
          <div className="form-field" style={{ marginTop: '15px' }}>
            <label htmlFor="route-name">Nombre de la ruta</label>
            <input
              type="text"
              id="route-name"
              value={saveRouteName}
              onChange={(e) => setSaveRouteName(e.target.value)}
              placeholder="Ej: Ruta de entrega centro"
              className="form-input"
            />
          </div>
          <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
            <button className="btn" onClick={handleSaveRoute} disabled={loading || !saveRouteName.trim()}>
              {loading ? 'Guardando...' : 'Guardar Ruta'}
            </button>
            <button className="btn btn--ghost" onClick={() => setShowSaveDialog(false)} disabled={loading}>
              Cancelar
            </button>
          </div>
        </div>
      )}

      {routeResult && (
        <div className="stats-panel" style={{ marginTop: '30px' }}>
          <h3>Ruta Óptima</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">
                {typeof routeResult.distance === 'number' 
                  ? routeResult.distance.toFixed(2) 
                  : routeResult.distance}
              </div>
              <div className="stat-label">Distancia Total (unidades)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {routeResult.is_direct_route 
                  ? routeResult.route.length 
                  : routeResult.route.length - 1}
              </div>
              <div className="stat-label">
                {routeResult.is_direct_route ? 'Puntos en la Ruta' : 'Puntos Visitados'}
              </div>
            </div>
          </div>
          
          <div className="history-list" style={{ marginTop: '20px' }}>
            <h3>{routeResult.is_direct_route ? 'Ruta Directa' : 'Ruta Optimizada'}</h3>
            {routeResult.is_direct_route && (
              <div className="message" style={{ marginBottom: '15px', background: 'rgba(76, 175, 80, 0.1)', padding: '10px', borderRadius: '8px', fontSize: '0.9em' }}>
                <p>Ruta directa calculada desde el punto de inicio hasta el destino.</p>
              </div>
            )}
            {routeResult.route.map((pointName, index) => {
              const pointInfo = routeResult.points_info?.find(p => p.name === pointName)
              const isLast = index === routeResult.route.length - 1
              const isReturn = routeResult.is_direct_route ? false : isLast && routeResult.route[0] === pointName
              
              return (
                <div key={index} className="history-item">
                  <div className="history-item-header">
                    <span>
                      {index === 0 && routeResult.is_direct_route && '🚩 '}
                      {isLast && routeResult.is_direct_route && index > 0 && '🏁 '}
                      {isReturn && '🔄 '}
                      <strong>{index + 1}.</strong> {pointName}
                      {isReturn && <span style={{ fontSize: '0.8em', color: 'var(--text-secondary)', marginLeft: '10px' }}>(Retorno al inicio)</span>}
                    </span>
                  </div>
                  {pointInfo && (
                    <div className="history-text" style={{ marginTop: '5px', fontSize: '0.9em', color: 'var(--text-secondary)' }}>
                      <p>{pointInfo.display_name || pointInfo.address}</p>
                    </div>
                  )}
                </div>
              )
            })}
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
        <p><strong>Geocodificación:</strong> Utilizamos Nominatim (OpenStreetMap) para convertir direcciones en coordenadas geográficas. Es una API gratuita y no requiere clave de acceso.</p>
        <p><strong>Algoritmo:</strong> A* (A estrella)</p>
        <p><strong>Tipo:</strong> Búsqueda heurística</p>
        <p><strong>Justificación:</strong> A* combina el costo real del camino con una heurística estimada, encontrando la ruta óptima de manera eficiente.</p>
        <p><strong>Proceso:</strong></p>
        <ol>
          <li>Geocodificar direcciones a coordenadas (lat, lng) usando Nominatim</li>
          <li>Crear grafo con puntos de entrega</li>
          <li>Calcular distancias entre todos los puntos (distancia euclidiana)</li>
          <li>Aplicar heurística para seleccionar el siguiente nodo</li>
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


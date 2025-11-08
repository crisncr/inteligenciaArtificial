import { useState, useEffect, useRef } from 'react'
import { routeOptimizationAPI } from '../utils/api'

function RouteOptimization({ user }) {
  const [startAddress, setStartAddress] = useState('')
  const [endAddress, setEndAddress] = useState('')
  const [startSuggestions, setStartSuggestions] = useState([])
  const [endSuggestions, setEndSuggestions] = useState([])
  const [showStartSuggestions, setShowStartSuggestions] = useState(false)
  const [showEndSuggestions, setShowEndSuggestions] = useState(false)
  const [startSelected, setStartSelected] = useState(null)
  const [endSelected, setEndSelected] = useState(null)
  const [algorithm, setAlgorithm] = useState('astar')
  const [routeResult, setRouteResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [savedRoutes, setSavedRoutes] = useState([])
  const [loadingRoutes, setLoadingRoutes] = useState(false)
  const [saveRouteName, setSaveRouteName] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [loadingSuggestions, setLoadingSuggestions] = useState(false)
  const startInputRef = useRef(null)
  const endInputRef = useRef(null)
  const suggestionsRef = useRef(null)
  const startDebounceRef = useRef(null)
  const endDebounceRef = useRef(null)

  // Cargar rutas guardadas al montar el componente
  useEffect(() => {
    loadSavedRoutes()
  }, [])

  // Cerrar sugerencias al hacer click fuera
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(event.target)) {
        setShowStartSuggestions(false)
        setShowEndSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
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

  // Autocompletado para dirección de inicio con debounce
  const handleStartAddressChange = (value) => {
    setStartAddress(value)
    setStartSelected(null)
    
    // Limpiar timeout anterior
    if (startDebounceRef.current) {
      clearTimeout(startDebounceRef.current)
    }
    
    if (value.length < 3) {
      setStartSuggestions([])
      setShowStartSuggestions(false)
      return
    }
    
    // Debounce: esperar 300ms después de que el usuario deje de escribir
    startDebounceRef.current = setTimeout(async () => {
      setLoadingSuggestions(true)
      try {
        const suggestions = await routeOptimizationAPI.autocomplete(value)
        console.log('Sugerencias de inicio:', suggestions)
        setStartSuggestions(suggestions)
        if (suggestions.length > 0) {
          setShowStartSuggestions(true)
        }
      } catch (err) {
        console.error('Error en autocompletado:', err)
        setStartSuggestions([])
        setShowStartSuggestions(false)
      } finally {
        setLoadingSuggestions(false)
      }
    }, 300)
  }

  // Autocompletado para dirección de destino con debounce
  const handleEndAddressChange = (value) => {
    setEndAddress(value)
    setEndSelected(null)
    
    // Limpiar timeout anterior
    if (endDebounceRef.current) {
      clearTimeout(endDebounceRef.current)
    }
    
    if (value.length < 3) {
      setEndSuggestions([])
      setShowEndSuggestions(false)
      return
    }
    
    // Debounce: esperar 300ms después de que el usuario deje de escribir
    endDebounceRef.current = setTimeout(async () => {
      setLoadingSuggestions(true)
      try {
        const suggestions = await routeOptimizationAPI.autocomplete(value)
        console.log('Sugerencias de destino:', suggestions)
        setEndSuggestions(suggestions)
        if (suggestions.length > 0) {
          setShowEndSuggestions(true)
        }
      } catch (err) {
        console.error('Error en autocompletado:', err)
        setEndSuggestions([])
        setShowEndSuggestions(false)
      } finally {
        setLoadingSuggestions(false)
      }
    }, 300)
  }

  const handleSelectStart = (suggestion) => {
    setStartAddress(suggestion.display_name || suggestion.text)
    setStartSelected(suggestion)
    setShowStartSuggestions(false)
  }

  const handleSelectEnd = (suggestion) => {
    setEndAddress(suggestion.display_name || suggestion.text)
    setEndSelected(suggestion)
    setShowEndSuggestions(false)
  }

  const handleCalculateRoute = async () => {
    if (!startAddress.trim() || !endAddress.trim()) {
      alert('Por favor ingresa direcciones de inicio y destino')
      return
    }

    setLoading(true)
    setRouteResult(null)
    setShowStartSuggestions(false)
    setShowEndSuggestions(false)

    try {
      const points = [
        { name: 'Punto de Inicio', address: startAddress.trim() },
        { name: 'Punto de Destino', address: endAddress.trim() }
      ]
      
      console.log('Calculando ruta con puntos:', points)
      const result = await routeOptimizationAPI.optimize(points, algorithm, 0, false, null)
      console.log('Resultado de la ruta:', result)
      setRouteResult(result)
      setShowSaveDialog(true)
    } catch (err) {
      console.error('Error al calcular ruta:', err)
      alert(`Error: ${err.message || 'No se pudo calcular la ruta. Verifica que las direcciones sean válidas.'}`)
    } finally {
      setLoading(false)
    }
  }

  const handleSaveRoute = async () => {
    if (!saveRouteName.trim()) {
      alert('Por favor ingresa un nombre para la ruta')
      return
    }

    if (!startSelected || !endSelected) {
      alert('Por favor selecciona direcciones válidas')
      return
    }

    setLoading(true)
    try {
      const points = [
        { name: 'Punto de Inicio', address: startAddress },
        { name: 'Punto de Destino', address: endAddress }
      ]
      
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
      
      if (sortedPoints.length >= 2) {
        setStartAddress(sortedPoints[0].address)
        setEndAddress(sortedPoints[1].address)
        setStartSelected({
          display_name: sortedPoints[0].display_name || sortedPoints[0].address,
          lat: sortedPoints[0].lat,
          lng: sortedPoints[0].lng
        })
        setEndSelected({
          display_name: sortedPoints[1].display_name || sortedPoints[1].address,
          lat: sortedPoints[1].lat,
          lng: sortedPoints[1].lng
        })
        setAlgorithm(loadedRoute.algorithm)
      }
      
      // Construir resultado desde la ruta guardada
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
        steps: []
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

      {/* Rutas Guardadas */}
      {savedRoutes.length > 0 && (
        <div className="apis-list" style={{ marginBottom: '20px' }}>
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

      {/* Formulario de Rutas - Solo 2 campos: Inicio y Destino */}
      <div className="api-form" style={{ marginBottom: '20px', position: 'relative' }} ref={suggestionsRef}>
        <div className="form-field" style={{ marginBottom: '20px', position: 'relative' }}>
          <label htmlFor="start-address">
            <span style={{ fontSize: '1.2em', marginRight: '10px' }}>🚩</span>
            Punto de Inicio
          </label>
          <input
            ref={startInputRef}
            type="text"
            id="start-address"
            value={startAddress}
            onChange={(e) => handleStartAddressChange(e.target.value)}
            onFocus={() => {
              if (startAddress.length >= 3 && startSuggestions.length > 0) {
                setShowStartSuggestions(true)
              }
            }}
            placeholder="Ej: Pintor Gustavo Cabello Olguin 944, Rancagua, Chile"
            className="form-input"
            style={{ width: '100%', padding: '12px' }}
          />
          {showStartSuggestions && startSuggestions.length > 0 && (
            <div style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              background: '#fff',
              border: '1px solid #ddd',
              borderRadius: '8px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              zIndex: 1000,
              maxHeight: '300px',
              overflowY: 'auto',
              marginTop: '5px'
            }}>
              {startSuggestions.map((suggestion, index) => (
                <div
                  key={index}
                  onClick={() => handleSelectStart(suggestion)}
                  style={{
                    padding: '12px',
                    cursor: 'pointer',
                    borderBottom: index < startSuggestions.length - 1 ? '1px solid #eee' : 'none',
                    backgroundColor: 'transparent',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#f5f5f5'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent'
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: '4px', color: '#333' }}>
                    {suggestion.display_name || suggestion.text || 'Dirección no disponible'}
                  </div>
                  {(suggestion.address_line2 || suggestion.city) && (
                    <div style={{ fontSize: '0.9em', color: '#666' }}>
                      {suggestion.address_line2 || suggestion.city}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {showStartSuggestions && startSuggestions.length === 0 && startAddress.length >= 3 && !loadingSuggestions && (
            <div style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              background: '#fff',
              border: '1px solid #ddd',
              borderRadius: '8px',
              padding: '12px',
              marginTop: '5px',
              color: '#666',
              zIndex: 1000
            }}>
              No se encontraron sugerencias
            </div>
          )}
        </div>

        <div className="form-field" style={{ marginBottom: '20px', position: 'relative' }}>
          <label htmlFor="end-address">
            <span style={{ fontSize: '1.2em', marginRight: '10px' }}>🏁</span>
            Punto de Destino
          </label>
          <input
            ref={endInputRef}
            type="text"
            id="end-address"
            value={endAddress}
            onChange={(e) => handleEndAddressChange(e.target.value)}
            onFocus={() => {
              if (endAddress.length >= 3 && endSuggestions.length > 0) {
                setShowEndSuggestions(true)
              }
            }}
            placeholder="Ej: Av. Libertador Bernardo O'Higgins 123, Santiago, Chile"
            className="form-input"
            style={{ width: '100%', padding: '12px' }}
          />
          {showEndSuggestions && endSuggestions.length > 0 && (
            <div style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              background: '#fff',
              border: '1px solid #ddd',
              borderRadius: '8px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              zIndex: 1000,
              maxHeight: '300px',
              overflowY: 'auto',
              marginTop: '5px'
            }}>
              {endSuggestions.map((suggestion, index) => (
                <div
                  key={index}
                  onClick={() => handleSelectEnd(suggestion)}
                  style={{
                    padding: '12px',
                    cursor: 'pointer',
                    borderBottom: index < endSuggestions.length - 1 ? '1px solid #eee' : 'none',
                    backgroundColor: 'transparent',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#f5f5f5'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent'
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: '4px', color: '#333' }}>
                    {suggestion.display_name || suggestion.text || 'Dirección no disponible'}
                  </div>
                  {(suggestion.address_line2 || suggestion.city) && (
                    <div style={{ fontSize: '0.9em', color: '#666' }}>
                      {suggestion.address_line2 || suggestion.city}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {showEndSuggestions && endSuggestions.length === 0 && endAddress.length >= 3 && !loadingSuggestions && (
            <div style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              background: '#fff',
              border: '1px solid #ddd',
              borderRadius: '8px',
              padding: '12px',
              marginTop: '5px',
              color: '#666',
              zIndex: 1000
            }}>
              No se encontraron sugerencias
            </div>
          )}
        </div>

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
          disabled={!startAddress.trim() || !endAddress.trim() || loading}
          style={{ width: '100%' }}
        >
          {loading ? 'Calculando ruta...' : 'Calcular Ruta Directa'}
        </button>
        
        {loadingSuggestions && (
          <div style={{ marginTop: '10px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            Buscando sugerencias...
          </div>
        )}
      </div>

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
        <p><strong>Geocodificación:</strong> Utilizamos Google Maps Geocoding API y Places API para convertir direcciones en coordenadas geográficas. Incluye autocompletado en tiempo real mientras escribes.</p>
        <p><strong>Algoritmo:</strong> A* (A estrella)</p>
        <p><strong>Tipo:</strong> Búsqueda heurística</p>
        <p><strong>Justificación:</strong> A* combina el costo real del camino con una heurística estimada, encontrando la ruta óptima de manera eficiente.</p>
        <p><strong>Proceso:</strong></p>
        <ol>
          <li>Autocompletar direcciones mientras el usuario escribe (Geoapify)</li>
          <li>Geocodificar direcciones a coordenadas (lat, lng) usando Geoapify</li>
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

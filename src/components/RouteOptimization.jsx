import { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import { routeOptimizationAPI } from '../utils/api'
import 'leaflet/dist/leaflet.css'

// Fix para iconos de Leaflet en React
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

// Componente para actualizar el centro del mapa
function MapUpdater({ center, zoom }) {
  const map = useMap()
  useEffect(() => {
    map.setView(center, zoom)
  }, [map, center, zoom])
  return null
}

// Componente para escuchar clicks en el mapa
function MapClickHandler({ onMapClick, selectingPoint }) {
  useMapEvents({
    click: (e) => {
      if (selectingPoint) {
        onMapClick(e.latlng.lat, e.latlng.lng)
      }
    },
  })
  return null
}

function RouteOptimization({ user }) {
  const [startAddress, setStartAddress] = useState('')
  const [endAddress, setEndAddress] = useState('')
  const [startPlace, setStartPlace] = useState(null)
  const [endPlace, setEndPlace] = useState(null)
  const [algorithm, setAlgorithm] = useState('astar')
  const [routeResult, setRouteResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [savedRoutes, setSavedRoutes] = useState([])
  const [loadingRoutes, setLoadingRoutes] = useState(false)
  const [saveRouteName, setSaveRouteName] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [selectingPoint, setSelectingPoint] = useState(null) // 'start' o 'end' o null
  const [startSuggestions, setStartSuggestions] = useState([])
  const [endSuggestions, setEndSuggestions] = useState([])
  const [showStartSuggestions, setShowStartSuggestions] = useState(false)
  const [showEndSuggestions, setShowEndSuggestions] = useState(false)
  const [loadingSuggestions, setLoadingSuggestions] = useState(false)
  const [mapCenter, setMapCenter] = useState([-33.4489, -70.6693]) // Santiago, Chile
  const [mapZoom, setMapZoom] = useState(10)
  
  const startInputRef = useRef(null)
  const endInputRef = useRef(null)
  const startDebounceRef = useRef(null)
  const endDebounceRef = useRef(null)

  // Cargar rutas guardadas al montar
  useEffect(() => {
    loadSavedRoutes()
  }, [])

  // Manejar clicks fuera de las sugerencias
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (startInputRef.current && !startInputRef.current.contains(event.target)) {
        setShowStartSuggestions(false)
      }
      if (endInputRef.current && !endInputRef.current.contains(event.target)) {
        setShowEndSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  // Ajustar vista del mapa cuando hay marcadores (solo actualizar si cambian los lugares)
  useEffect(() => {
    if (startPlace && endPlace) {
      // Centrar mapa para mostrar ambos marcadores
      const centerLat = (startPlace.lat + endPlace.lat) / 2
      const centerLng = (startPlace.lng + endPlace.lng) / 2
      setMapCenter([centerLat, centerLng])
      setMapZoom(12)
    } else if (startPlace) {
      setMapCenter([startPlace.lat, startPlace.lng])
      setMapZoom(15)
    } else if (endPlace) {
      setMapCenter([endPlace.lat, endPlace.lng])
      setMapZoom(15)
    }
  }, [startPlace?.lat, startPlace?.lng, endPlace?.lat, endPlace?.lng])

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

  // Autocompletado para dirección de inicio - Actualizado 2025
  const handleStartAddressChange = (value) => {
    // Limpiar resultado si cambia la dirección
    if (routeResult) {
      setRouteResult(null)
    }
    
    setStartAddress(value)
    setStartPlace(null)
    
    if (startDebounceRef.current) {
      clearTimeout(startDebounceRef.current)
    }
    
    // Mostrar sugerencias desde 2 caracteres (en lugar de 3)
    if (value.length < 2) {
      setStartSuggestions([])
      setShowStartSuggestions(false)
      return
    }
    
    // Mostrar loading inmediatamente
    setLoadingSuggestions(true)
    
    // Reducir debounce para respuesta más rápida (200ms en lugar de 300ms)
    startDebounceRef.current = setTimeout(async () => {
      try {
        const suggestions = await routeOptimizationAPI.autocomplete(value)
        setStartSuggestions(suggestions)
        setShowStartSuggestions(suggestions.length > 0)
      } catch (err) {
        console.error('Error en autocompletado:', err)
        setStartSuggestions([])
        setShowStartSuggestions(false)
      } finally {
        setLoadingSuggestions(false)
      }
    }, 200)
  }

  // Autocompletado para dirección de destino - Actualizado 2025
  const handleEndAddressChange = (value) => {
    // Limpiar resultado si cambia la dirección
    if (routeResult) {
      setRouteResult(null)
    }
    
    setEndAddress(value)
    setEndPlace(null)
    
    if (endDebounceRef.current) {
      clearTimeout(endDebounceRef.current)
    }
    
    // Mostrar sugerencias desde 2 caracteres (en lugar de 3)
    if (value.length < 2) {
      setEndSuggestions([])
      setShowEndSuggestions(false)
      return
    }
    
    // Mostrar loading inmediatamente
    setLoadingSuggestions(true)
    
    // Reducir debounce para respuesta más rápida (200ms en lugar de 300ms)
    endDebounceRef.current = setTimeout(async () => {
      try {
        const suggestions = await routeOptimizationAPI.autocomplete(value)
        setEndSuggestions(suggestions)
        setShowEndSuggestions(suggestions.length > 0)
      } catch (err) {
        console.error('Error en autocompletado:', err)
        setEndSuggestions([])
        setShowEndSuggestions(false)
      } finally {
        setLoadingSuggestions(false)
      }
    }, 200)
  }

  const handleSelectStart = (suggestion) => {
    // Limpiar resultado si cambia la selección
    if (routeResult) {
      setRouteResult(null)
    }
    
    setStartAddress(suggestion.display_name || suggestion.text)
    setStartPlace({
      address: suggestion.display_name || suggestion.text,
      lat: suggestion.lat,
      lng: suggestion.lng
    })
    setShowStartSuggestions(false)
  }

  const handleSelectEnd = (suggestion) => {
    // Limpiar resultado si cambia la selección
    if (routeResult) {
      setRouteResult(null)
    }
    
    setEndAddress(suggestion.display_name || suggestion.text)
    setEndPlace({
      address: suggestion.display_name || suggestion.text,
      lat: suggestion.lat,
      lng: suggestion.lng
    })
    setShowEndSuggestions(false)
  }

  // Geocodificar reversa cuando se hace click en el mapa
  const handleMapClick = async (lat, lng) => {
    // Limpiar resultado si cambia la selección
    if (routeResult) {
      setRouteResult(null)
    }
    
    try {
      setLoading(true)
      // Usar Nominatim para geocodificación reversa
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&accept-language=es`,
        {
          headers: {
            'User-Agent': 'RouteOptimizationApp/1.0'
          }
        }
      )
      
      if (response.ok) {
        const data = await response.json()
        const address = data.display_name || `${lat}, ${lng}`
        
        if (selectingPoint === 'start') {
          setStartAddress(address)
          setStartPlace({ address, lat, lng })
          setSelectingPoint(null)
        } else if (selectingPoint === 'end') {
          setEndAddress(address)
          setEndPlace({ address, lat, lng })
          setSelectingPoint(null)
        }
      }
    } catch (err) {
      console.error('Error en geocodificación reversa:', err)
      // Si falla, usar coordenadas directamente
      const address = `${lat.toFixed(6)}, ${lng.toFixed(6)}`
      if (selectingPoint === 'start') {
        setStartAddress(address)
        setStartPlace({ address, lat, lng })
        setSelectingPoint(null)
      } else if (selectingPoint === 'end') {
        setEndAddress(address)
        setEndPlace({ address, lat, lng })
        setSelectingPoint(null)
      }
    } finally {
      setLoading(false)
    }
  }

  const handleCalculateRoute = async () => {
    if (!startAddress.trim() || !endAddress.trim()) {
      alert('Por favor ingresa direcciones de inicio y destino')
      return
    }

    setLoading(true)
    setRouteResult(null)
    setShowSaveDialog(false)
    setShowStartSuggestions(false)
    setShowEndSuggestions(false)

    try {
      const points = startPlace && endPlace
        ? [
            { name: 'Punto de Inicio', address: startAddress, lat: startPlace.lat, lng: startPlace.lng },
            { name: 'Punto de Destino', address: endAddress, lat: endPlace.lat, lng: endPlace.lng }
          ]
        : [
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

    if (!startAddress.trim() || !endAddress.trim()) {
      alert('Por favor ingresa direcciones válidas')
      return
    }

    setLoading(true)
    try {
      const points = startPlace && endPlace
        ? [
            { name: 'Punto de Inicio', address: startAddress, lat: startPlace.lat, lng: startPlace.lng },
            { name: 'Punto de Destino', address: endAddress, lat: endPlace.lat, lng: endPlace.lng }
          ]
        : [
            { name: 'Punto de Inicio', address: startAddress.trim() },
            { name: 'Punto de Destino', address: endAddress.trim() }
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
      
      const sortedPoints = [...loadedRoute.points].sort((a, b) => a.order - b.order)
      
      if (sortedPoints.length >= 2) {
        setStartAddress(sortedPoints[0].address)
        setEndAddress(sortedPoints[1].address)
        setStartPlace({
          address: sortedPoints[0].address,
          lat: sortedPoints[0].lat,
          lng: sortedPoints[0].lng
        })
        setEndPlace({
          address: sortedPoints[1].address,
          lat: sortedPoints[1].lat,
          lng: sortedPoints[1].lng
        })
        setAlgorithm(loadedRoute.algorithm)
      }
      
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

  // Iconos personalizados para marcadores usando divIcon
  const startIcon = L.divIcon({
    className: 'custom-marker-icon',
    html: '<div style="background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); width: 40px; height: 40px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg); border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center;"><span style="transform: rotate(45deg); color: white; font-size: 20px;">🚩</span></div>',
    iconSize: [40, 40],
    iconAnchor: [20, 40],
    popupAnchor: [0, -40]
  })

  const endIcon = L.divIcon({
    className: 'custom-marker-icon',
    html: '<div style="background: linear-gradient(135deg, #44ff44 0%, #00cc00 100%); width: 40px; height: 40px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg); border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center;"><span style="transform: rotate(45deg); color: white; font-size: 20px;">🏁</span></div>',
    iconSize: [40, 40],
    iconAnchor: [20, 40],
    popupAnchor: [0, -40]
  })

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

      {/* Formulario de Rutas con Mapa Leaflet */}
      <div className="api-form" style={{ marginBottom: '20px' }}>
        <div className="form-field" style={{ marginBottom: '20px' }}>
          <label htmlFor="start-address">
            <span style={{ fontSize: '1.2em', marginRight: '10px' }}>🚩</span>
            Punto de Inicio
          </label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <div style={{ flex: 1, position: 'relative' }} ref={startInputRef}>
              <input
                type="text"
                id="start-address"
                value={startAddress}
                onChange={(e) => handleStartAddressChange(e.target.value)}
                onFocus={() => {
                  if (startAddress.length >= 2 && startSuggestions.length > 0) {
                    setShowStartSuggestions(true)
                  }
                }}
                placeholder="Escribe una dirección o selecciona del mapa"
                className="form-input"
                style={{ width: '100%', padding: '12px' }}
              />
              {showStartSuggestions && startSuggestions.length > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  background: 'var(--panel)',
                  border: '2px solid var(--primary)',
                  borderRadius: '12px',
                  boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
                  zIndex: 1000,
                  maxHeight: '300px',
                  overflowY: 'auto',
                  marginTop: '8px',
                  padding: '8px 0'
                }}>
                  {startSuggestions.map((suggestion, index) => (
                    <div
                      key={index}
                      onClick={() => handleSelectStart(suggestion)}
                      style={{
                        padding: '12px 16px',
                        cursor: 'pointer',
                        borderBottom: index < startSuggestions.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none',
                        backgroundColor: 'transparent',
                        transition: 'all 0.2s',
                        color: 'var(--text)'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = 'rgba(110, 139, 255, 0.15)'
                        e.currentTarget.style.transform = 'translateX(4px)'
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = 'transparent'
                        e.currentTarget.style.transform = 'translateX(0)'
                      }}
                    >
                      <div style={{ 
                        fontWeight: '600', 
                        marginBottom: '4px',
                        fontSize: '0.95em',
                        color: 'var(--text)'
                      }}>
                        📍 {suggestion.display_name || suggestion.text || 'Dirección no disponible'}
                      </div>
                      {(suggestion.address_line2 || suggestion.city) && (
                        <div style={{ 
                          fontSize: '0.85em', 
                          color: 'var(--text-secondary)',
                          marginLeft: '20px'
                        }}>
                          {suggestion.address_line2 || suggestion.city}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={() => setSelectingPoint(selectingPoint === 'start' ? null : 'start')}
              style={{ 
                whiteSpace: 'nowrap',
                backgroundColor: selectingPoint === 'start' ? 'var(--primary)' : 'transparent',
                color: selectingPoint === 'start' ? 'white' : 'var(--primary)'
              }}
            >
              {selectingPoint === 'start' ? '✓ Seleccionando...' : '📍 Seleccionar en mapa'}
            </button>
          </div>
          {selectingPoint === 'start' && (
            <small style={{ color: 'var(--primary)', display: 'block', marginTop: '5px', fontWeight: 'bold' }}>
              👆 Haz click en el mapa para seleccionar el punto de inicio
            </small>
          )}
        </div>

        <div className="form-field" style={{ marginBottom: '20px' }}>
          <label htmlFor="end-address">
            <span style={{ fontSize: '1.2em', marginRight: '10px' }}>🏁</span>
            Punto de Destino
          </label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <div style={{ flex: 1, position: 'relative' }} ref={endInputRef}>
              <input
                type="text"
                id="end-address"
                value={endAddress}
                onChange={(e) => handleEndAddressChange(e.target.value)}
                onFocus={() => {
                  if (endAddress.length >= 2 && endSuggestions.length > 0) {
                    setShowEndSuggestions(true)
                  }
                }}
                placeholder="Escribe una dirección o selecciona del mapa"
                className="form-input"
                style={{ width: '100%', padding: '12px' }}
              />
              {showEndSuggestions && endSuggestions.length > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  background: 'var(--panel)',
                  border: '2px solid var(--primary)',
                  borderRadius: '12px',
                  boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
                  zIndex: 1000,
                  maxHeight: '300px',
                  overflowY: 'auto',
                  marginTop: '8px',
                  padding: '8px 0'
                }}>
                  {endSuggestions.map((suggestion, index) => (
                    <div
                      key={index}
                      onClick={() => handleSelectEnd(suggestion)}
                      style={{
                        padding: '12px 16px',
                        cursor: 'pointer',
                        borderBottom: index < endSuggestions.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none',
                        backgroundColor: 'transparent',
                        transition: 'all 0.2s',
                        color: 'var(--text)'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = 'rgba(110, 139, 255, 0.15)'
                        e.currentTarget.style.transform = 'translateX(4px)'
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = 'transparent'
                        e.currentTarget.style.transform = 'translateX(0)'
                      }}
                    >
                      <div style={{ 
                        fontWeight: '600', 
                        marginBottom: '4px',
                        fontSize: '0.95em',
                        color: 'var(--text)'
                      }}>
                        📍 {suggestion.display_name || suggestion.text || 'Dirección no disponible'}
                      </div>
                      {(suggestion.address_line2 || suggestion.city) && (
                        <div style={{ 
                          fontSize: '0.85em', 
                          color: 'var(--text-secondary)',
                          marginLeft: '20px'
                        }}>
                          {suggestion.address_line2 || suggestion.city}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={() => setSelectingPoint(selectingPoint === 'end' ? null : 'end')}
              style={{ 
                whiteSpace: 'nowrap',
                backgroundColor: selectingPoint === 'end' ? 'var(--primary)' : 'transparent',
                color: selectingPoint === 'end' ? 'white' : 'var(--primary)'
              }}
            >
              {selectingPoint === 'end' ? '✓ Seleccionando...' : '📍 Seleccionar en mapa'}
            </button>
          </div>
          {selectingPoint === 'end' && (
            <small style={{ color: 'var(--primary)', display: 'block', marginTop: '5px', fontWeight: 'bold' }}>
              👆 Haz click en el mapa para seleccionar el punto de destino
            </small>
          )}
        </div>

        {/* Mapa de Leaflet */}
        <div className="form-field" style={{ marginBottom: '20px' }}>
          <label>Mapa Interactivo 2025 (CartoDB Voyager)</label>
          <div style={{
            width: '100%',
            height: '400px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            marginTop: '10px',
            zIndex: 0
          }}>
            <MapContainer
              center={mapCenter}
              zoom={mapZoom}
              style={{ height: '100%', width: '100%', borderRadius: '8px' }}
              scrollWheelZoom={true}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
                maxZoom={19}
                minZoom={3}
                updateWhenZooming={false}
                updateWhenIdle={true}
                keepBuffer={2}
              />
              <MapUpdater center={mapCenter} zoom={mapZoom} />
              <MapClickHandler onMapClick={handleMapClick} selectingPoint={selectingPoint} />
              {startPlace && (
                <Marker position={[startPlace.lat, startPlace.lng]} icon={startIcon}>
                  <Popup>🚩 Punto de Inicio<br />{startPlace.address}</Popup>
                </Marker>
              )}
              {endPlace && (
                <Marker position={[endPlace.lat, endPlace.lng]} icon={endIcon}>
                  <Popup>🏁 Punto de Destino<br />{endPlace.address}</Popup>
                </Marker>
              )}
            </MapContainer>
          </div>
          <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
            Escribe una dirección en los campos de arriba o haz click en el mapa para seleccionar puntos. Mapas actualizados 2025 - CartoDB Voyager.
          </small>
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

      {routeResult && routeResult.route && routeResult.route.length > 0 && (
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
        <p><strong>Geocodificación:</strong> Utilizamos Nominatim (OpenStreetMap) para convertir direcciones en coordenadas geográficas. Incluye autocompletado en tiempo real mientras escribes y geocodificación reversa al hacer click en el mapa.</p>
        <p><strong>Algoritmo:</strong> A* (A estrella)</p>
        <p><strong>Tipo:</strong> Búsqueda heurística</p>
        <p><strong>Justificación:</strong> A* combina el costo real del camino con una heurística estimada, encontrando la ruta óptima de manera eficiente.</p>
        <p><strong>Proceso:</strong></p>
        <ol>
          <li>Autocompletado de direcciones con Nominatim (OpenStreetMap) - Gratuito y sin API key</li>
          <li>Geocodificar direcciones a coordenadas (lat, lng) usando Nominatim</li>
          <li>Crear grafo con puntos de entrega</li>
          <li>Calcular distancias entre todos los puntos (distancia euclidiana)</li>
          <li>Aplicar heurística para seleccionar el siguiente nodo</li>
          <li>Seleccionar nodo con menor costo estimado</li>
          <li>Repetir hasta visitar todos los nodos</li>
          <li>Retornar ruta óptima</li>
        </ol>
        <p><strong>Selección de Nodos:</strong> En cada paso, el algoritmo evalúa todos los puntos no visitados, calcula la distancia desde el punto actual (heurística), y selecciona el punto más cercano. Esto minimiza la distancia total del recorrido.</p>
        <p><strong>Ventajas de OpenStreetMap:</strong> Completamente gratuito, sin límites de API key, funciona inmediatamente sin configuración, y proporciona datos de mapas de alta calidad.</p>
      </div>
    </section>
  )
}

export default RouteOptimization

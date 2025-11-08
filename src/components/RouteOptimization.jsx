import { useState, useEffect, useRef } from 'react'
import { routeOptimizationAPI } from '../utils/api'

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
  const [googleMapsApiKey, setGoogleMapsApiKey] = useState(null)
  const [mapsLoaded, setMapsLoaded] = useState(false)
  const [map, setMap] = useState(null)
  const [startMarker, setStartMarker] = useState(null)
  const [endMarker, setEndMarker] = useState(null)
  const startAutocompleteRef = useRef(null)
  const endAutocompleteRef = useRef(null)
  const startInputRef = useRef(null)
  const endInputRef = useRef(null)
  const mapRef = useRef(null)
  const startAutocompleteInstanceRef = useRef(null)
  const endAutocompleteInstanceRef = useRef(null)
  const [selectingPoint, setSelectingPoint] = useState(null) // 'start' o 'end' o null

  // Cargar API key de Google Maps y rutas guardadas
  useEffect(() => {
    loadGoogleMapsConfig()
    loadSavedRoutes()
  }, [])

  // Cargar Google Maps cuando la API key esté disponible
  useEffect(() => {
    if (googleMapsApiKey && !mapsLoaded) {
      loadGoogleMapsScript()
    }
  }, [googleMapsApiKey, mapsLoaded])

  // Inicializar autocompletado cuando Google Maps esté cargado
  useEffect(() => {
    if (mapsLoaded && window.google && startInputRef.current && endInputRef.current) {
      initializeAutocomplete()
    }
  }, [mapsLoaded, startInputRef.current, endInputRef.current])

  // Inicializar mapa cuando Google Maps esté cargado y el ref esté disponible
  useEffect(() => {
    if (mapsLoaded && window.google && mapRef.current && !map) {
      // Pequeño delay para asegurar que el DOM esté listo
      setTimeout(() => {
        initializeMap()
      }, 100)
    }
  }, [mapsLoaded, mapRef.current])

  // Actualizar marcadores cuando cambian los lugares
  useEffect(() => {
    if (map && startPlace && mapsLoaded) {
      updateStartMarker()
    }
  }, [map, startPlace, mapsLoaded])

  useEffect(() => {
    if (map && endPlace && mapsLoaded) {
      updateEndMarker()
    }
  }, [map, endPlace, mapsLoaded])

  const loadGoogleMapsConfig = async () => {
    try {
      const response = await fetch('/api/config/google-maps')
      const config = await response.json()
      if (config.hasApiKey && config.apiKey) {
        setGoogleMapsApiKey(config.apiKey)
      } else {
        console.warn('Google Maps API key no configurada')
      }
    } catch (err) {
      console.error('Error al cargar configuración de Google Maps:', err)
    }
  }

  const loadGoogleMapsScript = () => {
    if (window.google && window.google.maps && window.google.maps.places) {
      setMapsLoaded(true)
      return
    }

    const script = document.createElement('script')
    script.src = `https://maps.googleapis.com/maps/api/js?key=${googleMapsApiKey}&libraries=places&language=es&region=cl`
    script.async = true
    script.defer = true
    script.onload = () => {
      setMapsLoaded(true)
    }
    script.onerror = () => {
      console.error('Error al cargar Google Maps API')
    }
    document.head.appendChild(script)
  }

  const initializeAutocomplete = () => {
    if (!window.google || !window.google.maps || !window.google.maps.places) {
      return
    }

    // Limpiar instancias anteriores
    if (startAutocompleteInstanceRef.current) {
      window.google.maps.event.clearInstanceListeners(startAutocompleteInstanceRef.current)
    }
    if (endAutocompleteInstanceRef.current) {
      window.google.maps.event.clearInstanceListeners(endAutocompleteInstanceRef.current)
    }

    // Inicializar autocompletado para inicio
    if (startInputRef.current) {
      const startAutocomplete = new window.google.maps.places.Autocomplete(
        startInputRef.current,
        {
          componentRestrictions: { country: 'cl' },
          fields: ['geometry', 'formatted_address', 'address_components', 'place_id'],
          types: ['address']
        }
      )

      startAutocomplete.addListener('place_changed', () => {
        const place = startAutocomplete.getPlace()
        if (place.geometry) {
          setStartPlace({
            address: place.formatted_address,
            lat: place.geometry.location.lat(),
            lng: place.geometry.location.lng(),
            place_id: place.place_id
          })
          setStartAddress(place.formatted_address)
        }
      })

      startAutocomplete.addListener('place_changed', () => {
        const place = startAutocomplete.getPlace()
        if (place.geometry) {
          const location = {
            address: place.formatted_address,
            lat: place.geometry.location.lat(),
            lng: place.geometry.location.lng(),
            place_id: place.place_id
          }
          setStartPlace(location)
          setStartAddress(place.formatted_address)
        }
      })

      startAutocompleteInstanceRef.current = startAutocomplete
    }

    // Inicializar autocompletado para destino
    if (endInputRef.current) {
      const endAutocomplete = new window.google.maps.places.Autocomplete(
        endInputRef.current,
        {
          componentRestrictions: { country: 'cl' },
          fields: ['geometry', 'formatted_address', 'address_components', 'place_id'],
          types: ['address']
        }
      )

      endAutocomplete.addListener('place_changed', () => {
        const place = endAutocomplete.getPlace()
        if (place.geometry) {
          const location = {
            address: place.formatted_address,
            lat: place.geometry.location.lat(),
            lng: place.geometry.location.lng(),
            place_id: place.place_id
          }
          setEndPlace(location)
          setEndAddress(place.formatted_address)
        }
      })

      endAutocompleteInstanceRef.current = endAutocomplete
    }
  }

  const initializeMap = () => {
    if (!window.google || !window.google.maps || !mapRef.current) {
      return
    }

    // Crear mapa centrado en Chile
    const googleMap = new window.google.maps.Map(mapRef.current, {
      center: { lat: -33.4489, lng: -70.6693 }, // Santiago, Chile
      zoom: 10,
      mapTypeControl: true,
      streetViewControl: true,
      fullscreenControl: true
    })

    setMap(googleMap)

    // Agregar listener para clicks en el mapa
    googleMap.addListener('click', (event) => {
      if (selectingPoint) {
        const lat = event.latLng.lat()
        const lng = event.latLng.lng()
        
        // Geocodificar reversa para obtener la dirección
        const geocoder = new window.google.maps.Geocoder()
        geocoder.geocode({ location: { lat, lng } }, (results, status) => {
          if (status === 'OK' && results[0]) {
            const address = results[0].formatted_address
            
            if (selectingPoint === 'start') {
              setStartPlace({ address, lat, lng })
              setStartAddress(address)
            } else if (selectingPoint === 'end') {
              setEndPlace({ address, lat, lng })
              setEndAddress(address)
            }
            
            setSelectingPoint(null)
          }
        })
      }
    })
  }

  const updateStartMarker = () => {
    if (!map || !startPlace || !window.google) return

    // Eliminar marcador anterior si existe
    if (startMarker) {
      startMarker.setMap(null)
    }

    // Crear nuevo marcador
    const marker = new window.google.maps.Marker({
      position: { lat: startPlace.lat, lng: startPlace.lng },
      map: map,
      title: 'Punto de Inicio',
      label: {
        text: '🚩',
        fontSize: '20px',
        fontWeight: 'bold'
      },
      animation: window.google.maps.Animation.DROP
    })

    setStartMarker(marker)
    
    // Ajustar vista del mapa
    if (endMarker && endPlace) {
      const bounds = new window.google.maps.LatLngBounds()
      bounds.extend(new window.google.maps.LatLng(startPlace.lat, startPlace.lng))
      bounds.extend(new window.google.maps.LatLng(endPlace.lat, endPlace.lng))
      map.fitBounds(bounds)
    } else {
      map.setCenter({ lat: startPlace.lat, lng: startPlace.lng })
      if (map.getZoom() < 15) {
        map.setZoom(15)
      }
    }
  }

  const updateEndMarker = () => {
    if (!map || !endPlace || !window.google) return

    // Eliminar marcador anterior si existe
    if (endMarker) {
      endMarker.setMap(null)
    }

    // Crear nuevo marcador
    const marker = new window.google.maps.Marker({
      position: { lat: endPlace.lat, lng: endPlace.lng },
      map: map,
      title: 'Punto de Destino',
      label: {
        text: '🏁',
        fontSize: '20px',
        fontWeight: 'bold'
      },
      animation: window.google.maps.Animation.DROP
    })

    setEndMarker(marker)
    
    // Ajustar vista del mapa
    if (startMarker && startPlace) {
      const bounds = new window.google.maps.LatLngBounds()
      bounds.extend(new window.google.maps.LatLng(startPlace.lat, startPlace.lng))
      bounds.extend(new window.google.maps.LatLng(endPlace.lat, endPlace.lng))
      map.fitBounds(bounds)
    } else {
      map.setCenter({ lat: endPlace.lat, lng: endPlace.lng })
      if (map.getZoom() < 15) {
        map.setZoom(15)
      }
    }
  }

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
    if (!startAddress.trim() || !endAddress.trim()) {
      alert('Por favor ingresa direcciones de inicio y destino')
      return
    }

    // Si tenemos los lugares de Google Maps, usarlos directamente
    if (startPlace && endPlace) {
      setLoading(true)
      setRouteResult(null)
      setShowSaveDialog(false)

      try {
        const points = [
          { name: 'Punto de Inicio', address: startAddress, lat: startPlace.lat, lng: startPlace.lng },
          { name: 'Punto de Destino', address: endAddress, lat: endPlace.lat, lng: endPlace.lng }
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
    } else {
      // Si no tenemos los lugares, intentar geocodificar
      setLoading(true)
      setRouteResult(null)
      setShowSaveDialog(false)

      try {
        const points = [
          { name: 'Punto de Inicio', address: startAddress.trim() },
          { name: 'Punto de Destino', address: endAddress.trim() }
        ]
        
        console.log('Calculando ruta con direcciones:', points)
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

      {/* Formulario de Rutas con Autocompletado de Google Maps */}
      <div className="api-form" style={{ marginBottom: '20px' }}>
        <div className="form-field" style={{ marginBottom: '20px' }}>
          <label htmlFor="start-address">
            <span style={{ fontSize: '1.2em', marginRight: '10px' }}>🚩</span>
            Punto de Inicio
          </label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              ref={startInputRef}
              type="text"
              id="start-address"
              value={startAddress}
              onChange={(e) => setStartAddress(e.target.value)}
              placeholder="Escribe una dirección o selecciona del mapa"
              className="form-input"
              style={{ flex: 1, padding: '12px' }}
            />
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
          {!mapsLoaded && googleMapsApiKey && (
            <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
              Cargando Google Maps...
            </small>
          )}
          {!googleMapsApiKey && (
            <small style={{ color: 'var(--warning)', display: 'block', marginTop: '5px' }}>
              ⚠️ Google Maps API key no configurada. Configura GOOGLE_MAPS_API_KEY en Render.
            </small>
          )}
        </div>

        <div className="form-field" style={{ marginBottom: '20px' }}>
          <label htmlFor="end-address">
            <span style={{ fontSize: '1.2em', marginRight: '10px' }}>🏁</span>
            Punto de Destino
          </label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              ref={endInputRef}
              type="text"
              id="end-address"
              value={endAddress}
              onChange={(e) => setEndAddress(e.target.value)}
              placeholder="Escribe una dirección o selecciona del mapa"
              className="form-input"
              style={{ flex: 1, padding: '12px' }}
            />
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

        {/* Mapa de Google Maps */}
        {googleMapsApiKey && (
          <div className="form-field" style={{ marginBottom: '20px' }}>
            <label>Mapa de Google Maps</label>
            <div
              ref={mapRef}
              style={{
                width: '100%',
                height: '400px',
                border: '1px solid #ddd',
                borderRadius: '8px',
                marginTop: '10px'
              }}
            />
            <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
              Escribe una dirección en los campos de arriba o haz click en el mapa para seleccionar puntos.
            </small>
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
        <p><strong>Geocodificación:</strong> Utilizamos Google Maps Places API Autocomplete directamente en el navegador para autocompletado en tiempo real. Las direcciones se geocodifican usando Google Maps Geocoding API.</p>
        <p><strong>Algoritmo:</strong> A* (A estrella)</p>
        <p><strong>Tipo:</strong> Búsqueda heurística</p>
        <p><strong>Justificación:</strong> A* combina el costo real del camino con una heurística estimada, encontrando la ruta óptima de manera eficiente.</p>
        <p><strong>Proceso:</strong></p>
        <ol>
          <li>Autocompletado de direcciones con Google Maps Places API (en el navegador)</li>
          <li>Geocodificar direcciones a coordenadas (lat, lng) usando Google Maps Geocoding API</li>
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

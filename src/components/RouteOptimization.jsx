import { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap, Polyline } from 'react-leaflet'
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

// Componente para actualizar el centro del mapa y ajustar bounds
function MapUpdater({ center, zoom, bounds }) {
  const map = useMap()
  useEffect(() => {
    if (bounds && bounds.length > 0) {
      // Si hay bounds (rutas), ajustar la vista para mostrar toda la ruta
      const latLngBounds = L.latLngBounds(bounds)
      map.fitBounds(latLngBounds, { padding: [50, 50] })
    } else {
      // Si no hay bounds, usar center y zoom
      map.setView(center, zoom)
    }
  }, [map, center, zoom, bounds])
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
  const [applyingAddress, setApplyingAddress] = useState(null) // 'start' o 'end' o null
  const [selectedRouteIndex, setSelectedRouteIndex] = useState(0) // Índice de la ruta seleccionada (0 = mejor)
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

  // Refs para control de cálculo automático
  const calculateRoutesRef = useRef(null)
  const lastCalculatedRef = useRef(null)
  const isCalculatingRef = useRef(false) // Para evitar cálculos simultáneos
  
  // Aplicar dirección (geocodificar y mostrar marcador)
  const handleApplyAddress = async (pointType) => {
    const address = pointType === 'start' ? startAddress.trim() : endAddress.trim()
    
    if (!address || address.length < 2) {
      alert('Por favor ingresa una dirección válida')
      return
    }
    
    setApplyingAddress(pointType)
    setLoading(true)
    
    try {
      const result = await routeOptimizationAPI.applyAddress(address)
      
      if (result.success) {
        const place = {
          address: result.display_name || address,
          lat: result.lat,
          lng: result.lng
        }
        
        console.log(`✅ Dirección aplicada (${pointType}):`, place)
        
        // Actualizar el estado correspondiente
        if (pointType === 'start') {
          setStartPlace(place)
          setStartAddress(result.display_name || address)
        } else {
          setEndPlace(place)
          setEndAddress(result.display_name || address)
        }
        
        // Limpiar resultado anterior y resetear lastCalculatedRef para forzar recálculo
        setRouteResult(null)
        lastCalculatedRef.current = null
        
        console.log(`📍 Estado actualizado. startPlace:`, pointType === 'start' ? place : startPlace)
        console.log(`📍 Estado actualizado. endPlace:`, pointType === 'end' ? place : endPlace)
      }
    } catch (err) {
      console.error('Error al aplicar dirección:', err)
      alert(`Error: ${err.message || 'No se pudo geocodificar la dirección'}`)
    } finally {
      setApplyingAddress(null)
      setLoading(false)
    }
  }
  
  // Función para calcular rutas automáticamente (definida después de los refs)
  const calculateRoutesAutomatically = async () => {
    // Evitar cálculos simultáneos
    if (isCalculatingRef.current) {
      console.log('⏸️ Ya hay un cálculo en curso, omitiendo...')
      return
    }
    
    if (!startPlace || !endPlace) {
      console.log('⏸️ No se pueden calcular rutas: faltan puntos', { startPlace, endPlace })
      return
    }
    
    isCalculatingRef.current = true
    console.log('🚀 Calculando rutas automáticamente...', {
      start: { lat: startPlace.lat, lng: startPlace.lng, address: startPlace.address },
      end: { lat: endPlace.lat, lng: endPlace.lng, address: endPlace.address }
    })
    
    setLoading(true)
    setRouteResult(null)
    
    try {
      const points = [
        { name: 'Punto de Inicio', address: startPlace.address, lat: startPlace.lat, lng: startPlace.lng },
        { name: 'Punto de Destino', address: endPlace.address, lat: endPlace.lat, lng: endPlace.lng }
      ]
      
      console.log('📤 Enviando puntos a API:', points)
      const result = await routeOptimizationAPI.optimize(points, algorithm, 0, false, null)
      console.log('📥 Resultado de la API:', result)
      
      if (result && result.routes && result.routes.length > 0) {
        console.log(`✅ Se obtuvieron ${result.routes.length} rutas de OSRM`)
        console.log('Rutas:', result.routes.map(r => ({
          number: r.route_number,
          distance: r.distance_km,
          duration: r.duration_minutes,
          coords: r.coordinates?.length || 0
        })))
        setRouteResult(result)
        setSelectedRouteIndex(0) // Seleccionar la mejor ruta por defecto
      } else {
        console.warn('⚠️ No se obtuvieron rutas de OSRM')
        alert('No se pudieron calcular rutas. Verifica que las direcciones sean válidas y estén en una zona con datos de calles disponibles.')
      }
    } catch (err) {
      console.error('❌ Error al calcular rutas:', err)
      alert(`Error al calcular rutas: ${err.message || 'Por favor, intenta nuevamente'}`)
    } finally {
      setLoading(false)
      isCalculatingRef.current = false
    }
  }
  
  // Calcular rutas automáticamente cuando hay inicio y fin
  useEffect(() => {
    console.log('🔄 useEffect ejecutado:', {
      startPlace: startPlace ? { lat: startPlace.lat, lng: startPlace.lng } : null,
      endPlace: endPlace ? { lat: endPlace.lat, lng: endPlace.lng } : null,
      isCalculating: isCalculatingRef.current,
      lastCalculated: lastCalculatedRef.current
    })
    
    // Función interna para calcular rutas
    const calculateRoutes = async () => {
      // Evitar cálculos simultáneos
      if (isCalculatingRef.current) {
        console.log('⏸️ Ya hay un cálculo en curso, omitiendo...')
        return
      }
      
      // Usar los valores actuales del estado
      const currentStart = startPlace
      const currentEnd = endPlace
      
      if (!currentStart || !currentEnd) {
        console.log('⏸️ No se pueden calcular rutas: faltan puntos', { currentStart, currentEnd })
        return
      }
      
      isCalculatingRef.current = true
      console.log('🚀 Calculando rutas automáticamente...', {
        start: { lat: currentStart.lat, lng: currentStart.lng, address: currentStart.address },
        end: { lat: currentEnd.lat, lng: currentEnd.lng, address: currentEnd.address }
      })
      
      setLoading(true)
      setRouteResult(null)
      
      try {
        const points = [
          { name: 'Punto de Inicio', address: currentStart.address, lat: currentStart.lat, lng: currentStart.lng },
          { name: 'Punto de Destino', address: currentEnd.address, lat: currentEnd.lat, lng: currentEnd.lng }
        ]
        
        console.log('📤 Enviando puntos a API:', points)
        const result = await routeOptimizationAPI.optimize(points, algorithm, 0, false, null)
        console.log('📥 Resultado completo de la API:', JSON.stringify(result, null, 2))
        
        if (result && result.routes && result.routes.length > 0) {
          console.log(`✅ Se obtuvieron ${result.routes.length} rutas de OSRM`)
          result.routes.forEach((r, i) => {
            console.log(`  Ruta ${i + 1}:`, {
              number: r.route_number,
              distance: r.distance_km,
              duration: r.duration_minutes,
              coords_count: r.coordinates?.length || 0,
              first_coord: r.coordinates?.[0],
              last_coord: r.coordinates?.[r.coordinates?.length - 1]
            })
          })
          
          // Si el punto de destino fue ajustado a la calle, actualizar endPlace
          if (result.points_info && result.points_info.length >= 2) {
            const endPointInfo = result.points_info[1]
            // Si hay coordenadas ajustadas (lat/lng) y son diferentes de las originales, actualizar
            if (endPointInfo.lat && endPointInfo.lng && endPointInfo.original_lat && endPointInfo.original_lng) {
              const latDiff = Math.abs(endPointInfo.lat - endPointInfo.original_lat)
              const lngDiff = Math.abs(endPointInfo.lng - endPointInfo.original_lng)
              // Si la diferencia es mayor a 0.0001 (aproximadamente 10 metros), se ajustó
              if (latDiff > 0.0001 || lngDiff > 0.0001) {
                console.log(`📍 Punto de destino ajustado a la calle: (${endPointInfo.original_lat}, ${endPointInfo.original_lng}) -> (${endPointInfo.lat}, ${endPointInfo.lng})`)
                // Actualizar el estado del punto de destino con las coordenadas ajustadas
                setEndPlace(prev => ({
                  ...prev,
                  lat: endPointInfo.lat,
                  lng: endPointInfo.lng
                }))
              }
            } else if (endPointInfo.lat && endPointInfo.lng) {
              // Si solo hay coordenadas ajustadas (sin originales), actualizar de todas formas
              setEndPlace(prev => ({
                ...prev,
                lat: endPointInfo.lat,
                lng: endPointInfo.lng
              }))
            }
          }
          
          setRouteResult(result)
          setSelectedRouteIndex(0) // Seleccionar la mejor ruta por defecto
        } else {
          console.warn('⚠️ No se obtuvieron rutas de OSRM. Resultado:', result)
          if (result && !result.routes) {
            console.warn('⚠️ El resultado no tiene la propiedad "routes"')
          }
          alert('No se pudieron calcular rutas. Verifica que las direcciones sean válidas y estén en una zona con datos de calles disponibles.')
        }
      } catch (err) {
        console.error('❌ Error al calcular rutas:', err)
        console.error('❌ Stack trace:', err.stack)
        alert(`Error al calcular rutas: ${err.message || 'Por favor, intenta nuevamente'}`)
      } finally {
        setLoading(false)
        isCalculatingRef.current = false
      }
    }
    
    // Limpiar timeout anterior
    if (calculateRoutesRef.current) {
      clearTimeout(calculateRoutesRef.current)
      calculateRoutesRef.current = null
    }
    
    // Si tenemos ambos puntos, calcular rutas
    if (startPlace && endPlace) {
      const currentKey = `${startPlace.lat},${startPlace.lng},${endPlace.lat},${endPlace.lng}`
      
      console.log('🔍 Verificando si debe calcular:', {
        currentKey,
        lastCalculated: lastCalculatedRef.current,
        isCalculating: isCalculatingRef.current,
        shouldCalculate: lastCalculatedRef.current !== currentKey && !isCalculatingRef.current
      })
      
      // Solo calcular si cambió algo o si es la primera vez
      if (lastCalculatedRef.current !== currentKey && !isCalculatingRef.current) {
        console.log('🔄 Coordenadas cambiaron, programando cálculo de rutas en 800ms...', currentKey)
        // Esperar 800ms antes de calcular (debounce)
        calculateRoutesRef.current = setTimeout(() => {
          console.log('⏰ Timeout ejecutado, iniciando cálculo...')
          calculateRoutes()
          lastCalculatedRef.current = currentKey
        }, 800)
      } else {
        console.log('⏭️ Omitiendo cálculo:', {
          reason: lastCalculatedRef.current === currentKey ? 'coordenadas no cambiaron' : 'cálculo en curso'
        })
      }
    } else {
      console.log('⏸️ No hay suficientes puntos para calcular:', { startPlace: !!startPlace, endPlace: !!endPlace })
    }
    
    return () => {
      if (calculateRoutesRef.current) {
        console.log('🧹 Limpiando timeout...')
        clearTimeout(calculateRoutesRef.current)
        calculateRoutesRef.current = null
      }
    }
  }, [startPlace?.lat, startPlace?.lng, endPlace?.lat, endPlace?.lng, algorithm])
  
  // Función para obtener coordenadas de todas las rutas para ajustar bounds
  const getRoutesBounds = () => {
    if (!routeResult || !routeResult.routes || routeResult.routes.length === 0) {
      return []
    }
    
    const allCoordinates = []
    routeResult.routes.forEach(route => {
      if (route.coordinates && route.coordinates.length > 0) {
        allCoordinates.push(...route.coordinates)
      }
    })
    
    // Agregar puntos de inicio y fin
    if (routeResult.points_info) {
      routeResult.points_info.forEach(point => {
        if (point.lat && point.lng) {
          allCoordinates.push([point.lat, point.lng])
        }
      })
    }
    
    return allCoordinates
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
            'User-Agent': 'RouteOptimizationApp/2.0'
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
          Algoritmos de búsqueda - Optimiza rutas de distribución minimizando distancia
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
              className="btn"
              onClick={() => handleApplyAddress('start')}
              disabled={!startAddress.trim() || applyingAddress === 'start' || loading}
              style={{ 
                whiteSpace: 'nowrap',
                minWidth: '100px'
              }}
            >
              {applyingAddress === 'start' ? 'Aplicando...' : 'Aplicar'}
            </button>
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
              {selectingPoint === 'start' ? '✓ Seleccionando...' : '📍 Mapa'}
            </button>
          </div>
          {selectingPoint === 'start' && (
            <small style={{ color: 'var(--primary)', display: 'block', marginTop: '5px', fontWeight: 'bold' }}>
              👆 Haz click en el mapa para seleccionar el punto de inicio
            </small>
          )}
          {startPlace && (
            <small style={{ color: 'var(--success)', display: 'block', marginTop: '5px' }}>
              ✓ Dirección aplicada: {startPlace.address}
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
              className="btn"
              onClick={() => handleApplyAddress('end')}
              disabled={!endAddress.trim() || applyingAddress === 'end' || loading}
              style={{ 
                whiteSpace: 'nowrap',
                minWidth: '100px'
              }}
            >
              {applyingAddress === 'end' ? 'Aplicando...' : 'Aplicar'}
            </button>
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
              {selectingPoint === 'end' ? '✓ Seleccionando...' : '📍 Mapa'}
            </button>
          </div>
          {selectingPoint === 'end' && (
            <small style={{ color: 'var(--primary)', display: 'block', marginTop: '5px', fontWeight: 'bold' }}>
              👆 Haz click en el mapa para seleccionar el punto de destino
            </small>
          )}
          {endPlace && (
            <small style={{ color: 'var(--success)', display: 'block', marginTop: '5px' }}>
              ✓ Dirección aplicada: {endPlace.address}
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
              <MapUpdater center={mapCenter} zoom={mapZoom} bounds={getRoutesBounds()} />
              <MapClickHandler onMapClick={handleMapClick} selectingPoint={selectingPoint} />
              
              {/* Mostrar las 3 rutas en el mapa - La mejor ruta (índice 0) se destaca */}
              {routeResult && routeResult.routes && routeResult.routes.length > 0 && routeResult.routes.map((route, idx) => {
                if (!route.coordinates || route.coordinates.length === 0) {
                  console.warn(`⚠️ Ruta ${idx + 1} no tiene coordenadas`)
                  return null
                }
                
                const isSelected = idx === selectedRouteIndex
                const isBestRoute = idx === 0 // La primera ruta es siempre la mejor (más corta)
                const colors = ['#6e8bff', '#20c997', '#f8c22b'] // Azul (mejor), Verde, Amarillo
                const routeColor = colors[idx] || '#6e8bff'
                
                // La mejor ruta (índice 0) siempre se muestra más destacada
                const weight = isBestRoute ? 7 : (isSelected ? 5 : 3)
                const opacity = isBestRoute ? 1.0 : (isSelected ? 0.8 : 0.5)
                const dashArray = isBestRoute ? null : (isSelected ? null : '15, 10')
                
                return (
                  <Polyline
                    key={`route-${idx}`}
                    positions={route.coordinates}
                    pathOptions={{
                      color: routeColor,
                      weight: weight,
                      opacity: opacity,
                      dashArray: dashArray
                    }}
                    eventHandlers={{
                      click: () => {
                        console.log(`📍 Ruta ${idx + 1} seleccionada`)
                        setSelectedRouteIndex(idx)
                      }
                    }}
                  />
                )
              })}
              
              {/* Indicador de carga */}
              {loading && (
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  background: 'rgba(0, 0, 0, 0.7)',
                  color: 'white',
                  padding: '15px 25px',
                  borderRadius: '8px',
                  zIndex: 1000,
                  fontSize: '16px',
                  fontWeight: 'bold'
                }}>
                  Calculando rutas...
                </div>
              )}
              
              {/* Marcadores de inicio y fin */}
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
            Escribe una dirección y presiona "Aplicar" para añadirla al mapa. Las rutas se calculan automáticamente. Mapas actualizados 2025 - CartoDB Voyager.
          </small>
        </div>

        {/* Leyenda de colores de rutas - Mostrar las 3 mejores rutas */}
        {routeResult && routeResult.routes && routeResult.routes.length > 0 && (
          <div className="message" style={{ marginBottom: '20px', background: 'rgba(110, 139, 255, 0.1)', padding: '20px', borderRadius: '8px', border: '2px solid var(--primary)' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px', color: 'var(--primary)' }}>
              🗺️ {routeResult.routes.length} Mejores Rutas Calculadas
            </h3>
            <p style={{ marginBottom: '15px', color: 'var(--text-secondary)', fontSize: '0.9em' }}>
              Las rutas se muestran en el mapa siguiendo las calles reales. La ruta destacada en <strong style={{ color: '#6e8bff' }}>azul</strong> es la más corta.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {routeResult.routes.map((route, idx) => {
                const colors = ['#6e8bff', '#20c997', '#f8c22b']
                const routeColor = colors[idx] || '#6e8bff'
                const isSelected = idx === selectedRouteIndex
                const isBestRoute = idx === 0
                const labels = ['🥇 Mejor Ruta (Más Corta)', '🥈 Ruta Alternativa 1', '🥉 Ruta Alternativa 2']
                
                return (
                  <div 
                    key={idx}
                    onClick={() => setSelectedRouteIndex(idx)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '12px',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      backgroundColor: isBestRoute 
                        ? 'rgba(110, 139, 255, 0.25)' 
                        : (isSelected ? 'rgba(110, 139, 255, 0.15)' : 'rgba(255,255,255,0.05)'),
                      border: isBestRoute 
                        ? '2px solid #6e8bff' 
                        : (isSelected ? '2px solid var(--primary)' : '1px solid rgba(255,255,255,0.1)'),
                      transition: 'all 0.2s',
                      transform: isBestRoute ? 'scale(1.02)' : 'scale(1)'
                    }}
                  >
                    <div style={{
                      width: '40px',
                      height: '5px',
                      backgroundColor: routeColor,
                      borderRadius: '3px',
                      boxShadow: isBestRoute ? `0 0 8px ${routeColor}` : 'none'
                    }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ 
                        fontWeight: isBestRoute ? '700' : (isSelected ? '600' : '400'),
                        fontSize: isBestRoute ? '1.05em' : '1em',
                        color: isBestRoute ? '#6e8bff' : 'var(--text)',
                        marginBottom: '4px'
                      }}>
                        {labels[idx] || `Ruta ${idx + 1}`}
                      </div>
                      <div style={{ 
                        fontSize: '0.9em', 
                        color: 'var(--text-secondary)',
                        display: 'flex',
                        gap: '15px'
                      }}>
                        <span>📏 <strong>{route.distance_km}</strong> km</span>
                        <span>⏱️ <strong>{route.duration_minutes}</strong> min</span>
                      </div>
                    </div>
                    {isBestRoute && (
                      <span style={{ 
                        fontSize: '1.5em',
                        filter: 'drop-shadow(0 2px 4px rgba(110, 139, 255, 0.5))'
                      }}>
                        ⭐
                      </span>
                    )}
                  </div>
                )
              })}
            </div>
            <small style={{ display: 'block', marginTop: '15px', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
              💡 Haz click en una ruta de la leyenda para resaltarla en el mapa. La ruta azul es la más corta y se muestra destacada.
            </small>
          </div>
        )}
        
        {/* Mensaje de carga mientras se calculan las rutas */}
        {loading && startPlace && endPlace && (
          <div className="message" style={{ marginBottom: '20px', background: 'rgba(110, 139, 255, 0.1)', padding: '20px', borderRadius: '8px', textAlign: 'center' }}>
            <div style={{ fontSize: '2em', marginBottom: '10px' }}>🔄</div>
            <h4 style={{ marginTop: 0, marginBottom: '10px' }}>Calculando las mejores rutas...</h4>
            <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
              Obteniendo rutas reales por calles usando OSRM. Esto puede tomar unos segundos.
            </p>
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

      {routeResult && routeResult.route && routeResult.route.length > 0 && !routeResult.has_osrm_routes && (
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

    </section>
  )
}

export default RouteOptimization

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from pydantic import BaseModel
import httpx
import re
import os
from datetime import datetime
from app.database import get_db
from app.auth import get_current_user
from app.models import User, Route, RoutePoint
from app.schemas import RouteCreate, RouteResponse
from app.algorithms.route_optimizer import RouteOptimizer

router = APIRouter(prefix="/api/route-optimization", tags=["route-optimization"])

class PointInput(BaseModel):
    name: str
    address: Optional[str] = None  # Dirección
    lat: Optional[float] = None    # Latitud (si viene del autocompletado de Google Maps)
    lng: Optional[float] = None    # Longitud (si viene del autocompletado de Google Maps)

class RouteRequest(BaseModel):
    points: List[PointInput]
    algorithm: str = "astar"
    start_point: int = 0
    save_route: bool = False  # Si se debe guardar la ruta
    route_name: Optional[str] = None  # Nombre de la ruta si se guarda

def normalize_address(address: str) -> str:
    """Normalizar dirección: limpiar espacios alrededor de comas"""
    if not address:
        return ""
    
    # Reemplazar múltiples espacios con uno solo
    address = re.sub(r'\s+', ' ', address.strip())
    # Manejar comas sin espacios: "calle,ciudad" -> "calle, ciudad"
    # Pero preservar espacios existentes: "calle, ciudad" -> "calle, ciudad"
    address = re.sub(r',([^,\s])', r', \1', address)  # Agregar espacio después de coma si no existe
    # Eliminar espacios antes de comas
    address = re.sub(r'\s+,', ',', address)
    # Limpiar espacios múltiples nuevamente
    address = re.sub(r'\s+', ' ', address.strip())
    return address

# Constantes de API Nominatim (OpenStreetMap)
# Nominatim es completamente gratuito y no requiere API key
NOMINATIM_GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"

# Constante de OSRM (OpenStreetMap Routing Machine) - Rutas reales por calles
# Formato: https://router.project-osrm.org/route/v1/{profile}/{coordinates}
# coordinates: lng1,lat1;lng2,lat2
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"

async def geocode_address(address: str) -> Dict:
    """Geocodificar dirección usando Nominatim (OpenStreetMap) - Normaliza número adelante/atrás"""
    try:
        # Normalizar dirección base
        normalized_address = normalize_address(address)
        
        # Crear variantes de la dirección para manejar número adelante/atrás
        address_variants = [normalized_address]
        
        # Si hay un número al final, crear variante con número al inicio
        match_end_number = re.search(r'^(.+?)\s+(\d+)$', normalized_address)
        if match_end_number:
            street = match_end_number.group(1).strip()
            number = match_end_number.group(2).strip()
            variant = f"{number} {street}"
            if variant not in address_variants:
                address_variants.append(variant)
        
        # Si hay un número al inicio, crear variante con número al final
        match_start_number = re.search(r'^(\d+)\s+(.+)$', normalized_address)
        if match_start_number:
            number = match_start_number.group(1).strip()
            street = match_start_number.group(2).strip()
            variant = f"{street} {number}"
            if variant not in address_variants:
                address_variants.append(variant)
        
        # Intentar geocodificar con cada variante
        async with httpx.AsyncClient() as client:
            headers = {
                "User-Agent": "RouteOptimizationApp/2.0 (contact@example.com)"
            }
            
            for variant in address_variants:
                print(f"Geocodificando con Nominatim (variante): {variant}")
                
                try:
                    response = await client.get(
                        NOMINATIM_GEOCODE_URL,
                        params={
                            "q": variant,
                            "format": "json",
                            "limit": 1,
                            "addressdetails": 1,
                            "countrycodes": "cl",
                            "accept-language": "es"
                        },
                        headers=headers,
                        timeout=15.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if isinstance(data, list) and len(data) > 0:
                            result = data[0]
                            lat = float(result.get("lat", 0))
                            lng = float(result.get("lon", 0))
                            display_name = result.get("display_name", variant)
                            
                            address_details = result.get("address", {})
                            city = address_details.get("city") or address_details.get("town") or address_details.get("village") or ""
                            country = address_details.get("country", "")
                            address_line1 = address_details.get("road", "") or address_details.get("house_number", "")
                            address_line2 = ""
                            
                            parts = []
                            if address_details.get("suburb"):
                                parts.append(address_details.get("suburb"))
                            if address_details.get("postcode"):
                                parts.append(address_details.get("postcode"))
                            address_line2 = ", ".join(parts)
                            
                            geo_result = {
                                "lat": lat,
                                "lng": lng,
                                "display_name": display_name,
                                "address_line1": address_line1 or display_name.split(",")[0] if display_name else "",
                                "address_line2": address_line2,
                                "city": city,
                                "country": country
                            }
                            print(f"Geocodificación exitosa: {geo_result['display_name']} - ({geo_result['lat']}, {geo_result['lng']})")
                            return geo_result
                except Exception as e:
                    print(f"Error al geocodificar variante '{variant}': {str(e)}")
                    continue
            
            # Si ninguna variante funcionó, lanzar error
            raise ValueError(f"No se encontraron resultados para la dirección: {address} (variantes intentadas: {address_variants})")
            
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error al geocodificar '{address}': {str(e)}"
        print(f"Exception: {error_msg}")
        import traceback
        traceback.print_exc()
        raise ValueError(error_msg)

async def autocomplete_address(query: str) -> List[Dict]:
    """Autocompletar dirección usando Nominatim Search (OpenStreetMap) - Actualizado 2025"""
    try:
        if not query or len(query) < 2:  # Reducir a 2 caracteres para más respuestas
            return []
        
        print(f"Autocompletado Nominatim 2025 - Query: {query}")
        
        async with httpx.AsyncClient() as client:
            # Headers requeridos por Nominatim - Actualizado 2025
            headers = {
                "User-Agent": "RouteOptimizationApp/2.0 (contact@example.com)"
            }
            
            # Llamar a Nominatim Search API con más opciones
            response = await client.get(
                NOMINATIM_SEARCH_URL,
                params={
                    "q": query,
                    "format": "json",
                    "limit": 8,  # Aumentar a 8 resultados
                    "addressdetails": 1,
                    "countrycodes": "cl",  # Priorizar Chile
                    "accept-language": "es",
                    "dedupe": 1,  # Eliminar duplicados
                    "extratags": 1,  # Información adicional
                    "namedetails": 1  # Nombres alternativos
                },
                headers=headers,
                timeout=8.0  # Reducir timeout para respuesta más rápida
            )
            
            print(f"Autocompletado Nominatim 2025 - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    results = []
                    for item in data[:8]:  # Limitar a 8 resultados
                        lat = float(item.get("lat", 0))
                        lng = float(item.get("lon", 0))
                        display_name = item.get("display_name", query)
                        
                        # Extraer componentes de la dirección
                        address_details = item.get("address", {})
                        city = address_details.get("city") or address_details.get("town") or address_details.get("village") or address_details.get("municipality") or ""
                        country = address_details.get("country", "")
                        
                        # Mejorar address_line1 con más información
                        main_parts = []
                        if address_details.get("house_number"):
                            main_parts.append(address_details.get("house_number"))
                        if address_details.get("road"):
                            main_parts.append(address_details.get("road"))
                        if address_details.get("neighbourhood"):
                            main_parts.append(address_details.get("neighbourhood"))
                        address_line1 = " ".join(main_parts) if main_parts else display_name.split(",")[0]
                        
                        # Mejorar address_line2
                        address_line2_parts = []
                        if address_details.get("suburb"):
                            address_line2_parts.append(address_details.get("suburb"))
                        if address_details.get("postcode"):
                            address_line2_parts.append(address_details.get("postcode"))
                        address_line2 = ", ".join(address_line2_parts)
                        
                        # Priorizar resultados más relevantes
                        relevance_score = item.get("importance", 0)
                        
                        results.append({
                            "text": display_name,
                            "display_name": display_name,
                            "address_line1": address_line1,
                            "address_line2": address_line2,
                            "city": city,
                            "country": country,
                            "lat": lat,
                            "lng": lng,
                            "importance": relevance_score
                        })
                    
                    # Ordenar por importancia (relevancia)
                    results.sort(key=lambda x: x.get("importance", 0), reverse=True)
                    
                    print(f"Autocompletado Nominatim 2025 - Results: {len(results)}")
                    return results
                else:
                    print("Autocompletado Nominatim 2025 - No se encontraron resultados")
                    return []
            else:
                print(f"Autocompletado Nominatim 2025 - Error HTTP: {response.status_code}")
                return []
        
    except Exception as e:
        print(f"Error en autocompletado Nominatim 2025: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

async def get_osrm_routes(lat1: float, lng1: float, lat2: float, lng2: float, alternatives: int = 2) -> List[Dict]:
    """Obtener rutas reales por calles usando OSRM (OpenStreetMap Routing Machine)"""
    try:
        print(f"🔵 OSRM: Obteniendo rutas desde ({lat1}, {lng1}) hasta ({lat2}, {lng2})")
        
        async with httpx.AsyncClient() as client:
            # OSRM usa formato: lng,lat (longitud primero) y las coordenadas van en la URL
            coordinates = f"{lng1},{lat1};{lng2},{lat2}"
            # URL correcta: https://router.project-osrm.org/route/v1/driving/{coordinates}?alternatives=2&steps=true&geometries=geojson&overview=full
            url = f"{OSRM_ROUTE_URL}/{coordinates}"
            
            print(f"🔵 OSRM URL completa: {url}")
            print(f"🔵 Parámetros: alternatives={alternatives}, steps=true, geometries=geojson, overview=full")
            
            response = await client.get(
                url,
                params={
                    "alternatives": alternatives,  # Obtener rutas alternativas
                    "steps": "true",  # Incluir pasos detallados
                    "geometries": "geojson",  # Formato GeoJSON
                    "overview": "full",  # Vista completa de la ruta
                    "annotations": "true"  # Incluir distancia y tiempo
                },
                timeout=20.0
            )
            
            print(f"🔵 OSRM Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"🔵 OSRM Response Data: {data.get('code', 'No code')}, Routes: {len(data.get('routes', []))}")
                
                if data.get("code") == "Ok" and data.get("routes"):
                    routes = []
                    for idx, route in enumerate(data["routes"]):
                        distance_meters = route.get("distance", 0)
                        duration_seconds = route.get("duration", 0)
                        distance_km = distance_meters / 1000
                        duration_minutes = duration_seconds / 60
                        
                        geometry = route.get("geometry", {})
                        coordinates_list = geometry.get("coordinates", [])
                        
                        print(f"🔵 Ruta {idx + 1}: {len(coordinates_list)} coordenadas, {distance_km:.2f} km, {duration_minutes:.2f} min")
                        
                        # Convertir coordenadas de [lng, lat] a [lat, lng] para Leaflet
                        route_coordinates = [[coord[1], coord[0]] for coord in coordinates_list]
                        
                        routes.append({
                            "route_number": idx + 1,
                            "distance_km": round(distance_km, 2),
                            "duration_minutes": round(duration_minutes, 2),
                            "distance_meters": round(distance_meters, 2),
                            "duration_seconds": round(duration_seconds, 2),
                            "coordinates": route_coordinates,
                            "description": f"Ruta {idx + 1}: {round(distance_km, 2)} km, {round(duration_minutes, 2)} min"
                        })
                    
                    # Ordenar por distancia (las más cortas primero)
                    routes.sort(key=lambda x: x["distance_km"])
                    
                    # Limitar a las 3 mejores
                    routes = routes[:3]
                    
                    # Renumerar rutas
                    for idx, route in enumerate(routes):
                        route["route_number"] = idx + 1
                    
                    print(f"✅ OSRM: Se obtuvieron {len(routes)} rutas")
                    return routes
                else:
                    error_msg = data.get('message', 'Unknown error')
                    print(f"❌ OSRM: Error en respuesta - Code: {data.get('code', 'Unknown')}, Message: {error_msg}")
                    return []
            else:
                error_text = response.text[:200] if hasattr(response, 'text') else 'No error text'
                print(f"❌ OSRM: Error HTTP {response.status_code}: {error_text}")
                return []
                
    except Exception as e:
        print(f"❌ Error al obtener rutas de OSRM: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def normalize_plan(plan: str) -> str:
    """Normalizar plan a minúsculas sin espacios"""
    if not plan:
        return 'free'
    return plan.lower().strip()

@router.post("/optimize")
async def optimize_route(
    request: RouteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Optimizar ruta de distribución - Parte 2"""
    # Refrescar usuario desde la base de datos para obtener el plan más actualizado
    db.refresh(current_user)
    
    # Normalizar plan para comparación robusta
    user_plan = normalize_plan(current_user.plan)
    
    # Verificar que el usuario tenga un plan válido (pro o enterprise)
    if user_plan not in ['pro', 'enterprise']:
        raise HTTPException(
            status_code=403, 
            detail=f"Optimización de rutas disponible solo en planes Pro y Enterprise. Tu plan actual: {current_user.plan}"
        )
    
    max_points = 50 if user_plan == 'pro' else 1000
    
    if len(request.points) > max_points:
        raise HTTPException(status_code=403, detail=f"Plan {current_user.plan} permite máximo {max_points} puntos. Has proporcionado {len(request.points)} puntos.")
    
    if len(request.points) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 puntos")
    
    try:
        # Geocodificar direcciones si es necesario
        points_with_coords = []
        for idx, point in enumerate(request.points):
            print(f"Procesando punto {idx + 1}: {point.name} - {point.address}")
            
            if point.address:
                # Si se proporcionan coordenadas directamente, usarlas (vienen del autocompletado de Google Maps)
                if point.lat is not None and point.lng is not None:
                    points_with_coords.append({
                        "name": point.name,
                        "lat": point.lat,
                        "lng": point.lng,
                        "address": point.address,
                        "display_name": point.address
                    })
                    print(f"Punto {idx + 1} usando coordenadas proporcionadas: ({point.lat}, {point.lng})")
                else:
                    # Si no hay coordenadas, geocodificar la dirección
                    try:
                        coords = await geocode_address(point.address)
                        points_with_coords.append({
                            "name": point.name,
                            "lat": coords["lat"],
                            "lng": coords["lng"],
                            "address": point.address,
                            "display_name": coords.get("display_name", point.address)
                        })
                        print(f"Punto {idx + 1} geocodificado exitosamente: {coords.get('display_name')}")
                    except ValueError as geocode_err:
                        # Si falla la geocodificación, intentar con diferentes variaciones
                        print(f"Error al geocodificar '{point.address}': {str(geocode_err)}")
                        print(f"Intentando variaciones...")
                        
                        # Intentar agregar "Chile" si no está presente
                        address_variations = [point.address]
                        if 'chile' not in point.address.lower():
                            address_variations.append(f"{point.address}, Chile")
                        
                        # Intentar con diferentes formatos
                        geocoded = False
                        for variation in address_variations:
                            try:
                                coords = await geocode_address(variation)
                                points_with_coords.append({
                                    "name": point.name,
                                    "lat": coords["lat"],
                                    "lng": coords["lng"],
                                    "address": point.address,
                                    "display_name": coords.get("display_name", point.address)
                                })
                                print(f"Punto {idx + 1} geocodificado con variación '{variation}'")
                                geocoded = True
                                break
                            except ValueError:
                                continue
                        
                        if not geocoded:
                            raise HTTPException(
                                status_code=400,
                                detail=f"No se pudo geocodificar la dirección '{point.address}'. Verifica que la dirección sea correcta e intenta con un formato más completo (ej: 'Calle, Ciudad, País')."
                            )
                        
            elif point.lat is not None and point.lng is not None:
                # Usar coordenadas proporcionadas
                points_with_coords.append({
                    "name": point.name,
                    "lat": point.lat,
                    "lng": point.lng,
                    "address": None,
                    "display_name": f"{point.name} ({point.lat}, {point.lng})"
                })
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Punto '{point.name}' debe tener dirección o coordenadas (lat, lng)"
                )
        
        # Si hay exactamente 2 puntos, usar OSRM para rutas reales por calles
        if len(points_with_coords) == 2:
            start_point = points_with_coords[0]
            end_point = points_with_coords[1]
            
            # Obtener las 3 mejores rutas usando OSRM (rutas reales por calles)
            osrm_routes = await get_osrm_routes(
                start_point["lat"], 
                start_point["lng"],
                end_point["lat"], 
                end_point["lng"],
                alternatives=2  # Obtener 2 alternativas + la principal = 3 rutas
            )
            
            if osrm_routes and len(osrm_routes) > 0:
                # Construir respuesta con las 3 mejores rutas
                result = {
                    "routes": osrm_routes,
                    "points_info": [
                        {
                            "name": start_point["name"],
                            "address": start_point.get("address"),
                            "display_name": start_point.get("display_name"),
                            "lat": start_point["lat"],
                            "lng": start_point["lng"]
                        },
                        {
                            "name": end_point["name"],
                            "address": end_point.get("address"),
                            "display_name": end_point.get("display_name"),
                            "lat": end_point["lat"],
                            "lng": end_point["lng"]
                        }
                    ],
                    "algorithm": "OSRM (Rutas reales por calles)",
                    "is_direct_route": True,
                    "has_osrm_routes": True
                }
                
                return result
            else:
                # Si OSRM falla, usar algoritmo tradicional como fallback
                print("OSRM no devolvió rutas, usando algoritmo tradicional como fallback")
                points_for_optimizer = [{"name": p["name"], "lat": p["lat"], "lng": p["lng"]} for p in points_with_coords]
                optimizer = RouteOptimizer(points_for_optimizer)
                result = optimizer.astar(0)
                result["points_info"] = [
                    {
                        "name": p["name"],
                        "address": p.get("address"),
                        "display_name": p.get("display_name"),
                        "lat": p["lat"],
                        "lng": p["lng"]
                    }
                    for p in points_with_coords
                ]
                result["has_osrm_routes"] = False
        else:
            # Para múltiples puntos, usar algoritmo tradicional (TSP)
            points_for_optimizer = [{"name": p["name"], "lat": p["lat"], "lng": p["lng"]} for p in points_with_coords]
            optimizer = RouteOptimizer(points_for_optimizer)
            
            if request.algorithm == "astar":
                result = optimizer.astar(request.start_point)
            elif request.algorithm == "dijkstra":
                result = optimizer.dijkstra(request.start_point)
            elif request.algorithm == "tsp":
                result = optimizer.tsp_nearest_neighbor(request.start_point)
            else:
                raise HTTPException(status_code=400, detail="Algoritmo no válido. Use 'astar', 'dijkstra' o 'tsp'")
            
            # Agregar información de direcciones a la respuesta
            result["points_info"] = [
                {
                    "name": p["name"],
                    "address": p.get("address"),
                    "display_name": p.get("display_name"),
                    "lat": p["lat"],
                    "lng": p["lng"]
                }
                for p in points_with_coords
            ]
            result["has_osrm_routes"] = False
        
        # Guardar ruta en la base de datos si se solicita
        if request.save_route:
            route_name = request.route_name or f"Ruta {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            # Determinar distancia para guardar (usar la primera ruta si hay rutas OSRM)
            distance_to_save = None
            if result.get("has_osrm_routes") and result.get("routes"):
                distance_to_save = result["routes"][0]["distance_km"]
            elif result.get("distance"):
                distance_to_save = result["distance"]
            
            # Crear ruta en BD
            new_route = Route(
                user_id=current_user.id,
                name=route_name,
                algorithm=result.get("algorithm", request.algorithm),
                distance=distance_to_save
            )
            db.add(new_route)
            db.flush()  # Para obtener el ID
            
            # Crear puntos de la ruta en BD
            if result.get("has_osrm_routes"):
                # Para rutas OSRM, guardar solo inicio y fin
                for idx, point_info in enumerate(result["points_info"]):
                    route_point = RoutePoint(
                        route_id=new_route.id,
                        name=point_info["name"],
                        address=point_info.get("address", ""),
                        lat=point_info["lat"],
                        lng=point_info["lng"],
                        display_name=point_info.get("display_name"),
                        order=idx
                    )
                    db.add(route_point)
            else:
                # Para rutas tradicionales, guardar todos los puntos del recorrido
                for idx, point_name in enumerate(result["route"]):
                    point_info = next((p for p in points_with_coords if p["name"] == point_name), None)
                    if point_info:
                        route_point = RoutePoint(
                            route_id=new_route.id,
                            name=point_info["name"],
                            address=point_info.get("address", ""),
                            lat=point_info["lat"],
                            lng=point_info["lng"],
                            display_name=point_info.get("display_name"),
                            order=idx
                        )
                        db.add(route_point)
            
            db.commit()
            db.refresh(new_route)
            result["saved_route_id"] = new_route.id
        
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al optimizar ruta: {str(e)}")

@router.get("", response_model=List[RouteResponse])
async def get_routes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener todas las rutas guardadas del usuario"""
    routes = db.query(Route).filter(
        Route.user_id == current_user.id
    ).order_by(Route.created_at.desc()).all()
    return routes

@router.get("/{route_id}", response_model=RouteResponse)
async def get_route(
    route_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener una ruta específica"""
    route = db.query(Route).filter(
        Route.id == route_id,
        Route.user_id == current_user.id
    ).first()
    
    if not route:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    
    return route

@router.delete("/{route_id}")
async def delete_route(
    route_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Eliminar una ruta"""
    route = db.query(Route).filter(
        Route.id == route_id,
        Route.user_id == current_user.id
    ).first()
    
    if not route:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    
    db.delete(route)
    db.commit()
    
    return {"message": "Ruta eliminada correctamente"}

class ApplyAddressRequest(BaseModel):
    address: str

@router.post("/apply-address")
async def apply_address(
    request: ApplyAddressRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Aplicar dirección: geocodificar y retornar coordenadas (sin calcular ruta)"""
    if not request.address or len(request.address.strip()) < 2:
        raise HTTPException(status_code=400, detail="Dirección debe tener al menos 2 caracteres")
    
    try:
        coords = await geocode_address(request.address.strip())
        return {
            "success": True,
            "address": request.address,
            "lat": coords["lat"],
            "lng": coords["lng"],
            "display_name": coords.get("display_name", request.address)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al aplicar dirección: {str(e)}")

@router.get("/autocomplete/search")
async def autocomplete_search(
    query: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Autocompletar direcciones mientras el usuario escribe - Actualizado 2025"""
    if not query or len(query) < 2:  # Reducir a 2 caracteres
        return []
    
    try:
        results = await autocomplete_address(query)
        print(f"Autocompletado 2025 para '{query}': {len(results)} resultados")
        return results
    except Exception as e:
        print(f"Error en autocompletado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en autocompletado: {str(e)}")


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

async def geocode_address(address: str) -> Dict:
    """Geocodificar dirección usando Nominatim (OpenStreetMap) - Gratuito y sin API key"""
    try:
        # Normalizar dirección
        normalized_address = normalize_address(address)
        
        print(f"Geocodificando con Nominatim: {normalized_address}")
        
        # Nominatim Geocoding API - Completamente gratuito
        async with httpx.AsyncClient() as client:
            # Agregar headers requeridos por Nominatim (User-Agent)
            headers = {
                "User-Agent": "RouteOptimizationApp/1.0 (contact@example.com)"  # Nominatim requiere User-Agent
            }
            
            response = await client.get(
                NOMINATIM_GEOCODE_URL,
                params={
                    "q": normalized_address,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                    "countrycodes": "cl",  # Priorizar Chile
                    "accept-language": "es"  # Idioma español
                },
                headers=headers,
                timeout=15.0
            )
            
            print(f"Geocodificación Nominatim - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    # Usar el primer resultado (más relevante)
                    result = data[0]
                    lat = float(result.get("lat", 0))
                    lng = float(result.get("lon", 0))
                    display_name = result.get("display_name", normalized_address)
                    
                    # Extraer componentes de la dirección
                    address_details = result.get("address", {})
                    city = address_details.get("city") or address_details.get("town") or address_details.get("village") or ""
                    country = address_details.get("country", "")
                    address_line1 = address_details.get("road", "") or address_details.get("house_number", "")
                    address_line2 = ""
                    
                    # Construir address_line2 con información adicional
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
                else:
                    raise ValueError(f"No se encontraron resultados para la dirección: {address}")
            else:
                raise ValueError(f"Error HTTP {response.status_code} al geocodificar")
            
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
        
        # Crear optimizador con puntos geocodificados
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
        
        # Guardar ruta en la base de datos si se solicita
        if request.save_route:
            route_name = request.route_name or f"Ruta {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            # Crear ruta en BD
            new_route = Route(
                user_id=current_user.id,
                name=route_name,
                algorithm=request.algorithm,
                distance=result["distance"]
            )
            db.add(new_route)
            db.flush()  # Para obtener el ID
            
            # Crear puntos de la ruta en BD
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


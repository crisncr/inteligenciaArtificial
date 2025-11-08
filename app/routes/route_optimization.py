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
    address: Optional[str] = None  # Dirección (nuevo)
    lat: Optional[float] = None    # Latitud (opcional si se proporciona address)
    lng: Optional[float] = None    # Longitud (opcional si se proporciona address)

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

# Constantes de API Google Maps
# Nota: Necesitas obtener una API key de Google Cloud Console y habilitar:
# - Geocoding API
# - Places API (para autocompletado)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_PLACES_AUTOCOMPLETE_URL = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
GOOGLE_PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

async def geocode_address(address: str) -> Dict:
    """Geocodificar dirección usando Google Maps Geocoding API"""
    try:
        if not GOOGLE_MAPS_API_KEY:
            raise ValueError("Google Maps API key no configurada. Configura la variable de entorno GOOGLE_MAPS_API_KEY")
        
        # Normalizar dirección
        normalized_address = normalize_address(address)
        
        print(f"Geocodificando con Google Maps: {normalized_address}")
        
        # Google Maps Geocoding API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                GOOGLE_GEOCODE_URL,
                params={
                    "address": normalized_address,
                    "key": GOOGLE_MAPS_API_KEY,
                    "language": "es",  # Idioma español
                    "region": "cl"  # Priorizar resultados de Chile
                },
                timeout=15.0
            )
            
            print(f"Geocodificación Google - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Geocodificación Google - Status API: {data.get('status')}")
                
                if data.get("status") == "OK" and data.get("results"):
                    # Usar el primer resultado (más relevante)
                    result = data["results"][0]
                    location = result.get("geometry", {}).get("location", {})
                    formatted_address = result.get("formatted_address", normalized_address)
                    
                    # Extraer componentes de la dirección
                    address_components = result.get("address_components", [])
                    city = ""
                    country = ""
                    address_line1 = ""
                    address_line2 = ""
                    
                    for component in address_components:
                        types = component.get("types", [])
                        long_name = component.get("long_name", "")
                        
                        if "locality" in types or "administrative_area_level_2" in types:
                            city = long_name
                        elif "country" in types:
                            country = long_name
                        elif "street_number" in types or "route" in types:
                            if address_line1:
                                address_line1 = f"{long_name} {address_line1}"
                            else:
                                address_line1 = long_name
                    
                    geo_result = {
                        "lat": float(location.get("lat", 0)),
                        "lng": float(location.get("lng", 0)),
                        "display_name": formatted_address,
                        "address_line1": address_line1 or formatted_address.split(",")[0] if formatted_address else "",
                        "address_line2": ", ".join(formatted_address.split(",")[1:]) if "," in formatted_address else "",
                        "city": city,
                        "country": country
                    }
                    print(f"Geocodificación exitosa: {geo_result['display_name']} - ({geo_result['lat']}, {geo_result['lng']})")
                    return geo_result
                elif data.get("status") == "ZERO_RESULTS":
                    raise ValueError(f"No se encontraron resultados para la dirección: {address}")
                elif data.get("status") == "OVER_QUERY_LIMIT":
                    raise ValueError("Límite de consultas excedido en Google Maps API")
                elif data.get("status") == "REQUEST_DENIED":
                    error_msg = data.get("error_message", "Acceso denegado a Google Maps API")
                    raise ValueError(f"Error en Google Maps API: {error_msg}")
                else:
                    raise ValueError(f"Error en Google Maps API: {data.get('status')} - {data.get('error_message', '')}")
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
    """Autocompletar dirección usando Google Places API Autocomplete"""
    try:
        if not GOOGLE_MAPS_API_KEY:
            print("Google Maps API key no configurada, autocompletado no disponible")
            return []
        
        if not query or len(query) < 3:
            return []
        
        print(f"Autocompletado Google Places - Query: {query}")
        
        async with httpx.AsyncClient() as client:
            # Llamar a Places Autocomplete API
            response = await client.get(
                GOOGLE_PLACES_AUTOCOMPLETE_URL,
                params={
                    "input": query,
                    "key": GOOGLE_MAPS_API_KEY,
                    "language": "es",
                    "components": "country:cl",  # Priorizar Chile
                    "types": "address"  # Solo direcciones
                },
                timeout=10.0
            )
            
            print(f"Autocompletado Google - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Autocompletado Google - Status API: {data.get('status')}")
                print(f"Autocompletado Google - Predictions count: {len(data.get('predictions', []))}")
                
                if data.get("status") == "OK" and data.get("predictions"):
                    results = []
                    predictions = data.get("predictions", [])[:5]  # Limitar a 5 resultados
                    
                    # Para cada predicción, obtener detalles (incluyendo coordenadas)
                    for prediction in predictions:
                        place_id = prediction.get("place_id")
                        description = prediction.get("description", "")
                        structured_formatting = prediction.get("structured_formatting", {})
                        main_text = structured_formatting.get("main_text", description)
                        secondary_text = structured_formatting.get("secondary_text", "")
                        
                        # Obtener detalles del lugar para tener coordenadas
                        try:
                            details_response = await client.get(
                                GOOGLE_PLACES_DETAILS_URL,
                                params={
                                    "place_id": place_id,
                                    "key": GOOGLE_MAPS_API_KEY,
                                    "language": "es",
                                    "fields": "geometry,formatted_address,address_components"
                                },
                                timeout=10.0
                            )
                            
                            if details_response.status_code == 200:
                                details_data = details_response.json()
                                if details_data.get("status") == "OK" and details_data.get("result"):
                                    place_result = details_data.get("result", {})
                                    geometry = place_result.get("geometry", {})
                                    location = geometry.get("location", {})
                                    
                                    address_components = place_result.get("address_components", [])
                                    city = ""
                                    country = ""
                                    
                                    for component in address_components:
                                        types = component.get("types", [])
                                        long_name = component.get("long_name", "")
                                        
                                        if "locality" in types or "administrative_area_level_2" in types:
                                            city = long_name
                                        elif "country" in types:
                                            country = long_name
                                    
                                    formatted_address = place_result.get("formatted_address", description)
                                    
                                    results.append({
                                        "text": description,
                                        "display_name": formatted_address,
                                        "address_line1": main_text,
                                        "address_line2": secondary_text,
                                        "city": city,
                                        "country": country,
                                        "lat": float(location.get("lat", 0)),
                                        "lng": float(location.get("lng", 0))
                                    })
                        except Exception as detail_err:
                            print(f"Error obteniendo detalles del lugar {place_id}: {str(detail_err)}")
                            # Si falla obtener detalles, usar solo la descripción
                            results.append({
                                "text": description,
                                "display_name": description,
                                "address_line1": main_text,
                                "address_line2": secondary_text,
                                "city": "",
                                "country": "",
                                "lat": 0,
                                "lng": 0
                            })
                    
                    print(f"Autocompletado Google - Results: {len(results)}")
                    return results
                elif data.get("status") == "ZERO_RESULTS":
                    print("Autocompletado Google - No se encontraron resultados")
                    return []
                elif data.get("status") == "OVER_QUERY_LIMIT":
                    print("Autocompletado Google - Límite de consultas excedido")
                    return []
                else:
                    print(f"Autocompletado Google - Error: {data.get('status')} - {data.get('error_message', '')}")
                    return []
            else:
                print(f"Autocompletado Google - Error HTTP: {response.status_code}")
                return []
        
    except Exception as e:
        print(f"Error en autocompletado Google: {str(e)}")
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
                try:
                    # Geocodificar dirección
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
    """Autocompletar direcciones mientras el usuario escribe"""
    if not query or len(query) < 3:
        return []
    
    try:
        results = await autocomplete_address(query)
        print(f"Autocompletado para '{query}': {len(results)} resultados")
        return results
    except Exception as e:
        print(f"Error en autocompletado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en autocompletado: {str(e)}")


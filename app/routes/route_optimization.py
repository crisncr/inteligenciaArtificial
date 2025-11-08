from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from pydantic import BaseModel
import httpx
import re
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

# Constantes de API Geoapify
GEOAPIFY_API_KEY = "6a3880920aad4e4283628f8cdfef0f3b"
GEOAPIFY_GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
GEOAPIFY_AUTOCOMPLETE_URL = "https://api.geoapify.com/v1/geocode/autocomplete"
GEOAPIFY_REVERSE_URL = "https://api.geoapify.com/v1/geocode/reverse"
GEOAPIFY_ROUTING_URL = "https://api.geoapify.com/v1/routing"

async def geocode_address(address: str) -> Dict:
    """Geocodificar dirección usando Geoapify - API con mejor precisión"""
    try:
        # Normalizar dirección
        normalized_address = normalize_address(address)
        
        # Geoapify API - Geocodificación
        async with httpx.AsyncClient() as client:
            response = await client.get(
                GEOAPIFY_GEOCODE_URL,
                params={
                    "text": normalized_address,
                    "apiKey": GEOAPIFY_API_KEY,
                    "limit": 1,
                    "format": "json"
                },
                timeout=15.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("features") and len(data["features"]) > 0:
                    feature = data["features"][0]
                    properties = feature.get("properties", {})
                    geometry = feature.get("geometry", {})
                    coordinates = geometry.get("coordinates", [])
                    
                    if len(coordinates) >= 2:
                        return {
                            "lat": float(coordinates[1]),
                            "lng": float(coordinates[0]),
                            "display_name": properties.get("formatted", normalized_address),
                            "address_line1": properties.get("address_line1", ""),
                            "address_line2": properties.get("address_line2", ""),
                            "city": properties.get("city", ""),
                            "country": properties.get("country", "")
                        }
            
            raise ValueError(f"No se pudo geocodificar la dirección: {address}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error al geocodificar '{address}': {str(e)}")

async def autocomplete_address(query: str) -> List[Dict]:
    """Autocompletar dirección usando Geoapify - API de autocompletado"""
    try:
        if not query or len(query) < 3:
            return []
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                GEOAPIFY_AUTOCOMPLETE_URL,
                params={
                    "text": query,
                    "apiKey": GEOAPIFY_API_KEY,
                    "limit": 5,
                    "format": "json"
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("features"):
                    results = []
                    for feature in data["features"]:
                        properties = feature.get("properties", {})
                        geometry = feature.get("geometry", {})
                        coordinates = geometry.get("coordinates", [])
                        
                        if len(coordinates) >= 2:
                            results.append({
                                "text": properties.get("formatted", properties.get("name", query)),
                                "display_name": properties.get("formatted", properties.get("name", query)),
                                "address_line1": properties.get("address_line1", ""),
                                "address_line2": properties.get("address_line2", ""),
                                "city": properties.get("city", ""),
                                "country": properties.get("country", ""),
                                "lat": float(coordinates[1]),
                                "lng": float(coordinates[0])
                            })
                    return results
        
        return []
    except Exception as e:
        print(f"Error en autocompletado: {str(e)}")
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
        for point in request.points:
            if point.address:
                # Geocodificar dirección
                coords = await geocode_address(point.address)
                points_with_coords.append({
                    "name": point.name,
                    "lat": coords["lat"],
                    "lng": coords["lng"],
                    "address": point.address,
                    "display_name": coords.get("display_name", point.address)
                })
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


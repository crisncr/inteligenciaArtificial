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

async def geocode_address(address: str) -> Dict:
    """Geocodificar dirección usando Nominatim (OpenStreetMap) - API gratuita"""
    try:
        # Normalizar dirección: limpiar espacios alrededor de comas
        normalized_address = normalize_address(address)
        
        # Si no tiene comas, probablemente necesita mejor formato
        if ',' not in normalized_address:
            # Intentar agregar país si no está presente
            if 'chile' not in normalized_address.lower():
                normalized_address = f"{normalized_address}, Chile"
        
        # Nominatim es gratuito y no requiere API key
        async with httpx.AsyncClient() as client:
            # Crear variaciones de la dirección para mejorar la búsqueda
            address_variations = [
                normalized_address,
                address,  # Original
            ]
            
            # Si la dirección tiene partes separadas por comas, crear variaciones
            if ',' in normalized_address:
                parts = [p.strip() for p in normalized_address.split(',')]
                if len(parts) >= 2:
                    # Agregar variaciones con y sin país explícito
                    address_variations.extend([
                        ", ".join(parts),
                        f"{parts[0]}, {parts[1]}, Chile" if len(parts) >= 2 else normalized_address,
                        f"{parts[0]}, Chile" if len(parts) >= 1 else normalized_address,
                    ])
            
            # Intentar con todas las variaciones
            responses = []
            seen_locations = set()  # Para evitar duplicados
            
            for addr_variant in address_variations:
                if not addr_variant or addr_variant in seen_locations:
                    continue
                try:
                    response = await client.get(
                        "https://nominatim.openstreetmap.org/search",
                        params={
                            "q": addr_variant,
                            "format": "json",
                            "limit": 5,  # Obtener más resultados para mejor matching
                            "addressdetails": 1,
                            "countrycodes": "cl",  # Priorizar Chile
                            "accept-language": "es"  # Español
                        },
                        headers={
                            "User-Agent": "RouteOptimizer/1.0"  # Nominatim requiere User-Agent
                        },
                        timeout=15.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            # Agregar solo ubicaciones únicas
                            for loc in data:
                                loc_key = (loc.get("lat"), loc.get("lon"))
                                if loc_key not in seen_locations:
                                    responses.append(loc)
                                    seen_locations.add(loc_key)
                                    
                            # Si encontramos resultados, usar el primero
                            if responses:
                                break
                except Exception:
                    continue
            
            if responses:
                # Seleccionar el mejor resultado (el primero suele ser el más relevante)
                location = responses[0]
                return {
                    "lat": float(location["lat"]),
                    "lng": float(location["lon"]),
                    "display_name": location.get("display_name", address)
                }
            
            # Si no funciona, intentar con formato más específico
            # Dividir por comas (incluso sin espacios) y reconstruir
            # Manejar casos como "calle,ciudad,país" o "calle, ciudad, país"
            parts = [p.strip() for p in re.split(r',\s*', address) if p.strip()]
            
            if len(parts) >= 2:
                # Formatear como "calle, ciudad, país"
                formatted = ", ".join(parts)
                
                # Asegurar que tenga país
                if len(parts) < 3:
                    if 'chile' not in formatted.lower():
                        formatted = f"{formatted}, Chile"
                
                # Intentar con diferentes variaciones
                variations = [
                    formatted,
                    f"{parts[0]}, {parts[1]}, Chile" if len(parts) >= 2 else formatted,
                    ", ".join(parts[:2]) + ", Rancagua, Chile" if 'rancagua' in address.lower() else None,
                ]
                
                for variation in variations:
                    if not variation:
                        continue
                    try:
                        async with httpx.AsyncClient() as client2:
                            response = await client2.get(
                                "https://nominatim.openstreetmap.org/search",
                                params={
                                    "q": variation,
                                    "format": "json",
                                    "limit": 1,
                                    "addressdetails": 1,
                                    "countrycodes": "cl",
                                    "accept-language": "es"
                                },
                                headers={
                                    "User-Agent": "RouteOptimizer/1.0"
                                },
                                timeout=15.0
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                if data and len(data) > 0:
                                    location = data[0]
                                    return {
                                        "lat": float(location["lat"]),
                                        "lng": float(location["lon"]),
                                        "display_name": location.get("display_name", address)
                                    }
                    except Exception:
                        continue
            
            raise ValueError(f"No se pudo geocodificar la dirección: {address}. Intenta con formato: 'Calle, Ciudad, País'")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error al geocodificar '{address}': {str(e)}")

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


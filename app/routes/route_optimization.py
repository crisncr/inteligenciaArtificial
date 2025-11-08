from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from pydantic import BaseModel
import httpx
from app.database import get_db
from app.auth import get_current_user
from app.models import User
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

async def geocode_address(address: str) -> Dict:
    """Geocodificar dirección usando Nominatim (OpenStreetMap) - API gratuita"""
    try:
        # Nominatim es gratuito y no requiere API key
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": address,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1
                },
                headers={
                    "User-Agent": "RouteOptimizer/1.0"  # Nominatim requiere User-Agent
                },
                timeout=10.0
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
            
            raise ValueError(f"No se pudo geocodificar la dirección: {address}")
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
        
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al optimizar ruta: {str(e)}")


from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict
from pydantic import BaseModel
from app.database import get_db
from app.auth import get_current_user
from app.models import User
from app.algorithms.route_optimizer import RouteOptimizer

router = APIRouter(prefix="/api/route-optimization", tags=["route-optimization"])

class PointInput(BaseModel):
    name: str
    lat: float
    lng: float

class RouteRequest(BaseModel):
    points: List[PointInput]
    algorithm: str = "astar"
    start_point: int = 0

@router.post("/optimize")
async def optimize_route(
    request: RouteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Optimizar ruta de distribución - Parte 2"""
    if current_user.plan == 'free':
        raise HTTPException(status_code=403, detail="Optimización de rutas disponible solo en planes Pro y Enterprise")
    
    max_points = 50 if current_user.plan == 'pro' else 1000
    
    if len(request.points) > max_points:
        raise HTTPException(status_code=403, detail=f"Plan {current_user.plan} permite máximo {max_points} puntos")
    
    if len(request.points) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 puntos")
    
    try:
        points = [{"name": p.name, "lat": p.lat, "lng": p.lng} for p in request.points]
        optimizer = RouteOptimizer(points)
        
        if request.algorithm == "astar":
            result = optimizer.astar(request.start_point)
        elif request.algorithm == "dijkstra":
            result = optimizer.dijkstra(request.start_point)
        elif request.algorithm == "tsp":
            result = optimizer.tsp_nearest_neighbor(request.start_point)
        else:
            raise HTTPException(status_code=400, detail="Algoritmo no válido. Use 'astar', 'dijkstra' o 'tsp'")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al optimizar ruta: {str(e)}")


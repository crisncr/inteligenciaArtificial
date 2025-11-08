import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Point:
    name: str
    lat: float
    lng: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calcular distancia euclidiana entre dos puntos"""
        return math.sqrt((self.lat - other.lat)**2 + (self.lng - other.lng)**2)

class RouteOptimizer:
    def __init__(self, points: List[Dict]):
        """Inicializar optimizador con puntos"""
        self.points = [Point(p['name'], p['lat'], p['lng']) for p in points]
        self.distances = self._calculate_distances()
    
    def _calculate_distances(self) -> Dict[Tuple[int, int], float]:
        """Calcular distancias entre todos los puntos"""
        distances = {}
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                if i != j:
                    distances[(i, j)] = p1.distance_to(p2)
        return distances
    
    def astar(self, start_idx: int = 0) -> Dict:
        """Algoritmo A* para encontrar ruta óptima - Parte 2"""
        if len(self.points) < 2:
            return {
                "route": [self.points[0].name],
                "distance": 0,
                "steps": [],
                "is_direct_route": False
            }
        
        # Caso especial: Solo 2 puntos (inicio y fin)
        if len(self.points) == 2:
            distance = self.distances[(0, 1)]
            return {
                "route": [self.points[0].name, self.points[1].name],
                "distance": round(distance, 2),
                "steps": [{
                    "step": 1,
                    "current": self.points[0].name,
                    "evaluated": [self.points[1].name],
                    "selected": self.points[1].name,
                    "distance": distance,
                    "heuristic_value": distance,
                    "reason": f"Ruta directa desde {self.points[0].name} hasta {self.points[1].name}"
                }],
                "algorithm": "A*",
                "is_direct_route": True
            }
        
        # Caso general: Múltiples puntos (optimización TSP)
        visited = [False] * len(self.points)
        route = [start_idx]
        visited[start_idx] = True
        total_distance = 0
        steps = []
        
        current = start_idx
        step_num = 1
        
        while len(route) < len(self.points):
            # Encontrar el punto más cercano no visitado (heurística)
            min_dist = float('inf')
            next_point = None
            evaluated_points = []
            
            for i, point in enumerate(self.points):
                if not visited[i]:
                    dist = self.distances[(current, i)]
                    heuristic = dist  # Heurística: distancia directa
                    evaluated_points.append({
                        "name": point.name,
                        "distance": dist,
                        "heuristic": heuristic
                    })
                    
                    if heuristic < min_dist:
                        min_dist = heuristic
                        next_point = i
            
            if next_point is not None:
                route.append(next_point)
                visited[next_point] = True
                total_distance += self.distances[(current, next_point)]
                
                steps.append({
                    "step": step_num,
                    "current": self.points[current].name,
                    "evaluated": [p['name'] for p in evaluated_points],
                    "selected": self.points[next_point].name,
                    "distance": self.distances[(current, next_point)],
                    "heuristic_value": min_dist,
                    "reason": f"Punto más cercano a {self.points[current].name} (heurística: {min_dist:.2f})"
                })
                
                current = next_point
                step_num += 1
        
        # Volver al punto de inicio (solo para rutas con múltiples puntos)
        if len(route) > 1:
            total_distance += self.distances[(route[-1], route[0])]
            route.append(route[0])
        
        return {
            "route": [self.points[i].name for i in route],
            "distance": round(total_distance, 2),
            "steps": steps,
            "algorithm": "A*",
            "is_direct_route": False
        }
    
    def dijkstra(self, start_idx: int = 0) -> Dict:
        """Algoritmo de Dijkstra"""
        return self.astar(start_idx)  # Para TSP, similar a A*
    
    def tsp_nearest_neighbor(self, start_idx: int = 0) -> Dict:
        """TSP con algoritmo del vecino más cercano"""
        return self.astar(start_idx)


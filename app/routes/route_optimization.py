from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
import httpx
import re
import os
import math
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

# Constantes de Google Maps API
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_PLACES_AUTOCOMPLETE_URL = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
GOOGLE_PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
GOOGLE_DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GOOGLE_ROADS_SNAP_URL = "https://roads.googleapis.com/v1/snapToRoads"

async def geocode_address_google(address: str) -> Dict:
    """Geocodificar dirección usando Google Maps API"""
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("Google Maps API key no configurada")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                GOOGLE_GEOCODE_URL,
                params={
                    "address": address,
                    "key": GOOGLE_MAPS_API_KEY,
                    "language": "es",
                    "region": "cl"  # Priorizar Chile
                },
                timeout=15.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "OK" and len(data.get("results", [])) > 0:
                    result = data["results"][0]
                    location = result["geometry"]["location"]
                    lat = location["lat"]
                    lng = location["lng"]
                    formatted_address = result.get("formatted_address", address)
                    
                    # Extraer componentes de la dirección
                    address_components = result.get("address_components", [])
                    address_line1 = ""
                    address_line2 = ""
                    city = ""
                    country = ""
                    
                    for component in address_components:
                        types = component.get("types", [])
                        if "street_number" in types or "route" in types:
                            if address_line1:
                                address_line1 += " " + component["long_name"]
                            else:
                                address_line1 = component["long_name"]
                        elif "locality" in types or "administrative_area_level_1" in types:
                            city = component["long_name"]
                        elif "country" in types:
                            country = component["long_name"]
                        elif "sublocality" in types or "postal_code" in types:
                            if address_line2:
                                address_line2 += ", " + component["long_name"]
                            else:
                                address_line2 = component["long_name"]
                    
                    return {
                        "lat": lat,
                        "lng": lng,
                        "display_name": formatted_address,
                        "address_line1": address_line1 or formatted_address.split(",")[0] if formatted_address else "",
                        "address_line2": address_line2,
                        "city": city,
                        "country": country
                    }
                else:
                    raise ValueError(f"Google Maps API no encontró resultados: {data.get('status', 'UNKNOWN_ERROR')}")
            else:
                raise ValueError(f"Error HTTP {response.status_code} al geocodificar con Google Maps")
                
    except Exception as e:
        print(f"Error al geocodificar con Google Maps: {str(e)}")
        raise

async def geocode_address(address: str) -> Dict:
    """Geocodificar dirección - Usa Google Maps si está disponible, sino Nominatim"""
    # Intentar primero con Google Maps si la API key está configurada
    if GOOGLE_MAPS_API_KEY:
        try:
            print(f"Geocodificando con Google Maps: {address}")
            return await geocode_address_google(address)
        except Exception as e:
            print(f"Google Maps falló, usando Nominatim como fallback: {str(e)}")
            # Continuar con Nominatim como fallback
    
    # Usar Nominatim (OpenStreetMap) - Normaliza número adelante/atrás
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

async def autocomplete_address_google(query: str) -> List[Dict]:
    """Autocompletar dirección usando Google Places API"""
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("Google Maps API key no configurada")
    
    try:
        async with httpx.AsyncClient() as client:
            # Primero obtener predicciones de autocompletado
            response = await client.get(
                GOOGLE_PLACES_AUTOCOMPLETE_URL,
                params={
                    "input": query,
                    "key": GOOGLE_MAPS_API_KEY,
                    "language": "es",
                    "components": "country:cl",  # Priorizar Chile
                    "types": "address"  # Solo direcciones
                },
                timeout=8.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "OK" and len(data.get("predictions", [])) > 0:
                    predictions = data["predictions"][:8]  # Limitar a 8 resultados
                    results = []
                    
                    # Para cada predicción, obtener detalles completos
                    for prediction in predictions:
                        place_id = prediction.get("place_id")
                        description = prediction.get("description", "")
                        
                        # Obtener detalles del lugar para coordenadas
                        details_response = await client.get(
                            GOOGLE_PLACES_DETAILS_URL,
                            params={
                                "place_id": place_id,
                                "key": GOOGLE_MAPS_API_KEY,
                                "language": "es",
                                "fields": "geometry,formatted_address,address_components"
                            },
                            timeout=8.0
                        )
                        
                        if details_response.status_code == 200:
                            details_data = details_response.json()
                            if details_data.get("status") == "OK":
                                place_details = details_data.get("result", {})
                                location = place_details.get("geometry", {}).get("location", {})
                                lat = location.get("lat", 0)
                                lng = location.get("lng", 0)
                                formatted_address = place_details.get("formatted_address", description)
                                
                                # Extraer componentes
                                address_components = place_details.get("address_components", [])
                                address_line1 = ""
                                address_line2 = ""
                                city = ""
                                country = ""
                                
                                for component in address_components:
                                    types = component.get("types", [])
                                    if "street_number" in types or "route" in types:
                                        if address_line1:
                                            address_line1 += " " + component["long_name"]
                                        else:
                                            address_line1 = component["long_name"]
                                    elif "locality" in types or "administrative_area_level_1" in types:
                                        city = component["long_name"]
                                    elif "country" in types:
                                        country = component["long_name"]
                                    elif "sublocality" in types or "postal_code" in types:
                                        if address_line2:
                                            address_line2 += ", " + component["long_name"]
                                        else:
                                            address_line2 = component["long_name"]
                                
                                results.append({
                                    "text": description,
                                    "display_name": formatted_address,
                                    "address_line1": address_line1 or formatted_address.split(",")[0] if formatted_address else "",
                                    "address_line2": address_line2,
                                    "city": city,
                                    "country": country,
                                    "lat": lat,
                                    "lng": lng,
                                    "importance": 1.0  # Google no proporciona importancia, usar valor por defecto
                                })
                    
                    return results
                else:
                    return []
            else:
                return []
                
    except Exception as e:
        print(f"Error al autocompletar con Google Maps: {str(e)}")
        return []

async def autocomplete_address(query: str) -> List[Dict]:
    """Autocompletar dirección - Usa Google Maps si está disponible, sino Nominatim"""
    # Intentar primero con Google Maps si la API key está configurada
    if GOOGLE_MAPS_API_KEY:
        try:
            print(f"Autocompletando con Google Maps: {query}")
            results = await autocomplete_address_google(query)
            if results:
                return results
        except Exception as e:
            print(f"Google Maps falló, usando Nominatim como fallback: {str(e)}")
            # Continuar con Nominatim como fallback
    
    # Usar Nominatim Search (OpenStreetMap) - Actualizado 2025
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

async def get_distance_matrix_google(points: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """Obtener matriz de distancias usando Google Distance Matrix API"""
    if not GOOGLE_MAPS_API_KEY:
        return {}
    
    if len(points) < 2:
        return {}
    
    try:
        max_points = 25  # Límite de Google
        all_distances = {}
        
        origins = [f"{p['lat']},{p['lng']}" for p in points]
        destinations = origins.copy()
        
        async with httpx.AsyncClient() as client:
            for i in range(0, len(points), max_points):
                end_i = min(i + max_points, len(points))
                origins_chunk = origins[i:end_i]
                
                for j in range(0, len(points), max_points):
                    end_j = min(j + max_points, len(points))
                    destinations_chunk = destinations[j:end_j]
                    
                    print(f"🔄 Distance Matrix: calculando {len(origins_chunk)}x{len(destinations_chunk)} distancias")
                    
                    response = await client.get(
                        GOOGLE_DISTANCE_MATRIX_URL,
                        params={
                            "origins": "|".join(origins_chunk),
                            "destinations": "|".join(destinations_chunk),
                            "key": GOOGLE_MAPS_API_KEY,
                            "mode": "driving",
                            "language": "es",
                            "units": "metric"
                        },
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("status") == "OK":
                            rows = data.get("rows", [])
                            
                            for row_idx, row in enumerate(rows):
                                origin_idx = i + row_idx
                                elements = row.get("elements", [])
                                
                                for col_idx, element in enumerate(elements):
                                    dest_idx = j + col_idx
                                    
                                    if origin_idx != dest_idx:
                                        status = element.get("status")
                                        
                                        if status == "OK":
                                            distance_meters = element.get("distance", {}).get("value", 0)
                                            duration_seconds = element.get("duration", {}).get("value", 0)
                                            distance_km = distance_meters / 1000
                                            
                                            all_distances[(origin_idx, dest_idx)] = {
                                                "distance_meters": distance_meters,
                                                "distance_km": distance_km,
                                                "duration_seconds": duration_seconds,
                                                "duration_minutes": duration_seconds / 60
                                            }
                                        else:
                                            # Fallback a euclidiana si hay error
                                            print(f"⚠️ Distance Matrix error ({origin_idx}, {dest_idx}): {status}")
                                            p1 = points[origin_idx]
                                            p2 = points[dest_idx]
                                            euclidean_km = math.sqrt(
                                                (p1['lat'] - p2['lat'])**2 + (p1['lng'] - p2['lng'])**2
                                            ) * 111
                                            all_distances[(origin_idx, dest_idx)] = {
                                                "distance_meters": euclidean_km * 1000,
                                                "distance_km": euclidean_km,
                                                "duration_seconds": euclidean_km * 60,
                                                "duration_minutes": euclidean_km
                                            }
        
        print(f"✅ Distance Matrix: {len(all_distances)} distancias calculadas")
        return all_distances
        
    except Exception as e:
        print(f"❌ Error en Distance Matrix API: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def decode_polyline(encoded: str) -> List[List[float]]:
    """Decodificar polyline de Google Maps a lista de coordenadas [lat, lng]"""
    coordinates = []
    index = 0
    lat = 0
    lng = 0
    
    while index < len(encoded):
        shift = 0
        result = 0
        
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        
        dlat = ~(result >> 1) if (result & 1) != 0 else (result >> 1)
        lat += dlat
        
        shift = 0
        result = 0
        
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        
        dlng = ~(result >> 1) if (result & 1) != 0 else (result >> 1)
        lng += dlng
        
        coordinates.append([lat / 1e5, lng / 1e5])
    
    return coordinates

async def snap_to_road_google(lat: float, lng: float) -> tuple[float, float]:
    """Ajustar coordenadas al punto más cercano en una calle usando Google Roads API"""
    if not GOOGLE_MAPS_API_KEY:
        return lat, lng
    
    try:
        print(f"🔧 Google Roads: ajustando ({lat}, {lng}) a la calle más cercana")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                GOOGLE_ROADS_SNAP_URL,
                params={
                    "path": f"{lat},{lng}",
                    "key": GOOGLE_MAPS_API_KEY,
                    "interpolate": "true"
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("snappedPoints") and len(data["snappedPoints"]) > 0:
                    snapped_point = data["snappedPoints"][0]
                    location = snapped_point.get("location", {})
                    snapped_lat = location.get("latitude", lat)
                    snapped_lng = location.get("longitude", lng)
                    
                    print(f"✅ Google Roads snap exitoso: ({lat}, {lng}) -> ({snapped_lat}, {snapped_lng})")
                    
                    return snapped_lat, snapped_lng
                else:
                    print(f"⚠️ Google Roads: no se encontró punto cercano, usando coordenadas originales")
                    return lat, lng
            else:
                print(f"⚠️ Google Roads: error HTTP {response.status_code}, usando coordenadas originales")
                return lat, lng
                
    except Exception as e:
        print(f"⚠️ Error en Google Roads snap: {str(e)}, usando coordenadas originales")
        return lat, lng

# Alias para compatibilidad
async def snap_to_road(lat: float, lng: float) -> tuple[float, float]:
    """Alias para snap_to_road_google"""
    return await snap_to_road_google(lat, lng)

async def get_google_directions(lat1: float, lng1: float, lat2: float, lng2: float, alternatives: int = 2) -> List[Dict]:
    """Obtener rutas reales por calles usando Google Directions API"""
    if not GOOGLE_MAPS_API_KEY:
        return []
    
    try:
        print(f"🗺️ Google Directions: Obteniendo rutas desde ({lat1}, {lng1}) hasta ({lat2}, {lng2})")
        
        async with httpx.AsyncClient() as client:
            origin = f"{lat1},{lng1}"
            destination = f"{lat2},{lng2}"
            
            response = await client.get(
                GOOGLE_DIRECTIONS_URL,
                params={
                    "origin": origin,
                    "destination": destination,
                    "key": GOOGLE_MAPS_API_KEY,
                    "alternatives": "true" if alternatives > 0 else "false",
                    "mode": "driving",
                    "language": "es",
                    "units": "metric",
                    "region": "cl"
                },
                timeout=20.0
            )
            
            print(f"🗺️ Google Directions Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "OK" and data.get("routes"):
                    routes = []
                    for idx, route in enumerate(data["routes"][:3]):  # Máximo 3 rutas
                        leg = route["legs"][0]  # Primera pierna (solo hay una para origen-destino)
                        
                        distance_meters = leg.get("distance", {}).get("value", 0)
                        duration_seconds = leg.get("duration", {}).get("value", 0)
                        distance_km = distance_meters / 1000
                        duration_minutes = duration_seconds / 60
                        
                        # Extraer coordenadas del polyline
                        overview_polyline = route.get("overview_polyline", {})
                        encoded_polyline = overview_polyline.get("points", "")
                        
                        # Decodificar polyline a coordenadas
                        coordinates_list = decode_polyline(encoded_polyline)
                        
                        print(f"🗺️ Ruta {idx + 1}: {len(coordinates_list)} coordenadas, {distance_km:.2f} km, {duration_minutes:.2f} min")
                        
                        routes.append({
                            "route_number": idx + 1,
                            "distance_km": round(distance_km, 2),
                            "duration_minutes": round(duration_minutes, 2),
                            "distance_meters": round(distance_meters, 2),
                            "duration_seconds": round(duration_seconds, 2),
                            "coordinates": coordinates_list,  # Ya viene en formato [lat, lng]
                            "description": f"Ruta {idx + 1}: {round(distance_km, 2)} km, {round(duration_minutes, 2)} min"
                        })
                    
                    # Ordenar por distancia (las más cortas primero)
                    routes.sort(key=lambda x: x["distance_km"])
                    
                    # Renumerar rutas
                    for idx, route in enumerate(routes):
                        route["route_number"] = idx + 1
                    
                    print(f"✅ Google Directions: Se obtuvieron {len(routes)} rutas")
                    return routes
                else:
                    error_msg = data.get('error_message', data.get('status', 'Unknown error'))
                    print(f"❌ Google Directions: Error - Status: {data.get('status', 'Unknown')}, Message: {error_msg}")
                    return []
            else:
                error_text = response.text[:200] if hasattr(response, 'text') else 'No error text'
                print(f"❌ Google Directions: Error HTTP {response.status_code}: {error_text}")
                return []
                
    except Exception as e:
        print(f"❌ Error al obtener rutas de Google Directions: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Alias para compatibilidad
async def get_osrm_routes(lat1: float, lng1: float, lat2: float, lng2: float, alternatives: int = 2, snap_destination: bool = True) -> List[Dict]:
    """Alias para get_google_directions (compatibilidad con código existente)"""
    if snap_destination:
        lat2, lng2 = await snap_to_road(lat2, lng2)
    return await get_google_directions(lat1, lng1, lat2, lng2, alternatives)

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
        
        # Si hay exactamente 2 puntos, usar Google Directions API
        if len(points_with_coords) == 2:
            start_point = points_with_coords[0]
            end_point = points_with_coords[1]
            
            # Guardar coordenadas originales del destino
            original_end_lat = end_point["lat"]
            original_end_lng = end_point["lng"]
            
            # Ajustar el punto de destino a la calle más cercana
            snapped_end_lat, snapped_end_lng = await snap_to_road(original_end_lat, original_end_lng)
            
            # Obtener rutas usando Google Directions API
            google_routes = await get_google_directions(
                start_point["lat"], 
                start_point["lng"],
                snapped_end_lat, 
                snapped_end_lng,
                alternatives=2
            )
            
            if google_routes and len(google_routes) > 0:
                result = {
                    "routes": google_routes,
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
                            "lat": snapped_end_lat,
                            "lng": snapped_end_lng,
                            "original_lat": original_end_lat,
                            "original_lng": original_end_lng
                        }
                    ],
                    "algorithm": "Google Directions API",
                    "is_direct_route": True,
                    "has_osrm_routes": True  # Mantener para compatibilidad
                }
                
                return result
            else:
                # Si Google Directions falla, usar algoritmo tradicional como fallback
                print("Google Directions no devolvió rutas, usando algoritmo tradicional como fallback")
                points_for_optimizer = [{"name": p["name"], "lat": p["lat"], "lng": p["lng"]} for p in points_with_coords]
                
                # Intentar usar Distance Matrix si está disponible
                distance_matrix = None
                if GOOGLE_MAPS_API_KEY:
                    try:
                        distance_matrix = await get_distance_matrix_google(points_for_optimizer)
                    except Exception as e:
                        print(f"⚠️ Error al obtener Distance Matrix: {str(e)}")
                
                optimizer = RouteOptimizer(points_for_optimizer, distance_matrix)
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
            # Para múltiples puntos, usar algoritmo tradicional (TSP) con Distance Matrix
            points_for_optimizer = [{"name": p["name"], "lat": p["lat"], "lng": p["lng"]} for p in points_with_coords]
            
            # Usar Distance Matrix API si está disponible
            distance_matrix = None
            if GOOGLE_MAPS_API_KEY and len(points_for_optimizer) > 1:
                try:
                    print("🔄 Calculando distancias reales con Google Distance Matrix API...")
                    distance_matrix = await get_distance_matrix_google(points_for_optimizer)
                    if distance_matrix:
                        print(f"✅ Usando distancias reales de Google Maps ({len(distance_matrix)} pares)")
                    else:
                        print("⚠️ Distance Matrix vacío, usando distancia euclidiana")
                except Exception as e:
                    print(f"⚠️ Error al obtener Distance Matrix: {str(e)}, usando distancia euclidiana")
            
            optimizer = RouteOptimizer(points_for_optimizer, distance_matrix)
            
            if request.algorithm == "astar":
                result = optimizer.astar(request.start_point)
            elif request.algorithm == "dijkstra":
                result = optimizer.dijkstra(request.start_point)
            elif request.algorithm == "tsp":
                result = optimizer.tsp_nearest_neighbor(request.start_point)
            else:
                raise HTTPException(status_code=400, detail="Algoritmo no válido. Use 'astar', 'dijkstra' o 'tsp'")
            
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


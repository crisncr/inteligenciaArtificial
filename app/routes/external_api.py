from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import httpx
import json
from app.database import get_db
from app.models import User, ExternalAPI, Analysis
from app.schemas import (
    ExternalAPICreate, ExternalAPIResponse, ExternalAPIUpdate,
    ExternalAPITest, ExternalAPIAnalyze
)
from app.auth import get_current_user
from app.sentiment import analyze_sentiment

router = APIRouter(prefix="/api/external-apis", tags=["external-apis"])

def extract_comments_from_response(data):
    """Extrae comentarios de diferentes formatos de respuesta de API"""
    comments = []
    
    if isinstance(data, list):
        # Si es un array directo
        for item in data:
            if isinstance(item, str):
                comments.append(item)
            elif isinstance(item, dict):
                # Buscar campos comunes de comentarios
                for key in ['comment', 'text', 'content', 'message', 'body', 'description']:
                    if key in item and isinstance(item[key], str):
                        comments.append(item[key])
                        break
    elif isinstance(data, dict):
        # Si es un objeto, buscar arrays comunes
        for key in ['comments', 'data', 'items', 'results', 'posts', 'messages']:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, str):
                        comments.append(item)
                    elif isinstance(item, dict):
                        for subkey in ['comment', 'text', 'content', 'message', 'body', 'description']:
                            if subkey in item and isinstance(item[subkey], str):
                                comments.append(item[subkey])
                                break
                break
        # Si no hay array, buscar comentario directo
        if not comments:
            for key in ['comment', 'text', 'content', 'message', 'body', 'description']:
                if key in data and isinstance(data[key], str):
                    comments.append(data[key])
                    break
    
    return comments

async def fetch_external_api(external_api: ExternalAPI):
    """Obtiene datos de la API externa"""
    url = f"{external_api.api_url.rstrip('/')}/{external_api.endpoint.lstrip('/')}"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Agregar headers personalizados
    if external_api.headers:
        headers.update(external_api.headers)
    
    # Agregar autenticación
    if external_api.auth_type == "bearer" and external_api.auth_token:
        headers["Authorization"] = f"Bearer {external_api.auth_token}"
    elif external_api.auth_type == "api_key" and external_api.auth_token:
        headers["X-API-Key"] = external_api.auth_token
    elif external_api.auth_type == "basic" and external_api.auth_token:
        headers["Authorization"] = f"Basic {external_api.auth_token}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        if external_api.method.upper() == "GET":
            response = await client.get(url, headers=headers)
        elif external_api.method.upper() == "POST":
            response = await client.post(url, headers=headers)
        else:
            raise ValueError(f"Método {external_api.method} no soportado")
        
        response.raise_for_status()
        return response.json()

@router.post("", response_model=ExternalAPIResponse, status_code=status.HTTP_201_CREATED)
async def create_external_api(
    api_data: ExternalAPICreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Crear una nueva configuración de API externa"""
    new_api = ExternalAPI(
        user_id=current_user.id,
        name=api_data.name,
        api_url=api_data.api_url,
        endpoint=api_data.endpoint,
        method=api_data.method,
        headers=api_data.headers,
        auth_type=api_data.auth_type,
        auth_token=api_data.auth_token,
        active=api_data.active
    )
    
    db.add(new_api)
    db.commit()
    db.refresh(new_api)
    
    return new_api

@router.get("", response_model=List[ExternalAPIResponse])
async def get_external_apis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener todas las APIs externas del usuario"""
    apis = db.query(ExternalAPI).filter(
        ExternalAPI.user_id == current_user.id
    ).order_by(ExternalAPI.created_at.desc()).all()
    
    return apis

@router.get("/{api_id}", response_model=ExternalAPIResponse)
async def get_external_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener una API externa específica"""
    api = db.query(ExternalAPI).filter(
        ExternalAPI.id == api_id,
        ExternalAPI.user_id == current_user.id
    ).first()
    
    if not api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API externa no encontrada"
        )
    
    return api

@router.put("/{api_id}", response_model=ExternalAPIResponse)
async def update_external_api(
    api_id: int,
    api_data: ExternalAPIUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Actualizar una API externa"""
    api = db.query(ExternalAPI).filter(
        ExternalAPI.id == api_id,
        ExternalAPI.user_id == current_user.id
    ).first()
    
    if not api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API externa no encontrada"
        )
    
    # Actualizar campos
    if api_data.name is not None:
        api.name = api_data.name
    if api_data.api_url is not None:
        api.api_url = api_data.api_url
    if api_data.endpoint is not None:
        api.endpoint = api_data.endpoint
    if api_data.method is not None:
        api.method = api_data.method
    if api_data.headers is not None:
        api.headers = api_data.headers
    if api_data.auth_type is not None:
        api.auth_type = api_data.auth_type
    if api_data.auth_token is not None:
        api.auth_token = api_data.auth_token
    if api_data.active is not None:
        api.active = api_data.active
    
    db.commit()
    db.refresh(api)
    
    return api

@router.delete("/{api_id}")
async def delete_external_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Eliminar una API externa"""
    api = db.query(ExternalAPI).filter(
        ExternalAPI.id == api_id,
        ExternalAPI.user_id == current_user.id
    ).first()
    
    if not api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API externa no encontrada"
        )
    
    db.delete(api)
    db.commit()
    
    return {"message": "API externa eliminada correctamente"}

@router.post("/{api_id}/test", response_model=ExternalAPITest)
async def test_external_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Probar conexión a la API externa"""
    api = db.query(ExternalAPI).filter(
        ExternalAPI.id == api_id,
        ExternalAPI.user_id == current_user.id
    ).first()
    
    if not api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API externa no encontrada"
        )
    
    try:
        data = await fetch_external_api(api)
        comments = extract_comments_from_response(data)
        
        return ExternalAPITest(
            success=True,
            message=f"Conexión exitosa. Se encontraron {len(comments)} comentarios.",
            data={"comments_count": len(comments), "sample_data": data}
        )
    except httpx.HTTPStatusError as e:
        return ExternalAPITest(
            success=False,
            message=f"Error HTTP {e.response.status_code}: {e.response.text[:200]}",
            data=None
        )
    except httpx.RequestError as e:
        return ExternalAPITest(
            success=False,
            message=f"Error de conexión: {str(e)}",
            data=None
        )
    except Exception as e:
        return ExternalAPITest(
            success=False,
            message=f"Error: {str(e)}",
            data=None
        )

@router.post("/{api_id}/analyze", response_model=ExternalAPIAnalyze)
async def analyze_external_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener comentarios de la API externa y analizarlos"""
    api = db.query(ExternalAPI).filter(
        ExternalAPI.id == api_id,
        ExternalAPI.user_id == current_user.id,
        ExternalAPI.active == True
    ).first()
    
    if not api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API externa no encontrada o inactiva"
        )
    
    try:
        # Obtener datos de la API externa
        data = await fetch_external_api(api)
        
        # Extraer comentarios
        comments = extract_comments_from_response(data)
        
        if not comments:
            return ExternalAPIAnalyze(
                comments_count=0,
                analyses_created=0,
                errors=["No se encontraron comentarios en la respuesta de la API"]
            )
        
        # Analizar cada comentario
        analyses_created = 0
        errors = []
        
        for comment in comments:
            try:
                # Analizar sentimiento usando SOLO red neuronal LSTM
                # analyze_sentiment() ahora usa exclusivamente red neuronal
                result = analyze_sentiment(comment)
                
                # Verificar que se usó red neuronal
                if result.get('method') != 'neural_network':
                    print(f"⚠️ Advertencia: El análisis no marcó método neuronal, pero debería ser neuronal")
                
                # Guardar análisis en BD
                # source='api_external' indica que viene de API externa
                # Todos estos análisis son neuronales
                new_analysis = Analysis(
                    user_id=current_user.id,
                    text=comment,
                    sentiment=result["sentiment"],
                    score=result["score"],
                    emoji=result["emoji"],
                    source="api_external",  # Marca que viene de API externa
                    external_api_id=api.id
                )
                
                db.add(new_analysis)
                analyses_created += 1
            except Exception as e:
                errors.append(f"Error analizando comentario: {str(e)}")
        
        db.commit()
        
        return ExternalAPIAnalyze(
            comments_count=len(comments),
            analyses_created=analyses_created,
            errors=errors if errors else None
        )
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error HTTP {e.response.status_code} al conectar con la API externa"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error de conexión: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


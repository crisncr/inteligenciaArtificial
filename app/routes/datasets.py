from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import csv
import io
from pydantic import BaseModel
from app.database import get_db
from app.auth import get_current_user
from app.models import User
from app.ml_models.sentiment_nn import SentimentNeuralNetwork

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

class SearchRequest(BaseModel):
    query: str
    texts: List[str]

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cargar dataset de comentarios - Parte 1"""
    # Verificar límite según plan
    if current_user.plan == 'free':
        max_rows = 100
    else:
        max_rows = 10000
    
    contents = await file.read()
    
    try:
        # Intentar leer como CSV
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except:
        try:
            # Intentar leer como JSON
            df = pd.read_json(io.StringIO(contents.decode('utf-8')))
        except:
            raise HTTPException(status_code=400, detail="Formato de archivo inválido. Debe ser CSV o JSON")
    
    # Buscar columna de texto (case-insensitive y normalizar espacios)
    text_column = None
    
    # Guardar nombres originales de columnas para el mensaje de error
    original_columns = df.columns.tolist()
    
    # Crear mapeo entre nombres normalizados y originales
    column_mapping = {col.strip().lower(): col for col in original_columns}
    
    # Lista de nombres posibles (en minúsculas)
    possible_names = ['texto', 'text', 'comment', 'comentario', 'review', 'reseña', 'mensaje', 'message', 'opinion', 'opinión', 'contenido', 'content', 'descripcion', 'description', 'comentarios']
    
    # Buscar columna por nombre normalizado
    for normalized_name in possible_names:
        if normalized_name in column_mapping:
            text_column = column_mapping[normalized_name]  # Usar nombre original
            break
    
    # Si no se encontró, intentar usar la primera columna de tipo string
    if text_column is None:
        for col in original_columns:
            if df[col].dtype == 'object':  # Tipo string/object
                text_column = col
                break
    
    # Si aún no se encontró, usar la primera columna disponible
    if text_column is None and len(original_columns) > 0:
        text_column = original_columns[0]
    
    if text_column is None:
        raise HTTPException(
            status_code=400, 
            detail=f"El archivo debe contener una columna de texto. Columnas encontradas: {', '.join(original_columns)}. "
                   f"Por favor, asegúrate de que el archivo tenga una columna con texto (por ejemplo: 'texto', 'text', 'comment', 'review', etc.)"
        )
    
    print(f"✅ Columna de texto detectada: '{text_column}'")
    
    texts = df[text_column].dropna().astype(str).tolist()[:max_rows]
    
    return {
        "total": len(texts),
        "texts": texts,
        "message": f"Dataset cargado: {len(texts)} comentarios"
    }

@router.post("/analyze-batch")
async def analyze_batch(
    texts: List[str],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analizar múltiples textos con red neuronal - Parte 1"""
    if current_user.plan == 'free' and len(texts) > 100:
        raise HTTPException(status_code=403, detail="Plan free permite máximo 100 comentarios")
    
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="No hay textos para analizar")
    
    try:
        # Usar el modelo global desde sentiment.py para evitar reentrenarlo
        from app.sentiment import _get_or_create_model
        model = _get_or_create_model()
        
        if not model or not model.is_trained:
            raise HTTPException(status_code=500, detail="El modelo de red neuronal no está disponible. Por favor, intenta de nuevo en unos momentos.")
        
        results = model.predict(texts)
        
        positive_count = sum(1 for r in results if r['sentiment'] == 'positivo')
        negative_count = sum(1 for r in results if r['sentiment'] == 'negativo')
        neutral_count = sum(1 for r in results if r['sentiment'] == 'neutral')
        
        return {
            "total": len(results),
            "results": results,
            "summary": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "positive_percent": round((positive_count / len(results)) * 100, 2),
                "negative_percent": round((negative_count / len(results)) * 100, 2),
                "neutral_percent": round((neutral_count / len(results)) * 100, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar textos: {str(e)}")

@router.post("/search")
async def search_texts(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Búsqueda de texto en comentarios - Parte 1"""
    if not request.query or not request.texts:
        return {
            "query": request.query,
            "total_matches": 0,
            "results": []
        }
    
    query_lower = request.query.lower()
    results = []
    
    for i, text in enumerate(request.texts):
        if query_lower in text.lower():
            matches = text.lower().count(query_lower)
            results.append({
                "index": i,
                "text": text,
                "matches": matches
            })
    
    return {
        "query": request.query,
        "total_matches": len(results),
        "results": results
    }


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
    
    # Buscar columna de texto
    text_column = None
    for col in ['texto', 'text', 'comment', 'comentario', 'review', 'reseña']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        raise HTTPException(status_code=400, detail="El archivo debe contener una columna 'texto', 'text', 'comment', 'comentario', 'review' o 'reseña'")
    
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
        model = SentimentNeuralNetwork()
        model.load_model()
        
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


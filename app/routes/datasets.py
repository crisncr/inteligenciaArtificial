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
    
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo: {str(e)}")
    
    try:
        # Intentar leer como CSV
        try:
            # Intentar diferentes codificaciones comunes
            try:
                df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(io.StringIO(contents.decode('latin-1')))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(contents.decode('utf-8', errors='ignore')))
        except Exception as csv_error:
            # Si falla CSV, intentar JSON
            try:
                df = pd.read_json(io.StringIO(contents.decode('utf-8')))
            except Exception as json_error:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Formato de archivo inválido. Debe ser CSV o JSON. Error CSV: {str(csv_error)}, Error JSON: {str(json_error)}"
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")
    
    # Buscar columnas de texto de forma flexible
    # No requiere nombres específicos, solo busca columnas que contengan texto
    text_columns = []
    
    # Guardar nombres originales de columnas
    original_columns = df.columns.tolist()
    
    # Buscar todas las columnas que contengan texto (tipo object/string)
    for col in original_columns:
        if df[col].dtype == 'object':  # Tipo string/object
            # Verificar que realmente contenga texto (no solo números o vacíos)
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                # Si al menos el 50% de los valores no vacíos son strings con más de 2 caracteres
                text_count = sum(1 for val in sample_values if isinstance(val, str) and len(str(val).strip()) > 2)
                if text_count > len(sample_values) * 0.3:  # Al menos 30% son textos
                    text_columns.append(col)
    
    # Si no se encontraron columnas de texto, usar todas las columnas de tipo object
    if not text_columns:
        for col in original_columns:
            if df[col].dtype == 'object':
                text_columns.append(col)
    
    # Si aún no hay columnas, usar todas las columnas disponibles
    if not text_columns:
        text_columns = original_columns
    
    if not text_columns:
        raise HTTPException(
            status_code=400, 
            detail=f"El archivo no contiene columnas con texto. Columnas encontradas: {', '.join(original_columns)}"
        )
    
    # Usar la primera columna de texto encontrada (o combinar todas si hay múltiples)
    # Por simplicidad, usamos la primera columna con más datos
    text_column = text_columns[0]
    if len(text_columns) > 1:
        # Si hay múltiples columnas de texto, usar la que tiene más datos no vacíos
        text_column = max(text_columns, key=lambda col: df[col].notna().sum())
    
    print(f"✅ Columna de texto detectada: '{text_column}' (de {len(text_columns)} columnas de texto encontradas)")
    
    try:
        # Extraer textos, eliminar valores vacíos y limitar
        texts = df[text_column].dropna().astype(str).tolist()
        # Filtrar textos vacíos o muy cortos (menos de 2 caracteres)
        # También corregir encoding común de Excel
        cleaned_texts = []
        for t in texts:
            if t and t.strip() and len(t.strip()) >= 2:
                # Corregir encoding común de Excel (UTF-8 mal interpretado como Latin-1)
                text = t.strip()
                # Intentar decodificar y recodificar si hay problemas de encoding
                try:
                    # Si el texto tiene caracteres mal codificados, intentar corregirlos
                    if 'Ã' in text or 'â€™' in text:
                        # Intentar corregir encoding común
                        text = text.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                except:
                    pass
                cleaned_texts.append(text)
        texts = cleaned_texts[:max_rows]
        
        if len(texts) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No se encontraron textos válidos en la columna '{text_column}'. Verifica que el archivo contenga datos de texto."
            )
        
        return {
            "total": len(texts),
            "texts": texts,
            "message": f"Dataset cargado exitosamente: {len(texts)} comentarios",
            "column": text_column
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error al procesar los textos del dataset: {str(e)}"
        )

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


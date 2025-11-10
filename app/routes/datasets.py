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
    
    # Guardar el contenido original para debugging
    df = None
    encoding_used = None
    
    try:
        # Intentar leer como CSV con diferentes codificaciones
        encodings_to_try = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                decoded_content = contents.decode(encoding)
                # Intentar leer como CSV
                df = pd.read_csv(io.StringIO(decoded_content))
                encoding_used = encoding
                print(f"✅ CSV leído correctamente con encoding: {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # Si el encoding funciona pero hay otro error, guardar el error
                if encoding == encodings_to_try[-1]:  # Último encoding
                    raise e
                continue
        
        # Si no se pudo leer como CSV, intentar JSON
        if df is None:
            try:
                for encoding in encodings_to_try:
                    try:
                        decoded_content = contents.decode(encoding)
                        df = pd.read_json(io.StringIO(decoded_content))
                        encoding_used = encoding
                        print(f"✅ JSON leído correctamente con encoding: {encoding}")
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception:
                        continue
            except Exception as json_error:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Formato de archivo inválido. Debe ser CSV o JSON. Error: {str(json_error)}"
                )
        
        if df is None:
            raise HTTPException(
                status_code=400,
                detail="No se pudo leer el archivo con ninguna codificación estándar. Asegúrate de que el archivo esté en formato UTF-8 o Latin-1."
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
        
        # Filtrar textos vacíos o muy cortos y limpiar encoding
        cleaned_texts = []
        for idx, t in enumerate(texts):
            if not t or not isinstance(t, str):
                continue
                
            text = str(t).strip()
            
            # Filtrar textos muy cortos
            if len(text) < 2:
                continue
            
            # Limpiar texto: corregir encoding común de Excel/CSV
            # Problemas comunes: UTF-8 guardado pero leído como Latin-1
            original_text = text
            
            # Detectar y corregir problemas de encoding comunes
            if 'Ã' in text:
                try:
                    # Intentar corregir: texto UTF-8 mal leído como Latin-1
                    text = text.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                except:
                    pass
            
            # Limpiar caracteres de control y BOM
            text = text.replace('\ufeff', '').replace('\x00', '').strip()
            
            # Solo agregar si después de limpiar aún tiene contenido
            if len(text) >= 2:
                cleaned_texts.append(text)
                # Log solo para los primeros 3 textos para debugging
                if idx < 3:
                    print(f"🔍 [DEBUG] Texto {idx+1} original: {original_text[:50]}...")
                    print(f"🔍 [DEBUG] Texto {idx+1} limpiado: {text[:50]}...")
        
        texts = cleaned_texts[:max_rows]
        
        print(f"✅ Total de textos procesados: {len(texts)} (encoding usado: {encoding_used})")
        
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
        
        # Log de los primeros textos que se van a analizar
        print(f"🔍 [DEBUG] Analizando {len(texts)} textos")
        for i, text in enumerate(texts[:3]):
            print(f"🔍 [DEBUG] Texto {i+1} a analizar: {text[:80]}...")
        
        # Analizar textos
        results = model.predict(texts)
        
        # Log de los primeros resultados
        for i, result in enumerate(results[:3]):
            print(f"🔍 [DEBUG] Resultado {i+1}: texto='{result.get('text', '')[:50]}...', sentiment={result.get('sentiment')}, confidence={result.get('confidence', 0):.3f}")
        
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


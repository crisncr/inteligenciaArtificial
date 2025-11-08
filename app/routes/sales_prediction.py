from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import pandas as pd
import io
from typing import List
from pydantic import BaseModel
from app.database import get_db
from app.auth import get_current_user
from app.models import User
from app.ml_models.sales_predictor import SalesPredictor

router = APIRouter(prefix="/api/sales-prediction", tags=["sales-prediction"])


class PredictionRequest(BaseModel):
    region: str
    model_type: str = "linear_regression"
    start_date: str
    days: int = 30

# Almacenar datos cargados por usuario (en producción usar BD)
user_data_storage = {}

@router.post("/upload")
async def upload_sales_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cargar datos históricos de ventas - Parte 3"""
    if current_user.plan != 'enterprise':
        raise HTTPException(status_code=403, detail="Predicción de ventas disponible solo en plan Enterprise")
    
    contents = await file.read()
    
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except:
        raise HTTPException(status_code=400, detail="Formato de archivo inválido. Debe ser CSV")
    
    required_columns = ['fecha', 'region', 'ventas']
    if not all(col in df.columns for col in required_columns):
        raise HTTPException(status_code=400, detail=f"El archivo debe contener las columnas: {', '.join(required_columns)}")
    
    # Almacenar datos del usuario (en producción usar BD)
    user_data_storage[current_user.id] = df.to_dict('records')
    
    return {
        "total": len(df),
        "regions": df['region'].unique().tolist(),
        "date_range": {
            "start": str(df['fecha'].min()),
            "end": str(df['fecha'].max())
        }
    }

@router.post("/train")
async def train_model(
    file: UploadFile = File(...),
    region: str = None,
    model_type: str = "linear_regression",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Entrenar modelo de predicción - Parte 3"""
    if current_user.plan != 'enterprise':
        raise HTTPException(status_code=403, detail="Predicción de ventas disponible solo en plan Enterprise")
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    required_columns = ['fecha', 'region', 'ventas']
    if not all(col in df.columns for col in required_columns):
        raise HTTPException(status_code=400, detail=f"El archivo debe contener las columnas: {', '.join(required_columns)}")
    
    try:
        predictor = SalesPredictor()
        
        if model_type == "linear_regression":
            result = predictor.train_linear_regression(df, region)
        elif model_type == "neural_network":
            result = predictor.train_neural_network(df, region, epochs=50)
        else:
            raise HTTPException(status_code=400, detail="Tipo de modelo no válido. Use 'linear_regression' o 'neural_network'")
        
        # Almacenar predictor entrenado (en producción usar BD o caché)
        if current_user.id not in user_data_storage:
            user_data_storage[current_user.id] = {}
        user_data_storage[current_user.id]['predictor'] = predictor
        user_data_storage[current_user.id]['data'] = df.to_dict('records')
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar modelo: {str(e)}")

@router.post("/predict")
async def predict_sales(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predecir ventas futuras - Parte 3"""
    if current_user.plan != 'enterprise':
        raise HTTPException(status_code=403, detail="Predicción de ventas disponible solo en plan Enterprise")
    
    try:
        # Obtener predictor entrenado del usuario
        if current_user.id not in user_data_storage or 'predictor' not in user_data_storage[current_user.id]:
            raise HTTPException(status_code=400, detail="Primero debe entrenar el modelo usando /train")
        
        predictor = user_data_storage[current_user.id]['predictor']
        predictions = predictor.predict(request.start_date, request.days, request.region)
        
        total_predicted = sum(p['ventas_predichas'] for p in predictions)
        average_daily = total_predicted / len(predictions) if predictions else 0
        
        return {
            "region": request.region,
            "model_type": request.model_type,
            "predictions": predictions,
            "summary": {
                "total_predicted": round(total_predicted, 2),
                "average_daily": round(average_daily, 2),
                "days": len(predictions)
            },
            "process_explanation": {
                "step1": "Recolección de datos históricos de ventas",
                "step2": "Preprocesamiento (normalización, codificación de fechas)",
                "step3": "División en conjunto de entrenamiento y prueba",
                "step4": f"Entrenamiento del modelo ({request.model_type})",
                "step5": "Evaluación del modelo (R², MSE)",
                "step6": "Predicción de ventas futuras",
                "step7": "Visualización de resultados"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir ventas: {str(e)}")


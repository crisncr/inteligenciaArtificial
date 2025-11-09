from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from typing import List
from app.database import get_db
from app.models import User, Analysis
from app.schemas import AnalysisCreate, AnalysisResponse, AnalysisBatch, AnalysisBatchResponse
from app.auth import get_current_user
from app.sentiment import analyze_sentiment

router = APIRouter(prefix="/api/analyses", tags=["analyses"])

def check_analysis_limit(user: User, db: Session) -> bool:
    """Verifica si el usuario puede realizar más análisis según su plan"""
    if user.plan == "pro" or user.plan == "enterprise":
        return True  # Sin límite
    
    if user.plan == "free":
        # Contar análisis del día actual
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_analyses = db.query(Analysis).filter(
            and_(
                Analysis.user_id == user.id,
                Analysis.created_at >= today_start
            )
        ).count()
        
        return today_analyses < 10  # Límite de 10 por día para free
    
    return False

@router.post("", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis(
    analysis_data: AnalysisCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Crear un nuevo análisis"""
    print(f"🔍 [DEBUG] create_analysis() llamado para usuario {current_user.id}")
    print(f"🔍 [DEBUG] Texto: '{analysis_data.text[:50]}...'")
    
    # Verificar límite
    if not check_analysis_limit(current_user, db):
        print("❌ [DEBUG] Límite de análisis alcanzado")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Has alcanzado el límite de análisis para tu plan. Actualiza tu plan para continuar."
        )
    
    # Analizar sentimiento usando SOLO red neuronal LSTM
    # analyze_sentiment() ahora usa exclusivamente red neuronal
    print("🔍 [DEBUG] Llamando a analyze_sentiment()...")
    try:
        result = analyze_sentiment(analysis_data.text)
        print(f"✅ [DEBUG] Análisis completado: sentiment={result.get('sentiment')}")
    except Exception as e:
        print(f"❌ [DEBUG] Error en analyze_sentiment: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al analizar sentimiento: {str(e)}"
        )
    
    # Guardar en BD con source='manual'
    # Todos los análisis, incluso los manuales, son neuronales
    new_analysis = Analysis(
        user_id=current_user.id,
        text=analysis_data.text,
        sentiment=result["sentiment"],
        score=result["score"],
        emoji=result["emoji"],
        source="manual"  # Análisis manual, pero usando red neuronal
    )
    
    db.add(new_analysis)
    db.commit()
    db.refresh(new_analysis)
    
    return new_analysis

@router.get("", response_model=List[AnalysisResponse])
async def get_analyses(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener lista de análisis del usuario"""
    analyses = db.query(Analysis).filter(
        Analysis.user_id == current_user.id
    ).order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()
    
    return analyses

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener un análisis específico"""
    analysis = db.query(Analysis).filter(
        and_(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id
        )
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Análisis no encontrado"
        )
    
    return analysis

@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Eliminar un análisis"""
    analysis = db.query(Analysis).filter(
        and_(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id
        )
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Análisis no encontrado"
        )
    
    db.delete(analysis)
    db.commit()
    
    return None

@router.get("/stats/summary")
async def get_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener estadísticas del usuario"""
    total = db.query(Analysis).filter(Analysis.user_id == current_user.id).count()
    
    positive = db.query(Analysis).filter(
        and_(
            Analysis.user_id == current_user.id,
            Analysis.sentiment == "positivo"
        )
    ).count()
    
    negative = db.query(Analysis).filter(
        and_(
            Analysis.user_id == current_user.id,
            Analysis.sentiment == "negativo"
        )
    ).count()
    
    neutral = db.query(Analysis).filter(
        and_(
            Analysis.user_id == current_user.id,
            Analysis.sentiment.in_(["neutral", "moderado/neutral"])
        )
    ).count()
    
    avg_score = db.query(func.avg(Analysis.score)).filter(
        Analysis.user_id == current_user.id
    ).scalar() or 0.0
    
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "average_score": float(avg_score)
    }

@router.post("/batch", response_model=AnalysisBatchResponse)
async def analyze_batch(
    batch_data: AnalysisBatch,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analizar múltiples textos a la vez"""
    if not batch_data.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se proporcionaron textos para analizar"
        )
    
    # Verificar límite (contar análisis que se van a crear)
    if current_user.plan == "free":
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_analyses = db.query(Analysis).filter(
            and_(
                Analysis.user_id == current_user.id,
                Analysis.created_at >= today_start
            )
        ).count()
        
        if today_analyses + len(batch_data.texts) > 10:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Excederías el límite diario. Solo puedes analizar {10 - today_analyses} textos más hoy."
            )
    
    # Analizar todos los textos usando SOLO red neuronal LSTM
    # analyze_sentiment() ahora usa exclusivamente red neuronal
    results = []
    for text in batch_data.texts:
        result = analyze_sentiment(text)
        
        # Guardar en BD
        # Todos los análisis batch son neuronales
        new_analysis = Analysis(
            user_id=current_user.id,
            text=text,
            sentiment=result["sentiment"],
            score=result["score"],
            emoji=result["emoji"],
            source="manual"  # Análisis batch manual, pero usando red neuronal
        )
        db.add(new_analysis)
        results.append(new_analysis)
    
    db.commit()
    
    # Refrescar para obtener IDs
    for analysis in results:
        db.refresh(analysis)
    
    # Calcular resumen
    summary = {
        "total": len(results),
        "positive": sum(1 for r in results if r.sentiment == "positivo"),
        "negative": sum(1 for r in results if r.sentiment == "negativo"),
        "neutral": sum(1 for r in results if r.sentiment in ["neutral", "moderado/neutral"])
    }
    
    return AnalysisBatchResponse(results=results, summary=summary)



from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import User, Payment
from app.schemas import PaymentResponse
from app.auth import get_current_user

router = APIRouter(prefix="/api/payments", tags=["payments"])

@router.get("", response_model=List[PaymentResponse])
async def get_payments(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener todos los pagos del usuario"""
    payments = db.query(Payment).filter(
        Payment.user_id == current_user.id
    ).order_by(Payment.created_at.desc()).all()
    
    return payments

@router.get("/{payment_id}", response_model=PaymentResponse)
async def get_payment(
    payment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtener un pago específico"""
    payment = db.query(Payment).filter(
        Payment.id == payment_id,
        Payment.user_id == current_user.id
    ).first()
    
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pago no encontrado"
        )
    
    return payment


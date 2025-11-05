from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime, timedelta
from app.database import get_db
from app.models import User, PasswordResetToken
from app.schemas import (
    UserCreate, UserResponse, UserLogin, Token,
    PasswordResetRequest, PasswordReset
)
from app.auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, generate_reset_token
)
from app.email_service import (
    send_password_reset_email,
    send_welcome_email,
    send_password_reset_success_email
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Registrar un nuevo usuario"""
    # Verificar si el email ya existe
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El email ya está registrado"
        )
    
    # Crear nuevo usuario
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        password_hash=hashed_password,
        name=user_data.name,
        plan="free"
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Enviar email de bienvenida
    try:
        await send_welcome_email(new_user.email, new_user.name)
    except Exception as e:
        # Log error pero no fallar el registro
        print(f"Error enviando email de bienvenida: {e}")
    
    return new_user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Iniciar sesión y obtener token JWT"""
    # Buscar usuario por email
    user = db.query(User).filter(User.email == form_data.username).first()
    
    # Verificar si el usuario existe
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El email no está registrado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verificar contraseña
    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Contraseña incorrecta",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Crear token de acceso
    access_token = create_access_token(data={"sub": user.id})
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Cerrar sesión (el token se invalidará en el frontend)"""
    return {"message": "Sesión cerrada correctamente"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Obtener información del usuario actual"""
    return current_user

@router.post("/forgot-password")
async def forgot_password(
    request: PasswordResetRequest,
    db: Session = Depends(get_db)
):
    """Solicitar recuperación de contraseña"""
    user = db.query(User).filter(User.email == request.email).first()
    
    # Por seguridad, no revelar si el email existe o no
    if user:
        # Generar token de reset
        reset_token = generate_reset_token()
        expires_at = datetime.utcnow() + timedelta(hours=1)  # Token válido por 1 hora
        
        # Guardar token en BD
        reset_token_obj = PasswordResetToken(
            user_id=user.id,
            token=reset_token,
            expires_at=expires_at,
            used=False
        )
        db.add(reset_token_obj)
        db.commit()
        
        # Enviar email (implementar email_service)
        try:
            await send_password_reset_email(user.email, reset_token)
        except Exception as e:
            # Log error pero no revelar al usuario
            print(f"Error enviando email: {e}")
    
    # Siempre devolver éxito (por seguridad)
    return {
        "message": "Si el email existe, recibirás un enlace para recuperar tu contraseña"
    }

@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordReset,
    db: Session = Depends(get_db)
):
    """Restablecer contraseña con token"""
    # Buscar token válido
    reset_token_obj = db.query(PasswordResetToken).filter(
        PasswordResetToken.token == reset_data.token,
        PasswordResetToken.used == False,
        PasswordResetToken.expires_at > datetime.utcnow()
    ).first()
    
    if not reset_token_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token inválido o expirado"
        )
    
    # Obtener usuario
    user = db.query(User).filter(User.id == reset_token_obj.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado"
        )
    
    # Actualizar contraseña
    user.password_hash = get_password_hash(reset_data.new_password)
    
    # Marcar token como usado
    reset_token_obj.used = True
    
    db.commit()
    
    # Enviar email de confirmación de restablecimiento exitoso
    try:
        await send_password_reset_success_email(user.email, user.name)
    except Exception as e:
        # Log error pero no fallar el restablecimiento
        print(f"Error enviando email de confirmación: {e}")
    
    return {"message": "Contraseña actualizada correctamente"}



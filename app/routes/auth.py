from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from datetime import datetime, timedelta
from app.database import get_db
from app.models import User, PasswordResetToken, EmailVerificationToken
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
    send_password_reset_success_email,
    send_verification_email
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Registrar un nuevo usuario"""
    # Normalizar email a minúsculas
    email_lower = user_data.email.lower().strip()
    
    # Verificar si el email ya existe (case-insensitive)
    existing_user = db.query(User).filter(func.lower(User.email) == email_lower).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El email ya está registrado"
        )
    
    # Crear nuevo usuario (guardar email en minúsculas, email_verified=False por defecto)
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=email_lower,
        password_hash=hashed_password,
        name=user_data.name,
        plan="free",
        email_verified=False  # Requiere verificación
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Generar token de verificación
    verification_token = generate_reset_token()
    expires_at = datetime.utcnow() + timedelta(hours=24)  # Token válido por 24 horas
    
    # Guardar token en BD
    verification_token_obj = EmailVerificationToken(
        user_id=new_user.id,
        token=verification_token,
        expires_at=expires_at,
        used=False
    )
    db.add(verification_token_obj)
    db.commit()
    
    # Enviar email de verificación en background (no bloquea el registro)
    background_tasks.add_task(send_verification_email, new_user.email, new_user.name, verification_token)
    
    return new_user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Iniciar sesión y obtener token JWT"""
    # Normalizar email a minúsculas para búsqueda case-insensitive
    email_lower = form_data.username.lower().strip()
    
    # Buscar usuario por email (case-insensitive)
    user = db.query(User).filter(func.lower(User.email) == email_lower).first()
    
    # Verificar si el usuario existe
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El email no está registrado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verificar si el email está verificado
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Por favor verifica tu email antes de iniciar sesión. Revisa tu bandeja de entrada.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verificar contraseña
    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Contraseña incorrecta",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Crear token de acceso con el ID del usuario
    access_token = create_access_token(data={"sub": user.id})
    
    # Verificar que el token se creó correctamente
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al generar token de acceso"
        )
    
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
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Solicitar recuperación de contraseña"""
    # Normalizar email a minúsculas para búsqueda case-insensitive
    email_lower = request.email.lower().strip()
    
    # Buscar usuario por email (case-insensitive)
    user = db.query(User).filter(func.lower(User.email) == email_lower).first()
    
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
        
        # Enviar email en background (no bloquea la respuesta)
        background_tasks.add_task(send_password_reset_email, user.email, reset_token)
    
    # Siempre devolver éxito (por seguridad)
    return {
        "message": "Si el email existe, recibirás un enlace para recuperar tu contraseña"
    }

@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordReset,
    background_tasks: BackgroundTasks,
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
    
    # Enviar email de confirmación en background (no bloquea el restablecimiento)
    background_tasks.add_task(send_password_reset_success_email, user.email, user.name)
    
    return {"message": "Contraseña actualizada correctamente"}

@router.get("/verify-email")
async def verify_email_get(
    token: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Verificar email con token (GET para enlaces en email)"""
    # Buscar token válido
    verification_token_obj = db.query(EmailVerificationToken).filter(
        EmailVerificationToken.token == token,
        EmailVerificationToken.used == False,
        EmailVerificationToken.expires_at > datetime.utcnow()
    ).first()
    
    if not verification_token_obj:
        return {"success": False, "message": "Token de verificación inválido o expirado"}
    
    # Obtener usuario
    user = db.query(User).filter(User.id == verification_token_obj.user_id).first()
    if not user:
        return {"success": False, "message": "Usuario no encontrado"}
    
    # Verificar email del usuario
    user.email_verified = True
    
    # Marcar token como usado
    verification_token_obj.used = True
    
    db.commit()
    
    # Enviar email de bienvenida después de verificar (en background)
    background_tasks.add_task(send_welcome_email, user.email, user.name)
    
    return {"success": True, "message": "Email verificado correctamente. Bienvenido a Sentimetría!"}

@router.post("/verify-email")
async def verify_email_post(
    token: str = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Verificar email con token (POST para llamadas desde frontend)"""
    # Buscar token válido
    verification_token_obj = db.query(EmailVerificationToken).filter(
        EmailVerificationToken.token == token,
        EmailVerificationToken.used == False,
        EmailVerificationToken.expires_at > datetime.utcnow()
    ).first()
    
    if not verification_token_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token de verificación inválido o expirado"
        )
    
    # Obtener usuario
    user = db.query(User).filter(User.id == verification_token_obj.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado"
        )
    
    # Verificar email del usuario
    user.email_verified = True
    
    # Marcar token como usado
    verification_token_obj.used = True
    
    db.commit()
    
    # Enviar email de bienvenida después de verificar (en background)
    background_tasks.add_task(send_welcome_email, user.email, user.name)
    
    return {"success": True, "message": "Email verificado correctamente. Bienvenido a Sentimetría!"}



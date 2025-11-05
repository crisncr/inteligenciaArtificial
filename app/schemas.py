from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime

# Schemas para Usuario
class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('La contraseña debe tener al menos 8 caracteres')
        if not v[0].isupper():
            raise ValueError('La contraseña debe comenzar con una letra mayúscula')
        if not any(c.islower() for c in v):
            raise ValueError('La contraseña debe contener al menos una letra minúscula')
        if not any(c.isdigit() for c in v):
            raise ValueError('La contraseña debe contener al menos un número')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    plan: str
    email_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Schemas para Análisis
class AnalysisBase(BaseModel):
    text: str

class AnalysisCreate(AnalysisBase):
    pass

class AnalysisResponse(BaseModel):
    id: int
    text: str
    sentiment: str
    score: float
    emoji: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class AnalysisBatch(BaseModel):
    texts: List[str]

class AnalysisBatchResponse(BaseModel):
    results: List[AnalysisResponse]
    summary: dict

# Schemas para Tokens
class Token(BaseModel):
    access_token: str
    token_type: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str
    
    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('La contraseña debe tener al menos 8 caracteres')
        if not v[0].isupper():
            raise ValueError('La contraseña debe comenzar con una letra mayúscula')
        if not any(c.islower() for c in v):
            raise ValueError('La contraseña debe contener al menos una letra minúscula')
        if not any(c.isdigit() for c in v):
            raise ValueError('La contraseña debe contener al menos un número')
        return v

# Schemas para Planes
class PlanResponse(BaseModel):
    id: int
    name: str
    price: float
    features: Optional[dict] = None
    
    class Config:
        from_attributes = True

# Schemas para Pagos
class PaymentCreate(BaseModel):
    plan_id: int

class PaymentResponse(BaseModel):
    id: int
    amount: float
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True



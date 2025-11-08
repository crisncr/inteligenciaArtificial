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
    source: str = "manual"
    external_api_id: Optional[int] = None
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

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    plan: Optional[str] = None

class PasswordChange(BaseModel):
    current_password: str
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
    payment_method: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Schemas para API Externa
class ExternalAPIBase(BaseModel):
    name: str
    api_url: str
    endpoint: str
    method: str = "GET"
    headers: Optional[dict] = None
    auth_type: Optional[str] = None
    auth_token: Optional[str] = None
    active: bool = True

class ExternalAPICreate(ExternalAPIBase):
    pass

class ExternalAPIUpdate(BaseModel):
    name: Optional[str] = None
    api_url: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    headers: Optional[dict] = None
    auth_type: Optional[str] = None
    auth_token: Optional[str] = None
    active: Optional[bool] = None

class ExternalAPIResponse(ExternalAPIBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class ExternalAPITest(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class ExternalAPIAnalyze(BaseModel):
    comments_count: int
    analyses_created: int
    errors: Optional[List[str]] = None

# Schemas para Rutas
class RoutePointBase(BaseModel):
    name: str
    address: str
    lat: float
    lng: float
    display_name: Optional[str] = None
    order: int

class RoutePointResponse(RoutePointBase):
    id: int
    route_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class RouteBase(BaseModel):
    name: str
    algorithm: str = "astar"

class RouteCreate(RouteBase):
    points: List[RoutePointBase]
    distance: Optional[float] = None

class RouteResponse(RouteBase):
    id: int
    user_id: int
    distance: Optional[float] = None
    points: List[RoutePointResponse]
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True



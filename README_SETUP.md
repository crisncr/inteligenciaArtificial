# Guía de Configuración - Sentimetría

## Configuración de Base de Datos PostgreSQL

### 1. Instalar PostgreSQL

**Windows:**
- Descargar desde: https://www.postgresql.org/download/windows/
- Instalar PostgreSQL
- Durante la instalación, configurar:
  - Usuario: `postgres`
  - Contraseña: `postgres` (o la que prefieras)
  - Puerto: `5432`

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# Mac
brew install postgresql
```

### 2. Crear Base de Datos

```sql
-- Conectarse a PostgreSQL
psql -U postgres

-- Crear base de datos
CREATE DATABASE sentimetria;

-- Salir
\q
```

### 3. Configurar Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/sentimetria

# JWT
SECRET_KEY=tu-clave-secreta-muy-segura-aqui

# Email (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tu-email@gmail.com
SMTP_PASSWORD=tu-contraseña-de-aplicacion

# Frontend URL
FRONTEND_URL=http://localhost:5173

# Stripe (opcional)
STRIPE_PUBLIC_KEY=pk_test_tu_key
STRIPE_SECRET_KEY=sk_test_tu_key
STRIPE_WEBHOOK_SECRET=whsec_tu_webhook_secret
```

### 4. Instalar Dependencias de Python

```bash
# Activar entorno virtual
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 5. Crear Migraciones

```bash
# Crear migración inicial
alembic revision --autogenerate -m "Initial migration"

# Aplicar migraciones
alembic upgrade head
```

### 6. Iniciar Servidor

```bash
# Activar entorno virtual
.\venv\Scripts\Activate.ps1  # Windows

# Iniciar servidor
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Configuración de Email (Opcional)

Para enviar emails de recuperación de contraseña:

### Gmail:
1. Ir a configuración de cuenta de Google
2. Habilitar "Verificación en 2 pasos"
3. Generar "Contraseña de aplicación"
4. Usar esa contraseña en `SMTP_PASSWORD`

### Otros proveedores:
- **Outlook**: `smtp-mail.outlook.com`, puerto 587
- **SendGrid**: Servicio de email profesional
- **AWS SES**: Servicio de email de AWS

## Notas Importantes

- **Validación de contraseña**: Primera letra debe ser mayúscula, mínimo 8 caracteres, debe contener minúsculas y números
- **Base de datos**: Asegúrate de que PostgreSQL esté corriendo antes de iniciar el servidor
- **Variables de entorno**: No subir el archivo `.env` a Git (está en `.gitignore`)

## Estructura de Archivos

```
app/
├── database.py          # Configuración de BD
├── models.py            # Modelos SQLAlchemy
├── schemas.py           # Schemas Pydantic
├── auth.py              # Autenticación JWT
├── routes/
│   ├── auth.py          # Endpoints de autenticación
│   └── analyses.py      # Endpoints de análisis
└── email_service.py     # Servicio de email

alembic/
├── env.py               # Configuración de Alembic
└── versions/            # Migraciones de BD
```


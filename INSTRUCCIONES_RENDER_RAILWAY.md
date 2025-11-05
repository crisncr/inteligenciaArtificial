# Guía: Servicio Web en Render + Base de Datos en Railway

## Arquitectura
- **Servicio Web (Frontend + Backend)**: Render
- **Base de Datos PostgreSQL**: Railway

## Paso 1: Configurar PostgreSQL en Railway

### 1.1 Crear Base de Datos en Railway

1. Ve a [railway.app](https://railway.app)
2. Regístrate con GitHub
3. Click en "New Project"
4. Click en "+ New" → "Database" → "PostgreSQL"
5. Railway creará automáticamente la base de datos PostgreSQL
6. **NO necesitas configurar nada más**, Railway lo hace automáticamente

### 1.2 Obtener la URL Externa de la Base de Datos

1. En Railway, ve a tu base de datos PostgreSQL
2. Click en la pestaña "Connect"
3. Busca "Public Network" o "Connection URL"
4. Copia la "External Database URL" (la que puedes usar desde fuera de Railway)
   - Formato: `postgresql://usuario:password@host:puerto/database`
   - Ejemplo: `postgresql://postgres:password@containers-us-west-xxx.railway.app:5432/railway`

### 1.3 Configurar Acceso Público (si es necesario)

1. En Railway, ve a tu base de datos PostgreSQL
2. Click en "Settings"
3. Busca "Public Networking" o "Public Access"
4. Activa el acceso público (si Railway lo requiere)
5. Esto permite que Render se conecte desde fuera de Railway

## Paso 2: Configurar Servicio Web en Render

### 2.1 Crear/Actualizar Servicio Web en Render

1. Ve a [render.com](https://render.com)
2. Si ya tienes un servicio web, ve a él
3. Si no, crea uno nuevo:
   - "New +" → "Web Service"
   - Conecta tu repositorio de GitHub
   - Render detectará automáticamente el `render.yaml`

### 2.2 Configurar Variable DATABASE_URL

1. En tu servicio web de Render, ve a "Environment"
2. Click en "Add Environment Variable"
3. Agrega:
   - **Key**: `DATABASE_URL`
   - **Value**: Pega la "External Database URL" que copiaste de Railway
     - Debe ser la URL externa, no la interna
     - Formato: `postgresql://usuario:password@host:puerto/database`
4. Click en "Save Changes"

### 2.3 Configurar Otras Variables de Entorno

En Render, agrega estas variables adicionales:

**Variables Obligatorias:**
- `SECRET_KEY`: Genera una clave secreta (ej: `openssl rand -hex 32`)
- `SMTP_HOST`: `smtp.gmail.com`
- `SMTP_PORT`: `587`
- `SMTP_USER`: Tu email (ej: `tu-email@gmail.com`)
- `SMTP_PASSWORD`: Tu contraseña de aplicación de Gmail
- `FRONTEND_URL`: La URL que Render te dará (ej: `https://tu-app.onrender.com`)

**Variables Opcionales:**
- `STRIPE_PUBLIC_KEY`: (opcional)
- `STRIPE_SECRET_KEY`: (opcional)
- `STRIPE_WEBHOOK_SECRET`: (opcional)

### 2.4 Actualizar render.yaml (si es necesario)

El archivo `render.yaml` actual está configurado para crear una base de datos en Render. 
Como usarás Railway, puedes:

**Opción A: Mantener render.yaml sin cambios**
- Solo asegúrate de que `DATABASE_URL` esté configurada manualmente en Render
- El `render.yaml` seguirá funcionando, pero ignorará la sección de `databases`

**Opción B: Actualizar render.yaml**
- Eliminar la sección `databases` del `render.yaml`
- Mantener solo la sección `services`

## Paso 3: Configurar Email (Gmail)

Para que funcione la recuperación de contraseña:

1. Ve a [Google Account Security](https://myaccount.google.com/security)
2. Activa "Verificación en 2 pasos"
3. Ve a "Contraseñas de aplicaciones"
4. Crea una nueva contraseña para "Correo"
5. Usa esa contraseña en `SMTP_PASSWORD` en Render

## Paso 4: Verificar la Conexión

### 4.1 Verificar que Render se Conecte a Railway

1. En Render, ve a los logs de tu servicio web
2. Busca mensajes de conexión a la base de datos
3. Si hay errores, verifica:
   - Que `DATABASE_URL` esté correcta
   - Que Railway tenga acceso público habilitado
   - Que la URL sea la "External Database URL", no la interna

### 4.2 Ejecutar Migraciones

Las migraciones se ejecutarán automáticamente durante el build en Render:

1. Ve a "Events" en tu servicio de Render
2. Revisa los logs del build
3. Deberías ver: `alembic upgrade head`
4. Si fallan, las tablas se crearán automáticamente al iniciar (tiene respaldo)

### 4.3 Probar la Aplicación

1. Abre la URL de Render
2. Registra un usuario de prueba
3. Verifica que los datos se guarden en la base de datos de Railway
4. Prueba el análisis de sentimientos

## Notas Importantes

### Seguridad

- **Railway PostgreSQL**: Asegúrate de que el acceso público esté configurado correctamente
- **Contraseña**: Railway genera una contraseña fuerte automáticamente
- **URL Externa**: Solo comparte esta URL en variables de entorno, nunca en código público

### Ventajas de esta Configuración

✅ **Render**: Servicio web estable y fácil de usar
✅ **Railway**: Base de datos PostgreSQL con plan gratuito generoso
✅ **Separación**: Base de datos independiente del servicio web
✅ **Flexibilidad**: Puedes cambiar el servicio web sin afectar la base de datos

### Posibles Problemas

1. **Error de conexión:**
   - Verifica que Railway tenga acceso público habilitado
   - Verifica que la URL sea la "External Database URL"
   - Verifica que no haya firewall bloqueando la conexión

2. **Migraciones no se ejecutan:**
   - Revisa los logs del build en Render
   - Verifica que `DATABASE_URL` esté configurada antes del build
   - Las tablas se crearán automáticamente al iniciar si las migraciones fallan

3. **Timeout de conexión:**
   - Railway puede tener límites de conexión en el plan gratuito
   - Considera usar el plan de pago si necesitas más conexiones

## Resumen de Pasos

1. ✅ Crear PostgreSQL en Railway
2. ✅ Copiar "External Database URL" de Railway
3. ✅ Agregar `DATABASE_URL` en Render con la URL de Railway
4. ✅ Configurar otras variables de entorno en Render
5. ✅ Configurar Gmail para emails
6. ✅ Desplegar y verificar


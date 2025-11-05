# Guía de Despliegue en Railway

## Paso 1: Crear Cuenta y Proyecto

1. Ve a [railway.app](https://railway.app)
2. Regístrate con GitHub (recomendado)
3. Click en "New Project"
4. Selecciona "Deploy from GitHub repo"
5. Conecta tu repositorio y selecciona el repositorio de tu proyecto

## Paso 2: Configurar Base de Datos PostgreSQL

1. **Crear Base de Datos:**
   - En tu proyecto de Railway, click en "+ New"
   - Selecciona "Database" → "PostgreSQL"
   - Railway creará automáticamente la base de datos PostgreSQL
   - **NO necesitas configurar nada**, Railway lo hace automáticamente

2. **Conectar al Servicio Web:**
   - Railway detectará automáticamente la base de datos
   - La variable `DATABASE_URL` se configurará automáticamente
   - **NO necesitas hacer nada manualmente**

## Paso 3: Configurar Variables de Entorno

1. **En tu servicio web de Railway:**
   - Click en tu servicio web (el que se creó automáticamente)
   - Ve a la pestaña "Variables"
   - Railway ya habrá configurado `DATABASE_URL` automáticamente

2. **Agregar Variables Manuales:**
   Click en "New Variable" y agrega:

   **Variables Obligatorias:**
   - `SECRET_KEY`: Genera una clave secreta (puedes usar: `openssl rand -hex 32` o cualquier string aleatorio largo)
   - `SMTP_HOST`: `smtp.gmail.com`
   - `SMTP_PORT`: `587`
   - `SMTP_USER`: Tu email (ej: `tu-email@gmail.com`)
   - `SMTP_PASSWORD`: Tu contraseña de aplicación de Gmail (ver sección Email)
   - `FRONTEND_URL`: La URL que Railway te dará (ej: `https://tu-app.up.railway.app`)

   **Variables Opcionales (para pagos):**
   - `STRIPE_PUBLIC_KEY`: (opcional)
   - `STRIPE_SECRET_KEY`: (opcional)
   - `STRIPE_WEBHOOK_SECRET`: (opcional)

## Paso 4: Configurar Email (Gmail)

Para que funcione la recuperación de contraseña:

1. Ve a tu cuenta de Google → [Seguridad](https://myaccount.google.com/security)
2. Activa "Verificación en 2 pasos" (si no está activada)
3. Ve a "Contraseñas de aplicaciones"
4. Click en "Seleccionar app" → "Correo"
5. Click en "Seleccionar dispositivo" → "Otro (nombre personalizado)"
6. Escribe "Railway Sentimetria" y click "Generar"
7. Copia la contraseña generada de 16 caracteres
8. Usa esa contraseña en la variable `SMTP_PASSWORD` en Railway

## Paso 5: Verificar el Despliegue

1. **Railway desplegará automáticamente:**
   - Railway detectará que es un proyecto Python
   - Instalará las dependencias de `requirements.txt`
   - Instalará Node.js y ejecutará `npm install`
   - Ejecutará `npm run build` para construir React
   - Ejecutará las migraciones de Alembic

2. **Revisar Logs:**
   - Ve a la pestaña "Deployments" en tu servicio
   - Click en el deployment más reciente
   - Revisa los logs para ver si hay errores

3. **Obtener URL:**
   - Railway generará automáticamente una URL
   - Ve a la pestaña "Settings" → "Domains"
   - Ahí verás la URL (ej: `https://tu-app.up.railway.app`)
   - Actualiza la variable `FRONTEND_URL` con esta URL

## Paso 6: Ejecutar Migraciones (si es necesario)

Railway debería ejecutar las migraciones automáticamente durante el build, pero si no:

1. Ve a tu servicio web
2. Click en "Settings" → "Deploy"
3. Click en "Deploy" para forzar un nuevo deployment

O desde la terminal (si tienes Railway CLI instalado):
```bash
railway run alembic upgrade head
```

## Notas Importantes

### Ventajas de Railway:
- ✅ Configuración automática de PostgreSQL
- ✅ `DATABASE_URL` se configura automáticamente
- ✅ No requiere archivos de configuración complejos
- ✅ Detección automática de Python y Node.js
- ✅ Build automático

### Plan Gratuito:
- **$5 de crédito gratis al mes** (suficiente para desarrollo)
- Después del crédito, pasa a pago por uso
- La base de datos PostgreSQL tiene un plan gratuito generoso

### Variables de Entorno:
- Railway configurará automáticamente `DATABASE_URL` cuando conectes PostgreSQL
- Solo necesitas agregar las variables manuales (SMTP, SECRET_KEY, etc.)

### Si hay problemas:

1. **Error de conexión a BD:**
   - Verifica que PostgreSQL esté conectado al servicio web
   - Ve a PostgreSQL → "Connect" → Verifica la variable `DATABASE_URL`

2. **Error en build:**
   - Revisa los logs en "Deployments"
   - Verifica que todas las dependencias estén en `requirements.txt`
   - Verifica que `package.json` esté correcto

3. **Migraciones no se ejecutan:**
   - Las migraciones están en el `buildCommand` de `railway.json`
   - Si fallan, las tablas se crearán automáticamente al iniciar (tiene respaldo)

4. **Frontend no se muestra:**
   - Verifica que `npm run build` se ejecutó correctamente
   - Verifica que `app/static/dist/` tenga los archivos generados
   - Revisa los logs del build


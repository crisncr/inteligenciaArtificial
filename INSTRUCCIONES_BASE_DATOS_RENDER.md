# Instrucciones para Configurar Base de Datos PostgreSQL en Render

## Situación: Tienes un servicio web existente en Render

### Opción 1: Crear Base de Datos Manualmente (Recomendado)

1. **Crear la Base de Datos PostgreSQL:**
   - Ve a tu dashboard de Render
   - Click en "New +" → "PostgreSQL"
   - Configura:
     - **Name**: `sentimetria-db` (o el nombre que prefieras)
     - **Database**: `sentimetria`
     - **User**: `sentimetria` (o el usuario que prefieras)
     - **Region**: La misma región que tu servicio web
     - **PostgreSQL Version**: La más reciente
     - **Plan**: Free (o el plan que prefieras)
   - Click en "Create Database"

2. **Conectar la Base de Datos a tu Servicio Web:**
   - Ve a tu servicio web existente en Render
   - Click en "Environment" (en el menú lateral)
   - Busca la sección "Environment Variables"
   - Click en "Add Environment Variable"
   - Agrega:
     - **Key**: `DATABASE_URL`
     - **Value**: Copia la "Internal Database URL" de tu base de datos PostgreSQL
       - Ve a tu base de datos PostgreSQL → "Connections" → "Internal Database URL"
       - Copia esa URL completa
   - Click en "Save Changes"

3. **Reiniciar el Servicio:**
   - Después de agregar la variable, Render reiniciará automáticamente el servicio
   - O puedes hacerlo manualmente desde "Manual Deploy" → "Clear build cache & deploy"

### Opción 2: Usar render.yaml (Automático)

Si prefieres usar el archivo `render.yaml`:

1. **Actualizar tu servicio existente:**
   - El archivo `render.yaml` está configurado para crear la base de datos automáticamente
   - Pero si ya tienes un servicio, necesitas:
     - Eliminar el servicio existente (si es necesario)
     - O actualizar el servicio para que use el `render.yaml`

2. **Si Render ya detecta el render.yaml:**
   - Render creará automáticamente la base de datos `sentimetria-db`
   - Y conectará `DATABASE_URL` automáticamente al servicio web

### Verificar la Configuración

1. **Verificar que DATABASE_URL esté configurada:**
   - Ve a tu servicio web → "Environment"
   - Verifica que existe `DATABASE_URL` con el valor correcto

2. **Verificar el formato de la URL:**
   - La URL de Render puede ser: `postgres://...` o `postgresql://...`
   - El código ya está configurado para convertir automáticamente `postgres://` a `postgresql://`

3. **Probar la conexión:**
   - Los logs del servicio mostrarán si hay errores de conexión
   - Las migraciones se ejecutarán automáticamente durante el build

### Notas Importantes

- **URL Externa vs Interna:**
  - **Internal Database URL**: Para servicios dentro de Render (más rápido, gratis)
  - **External Database URL**: Para conexiones desde fuera de Render (requiere whitelist de IPs)
  
- **Para tu caso (servicio web en Render):**
  - Usa la **Internal Database URL**
  - Es más rápida y no requiere configuración adicional

- **Formato de la URL:**
  ```
  postgresql://usuario:contraseña@host:puerto/nombre_base_datos
  ```
  
- **Ejemplo de URL interna de Render:**
  ```
  postgresql://sentimetria:password@dpg-xxxxx-a.oregon-postgres.render.com/sentimetria
  ```

### Si hay problemas

1. **Error de conexión:**
   - Verifica que la URL esté correcta
   - Verifica que la base de datos esté en la misma región que tu servicio web
   - Verifica que el servicio web tenga acceso a la base de datos

2. **Error en migraciones:**
   - Los logs mostrarán el error específico
   - Verifica que `DATABASE_URL` esté configurada antes del build
   - Las migraciones se ejecutan durante el build command

3. **Tablas no se crean:**
   - El código también tiene `Base.metadata.create_all(bind=engine)` como respaldo
   - Si las migraciones fallan, las tablas se crearán automáticamente al iniciar


# Guía: Configurar Google Maps API Key

## ¿Por qué necesitas hacer esto?

Yo integré el código para usar Google Maps, pero **Google requiere que cada usuario tenga su propia API key** por seguridad y para controlar el uso. No puedo proporcionar una API key compartida porque:
1. Google requiere una cuenta de Google Cloud con tarjeta de crédito
2. Cada API key tiene límites de uso personalizados
3. Es una práctica de seguridad (cada aplicación debe tener su propia key)

## Paso 1: Obtener API Key de Google Cloud

### 1.1 Crear cuenta en Google Cloud Console

1. Ve a: https://console.cloud.google.com/
2. Inicia sesión con tu cuenta de Google
3. Acepta los términos si es la primera vez

### 1.2 Crear un proyecto

1. Haz click en el selector de proyectos (arriba a la izquierda)
2. Click en "Nuevo proyecto"
3. Nombre del proyecto: `sentiment-analysis-maps` (o el que prefieras)
4. Click en "Crear"
5. Espera a que se cree (puede tardar unos segundos)

### 1.3 Habilitar facturación (GRATIS hasta $200/mes)

**⚠️ IMPORTANTE:** Google Maps ofrece $200 USD de crédito GRATIS cada mes, que es más que suficiente para desarrollo y uso moderado.

1. En el menú lateral, ve a "Facturación"
2. Si no tienes cuenta de facturación, Google te pedirá crear una
3. Agrega una tarjeta de crédito (Google NO cobrará automáticamente, solo si superas los $200/mes)
4. Puedes configurar alertas de gasto para no superar el límite gratuito

### 1.4 Habilitar las APIs necesarias

1. En el menú lateral, ve a "APIs y servicios" > "Biblioteca"
2. Busca y habilita estas APIs (una por una):
   - **Maps JavaScript API** - Para mostrar el mapa
   - **Places API** - Para autocompletado de direcciones
   - **Geocoding API** - Para convertir direcciones en coordenadas

   Para cada API:
   - Click en el nombre
   - Click en "Habilitar"
   - Espera a que se habilite

### 1.5 Crear la API Key

1. Ve a "APIs y servicios" > "Credenciales"
2. Click en "Crear credenciales" > "Clave de API"
3. Se creará una API key automáticamente
4. **COPIA la API key** (será algo como: `AIzaSyC...`)
5. (Opcional pero recomendado) Click en "Restringir clave" para mayor seguridad:
   - **Restricciones de aplicación:** Selecciona "Aplicaciones web"
   - Agrega tu dominio de Render (ej: `*.onrender.com`)
   - **Restricciones de API:** Selecciona solo las 3 APIs que habilitaste

## Paso 2: Configurar en Render

### 2.1 Agregar variable de entorno

1. Ve a tu dashboard de Render: https://dashboard.render.com/
2. Selecciona tu servicio (el que tiene el backend)
3. Ve a la pestaña "Environment"
4. Scroll hasta "Environment Variables"
5. Click en "Add Environment Variable"
6. Agrega:
   - **Key:** `GOOGLE_MAPS_API_KEY`
   - **Value:** Pega la API key que copiaste de Google Cloud
7. Click en "Save Changes"

### 2.2 Reiniciar el servicio

1. Ve a la pestaña "Events" o "Logs"
2. Click en "Manual Deploy" > "Clear build cache & deploy" (o simplemente reinicia el servicio)
3. Espera a que se reinicie (puede tardar 1-2 minutos)

## Paso 3: Verificar que funciona

1. Ve a tu aplicación en Render
2. Navega a la sección "Optimización de Rutas"
3. Deberías ver:
   - El mapa de Google Maps cargado
   - Autocompletado funcionando en los campos de dirección
   - Los botones "Seleccionar en mapa" habilitados

## Límites gratuitos

Google Maps ofrece $200 USD de crédito gratuito mensual, que equivale aproximadamente a:
- **28,000 solicitudes de Maps JavaScript API**
- **17,000 solicitudes de Places API (Autocomplete)**
- **40,000 solicitudes de Geocoding API**

Para una aplicación en desarrollo o con uso moderado, esto es más que suficiente.

## Solución de problemas

### El mapa no aparece
- Verifica que la API key esté correctamente configurada en Render
- Verifica que las 3 APIs estén habilitadas en Google Cloud
- Revisa los logs de Render para ver si hay errores
- Verifica en la consola del navegador (F12) si hay errores de JavaScript

### Error: "API key not valid"
- Verifica que copiaste la API key completa (no debe tener espacios)
- Verifica que la API key no tenga restricciones que bloqueen tu dominio
- Verifica que las APIs estén habilitadas

### Error: "Billing not enabled"
- Necesitas habilitar facturación en Google Cloud (aunque sea gratuita)

## Costos

- **$200 USD/mes GRATIS** (más que suficiente para desarrollo)
- Solo se cobra si superas los $200/mes
- Puedes configurar alertas de gasto en Google Cloud Console
- Puedes configurar límites de uso diario/mensual

## Seguridad

**⚠️ IMPORTANTE:** La API key estará visible en el frontend (es normal para Google Maps), pero puedes restringirla:
1. Restringir por dominio (solo `*.onrender.com`)
2. Restringir por API (solo las 3 APIs que necesitas)
3. Configurar límites de uso diario

Esto previene el uso no autorizado de tu API key.

## ¿Necesitas ayuda?

Si tienes problemas, verifica:
1. ✅ API key creada en Google Cloud
2. ✅ 3 APIs habilitadas (Maps JavaScript, Places, Geocoding)
3. ✅ Facturación habilitada
4. ✅ Variable `GOOGLE_MAPS_API_KEY` configurada en Render
5. ✅ Servicio reiniciado en Render

Si todo está correcto y aún no funciona, revisa los logs de Render y la consola del navegador para ver errores específicos.


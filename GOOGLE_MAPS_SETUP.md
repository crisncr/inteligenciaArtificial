# Configuración de Google Maps API

## Pasos para configurar Google Maps API

### 1. Crear un proyecto en Google Cloud Console

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Habilita la facturación (Google Maps ofrece $200 de crédito gratuito mensual)

### 2. Habilitar las APIs necesarias

Habilita las siguientes APIs en tu proyecto:
- **Geocoding API**: Para convertir direcciones en coordenadas
- **Places API**: Para autocompletado de direcciones

### 3. Crear una API Key

1. Ve a "Credenciales" en Google Cloud Console
2. Haz clic en "Crear credenciales" > "Clave de API"
3. Copia la API key generada

### 4. Configurar restricciones de la API Key (Recomendado)

Por seguridad, configura restricciones:
- **Restricciones de aplicación**: Restringe por IP o dominio HTTP
- **Restricciones de API**: Limita solo a Geocoding API y Places API

### 5. Configurar la variable de entorno

En Render:
1. Ve a tu servicio en Render
2. Ve a "Environment"
3. Agrega la variable de entorno:
   - **Key**: `GOOGLE_MAPS_API_KEY`
   - **Value**: Tu API key de Google Maps

### 6. Verificar que funciona

Una vez configurada la API key, el sistema usará automáticamente:
- **Google Geocoding API** para convertir direcciones en coordenadas
- **Google Places API** para autocompletado de direcciones

## Límites gratuitos

Google Maps ofrece:
- **$200 de crédito gratuito mensual**
- Aproximadamente 40,000 solicitudes de Geocoding API
- Aproximadamente 17,000 solicitudes de Places API Autocomplete

## Solución de problemas

Si ves errores como "API key no configurada":
1. Verifica que la variable de entorno `GOOGLE_MAPS_API_KEY` esté configurada en Render
2. Verifica que las APIs estén habilitadas en Google Cloud Console
3. Verifica que la API key tenga los permisos correctos
4. Revisa los logs del servidor para ver mensajes de error específicos


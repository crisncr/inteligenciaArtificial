# 🚀 Instrucciones para Usar Modelo Pre-entrenado

## 📋 Resumen

Este sistema ahora descarga automáticamente el modelo pre-entrenado desde GitHub Releases, evitando que se entrene cada vez que Render inicia. Esto ahorra tiempo y memoria.

## 🔧 Pasos para Configurar

### Paso 1: Entrenar el Modelo Localmente

Ejecuta el script en tu computadora:

```bash
python train_model_local.py
```

Esto creará los siguientes archivos en `app/ml_models/`:
- `sentiment_model.h5` (modelo entrenado)
- `tokenizer.pkl` (tokenizer)
- `label_encoder.pkl` (codificador de etiquetas)

### Paso 2: Verificar que los Archivos se Crearon

Asegúrate de que los 3 archivos existen:
- `app/ml_models/sentiment_model.h5`
- `app/ml_models/tokenizer.pkl`
- `app/ml_models/label_encoder.pkl`

### Paso 3: Subir Archivos a GitHub Releases

1. Ve a tu repositorio en GitHub: `https://github.com/crisncr/inteligenciaArtificial`
2. Haz clic en "Releases" → "Create a new release"
3. Configura el release:
   - **Tag version**: `v1.0.0`
   - **Release title**: `Modelo Pre-entrenado v1.0`
   - **Description**: `Modelo de análisis de sentimientos pre-entrenado`
4. Arrastra y suelta los 3 archivos:
   - `sentiment_model.h5`
   - `tokenizer.pkl`
   - `label_encoder.pkl`
5. Haz clic en "Publish release"

### Paso 4: Obtener las URLs de Descarga

Después de publicar el release:

1. Ve a la página del release: `https://github.com/crisncr/inteligenciaArtificial/releases/tag/v1.0.0`
2. Haz clic derecho en cada archivo → "Copy link address"
3. Las URLs deberían verse así:
   - `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5`
   - `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl`
   - `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl`

### Paso 5: Actualizar las URLs en el Código (si es necesario)

Si las URLs son diferentes, actualiza `app/ml_models/sentiment_nn.py` en las líneas 582-592, o configura variables de entorno en Render:

```bash
MODEL_URL=https://github.com/tu-usuario/tu-repo/releases/download/v1.0.0/sentiment_model.h5
TOKENIZER_URL=https://github.com/tu-usuario/tu-repo/releases/download/v1.0.0/tokenizer.pkl
LABEL_ENCODER_URL=https://github.com/tu-usuario/tu-repo/releases/download/v1.0.0/label_encoder.pkl
```

### Paso 6: Hacer Commit y Push

```bash
git add train_model_local.py app/ml_models/sentiment_nn.py requirements.txt INSTRUCCIONES_MODELO_PRETRENADO.md
git commit -m "Agregar descarga automática de modelo pre-entrenado desde GitHub Releases"
git push
```

### Paso 7: Verificar en Render

Después del despliegue, verifica en los logs de Render que:

1. ✅ Se descarguen los archivos desde GitHub Releases (no se entrenen)
2. ✅ El modelo se cargue correctamente
3. ✅ La aplicación funcione correctamente

## 🔍 Cómo Funciona

1. **Al iniciar la aplicación en Render:**
   - El código intenta cargar los archivos localmente
   - Si no existen, intenta descargarlos desde GitHub Releases
   - Si la descarga falla, entrena el modelo como fallback

2. **Ventajas:**
   - ⚡ Inicio más rápido (no hay que entrenar)
   - 💾 Menor uso de memoria (no se entrena)
   - 🔄 Modelo consistente (siempre el mismo)
   - 📦 Fácil de actualizar (solo subir nuevo release)

## ⚠️ Notas Importantes

1. **No subas los archivos del modelo a Git directamente**
   - Son archivos grandes y Render los borra de todas formas
   - Úsalos solo para subirlos a GitHub Releases

2. **Si cambias el modelo:**
   - Entrénalo localmente de nuevo
   - Sube los nuevos archivos a un nuevo release (v1.1.0, etc.)
   - Actualiza las URLs en el código o variables de entorno

3. **Si GitHub Releases no está disponible:**
   - El sistema automáticamente entrenará el modelo como fallback
   - Esto tomará 30-60 segundos pero funcionará

## 🐛 Solución de Problemas

### Error: "No se pudo descargar el modelo"
- Verifica que las URLs en el código sean correctas
- Verifica que el release esté publicado (no draft)
- Verifica que los archivos estén en el release

### Error: "El modelo no se carga correctamente"
- Verifica que los archivos se descargaron correctamente
- Verifica que el modelo fue entrenado con la misma versión del código
- Si persiste, reentrena el modelo localmente y vuelve a subirlo

### El modelo se entrena cada vez (no descarga)
- Verifica que las URLs sean correctas
- Verifica que el release esté público
- Verifica que `requests` esté instalado (está en requirements.txt)

## ✅ Checklist

- [ ] Script `train_model_local.py` creado
- [ ] Modelo entrenado localmente
- [ ] Archivos creados en `app/ml_models/`
- [ ] Release creado en GitHub
- [ ] Archivos subidos al release
- [ ] URLs copiadas y verificadas
- [ ] Código actualizado (si es necesario)
- [ ] Commit y push realizado
- [ ] Verificado en Render

## 📞 Soporte

Si tienes problemas, revisa los logs de Render para ver qué está pasando. Los mensajes de log te dirán si está descargando o entrenando el modelo.


# 📋 Pasos Exactos para Entrenar Modelo Localmente

## ✅ Paso 1: Instalar Microsoft Visual C++ Redistributable

1. **Abre tu navegador** y ve a:
   ```
   https://aka.ms/vs/17/release/vc_redist.x64.exe
   ```

2. **Descarga el archivo** (vc_redist.x64.exe)

3. **Ejecuta el instalador** (haz doble clic en el archivo descargado)

4. **Sigue las instrucciones** del instalador (solo haz clic en "Instalar" y "Cerrar")

5. **Reinicia PowerShell** (ciérralo y ábrelo de nuevo)

## ✅ Paso 2: Activar el Entorno Virtual

En PowerShell, ejecuta:
```powershell
.\venv\Scripts\Activate.ps1
```

## ✅ Paso 3: Ejecutar el Script de Entrenamiento

```powershell
python train_model_local.py
```

Esto tomará **30-60 segundos** y creará 3 archivos en `app/ml_models/`:
- `sentiment_model.h5`
- `tokenizer.pkl`
- `label_encoder.pkl`

## ✅ Paso 4: Verificar que los Archivos se Crearon

Verifica que existen estos archivos:
```
app/ml_models/sentiment_model.h5
app/ml_models/tokenizer.pkl
app/ml_models/label_encoder.pkl
```

## ✅ Paso 5: Subir Archivos a GitHub Releases

1. Ve a: `https://github.com/crisncr/inteligenciaArtificial`
2. Haz clic en **"Releases"** (en el menú de la derecha)
3. Haz clic en **"Create a new release"**
4. Configura:
   - **Tag version**: `v1.0.0`
   - **Release title**: `Modelo Pre-entrenado v1.0`
   - **Description**: `Modelo de análisis de sentimientos pre-entrenado`
5. **Arrastra y suelta** los 3 archivos desde `app/ml_models/`:
   - `sentiment_model.h5`
   - `tokenizer.pkl`
   - `label_encoder.pkl`
6. Haz clic en **"Publish release"**

## ✅ Paso 6: Verificar en Render

Después del próximo despliegue en Render, verifica en los logs que:
- ✅ Descarga los archivos desde GitHub Releases
- ✅ NO entrena el modelo
- ✅ Carga el modelo pre-entrenado rápidamente

## 🎯 Resultado Final

- ⚡ Render carga rápido (solo descarga archivos)
- 💾 No usa memoria para entrenar (solo carga modelo)
- 🔄 Modelo consistente (siempre el mismo)
- ✅ No hay problemas de memoria


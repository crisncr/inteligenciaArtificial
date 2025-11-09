# download_model_from_render.py
"""
Script para descargar los archivos del modelo desde Render después de que se entrenen.

Este script crea un endpoint temporal en la aplicación para descargar los archivos.
"""
import os

def create_download_endpoint():
    """Crear endpoint temporal para descargar archivos del modelo desde Render"""
    
    endpoint_code = """
# TEMPORAL: Endpoint para descargar archivos del modelo desde Render
# ELIMINA ESTE ENDPOINT después de descargar los archivos

from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

@router.get("/api/download-model/{filename}")
async def download_model_file(filename: str):
    \"\"\"Descargar archivo del modelo (TEMPORAL)\"\"\"
    model_dir = 'app/ml_models'
    file_path = os.path.join(model_dir, filename)
    
    if not os.path.exists(file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type='application/octet-stream',
        filename=filename
    )
"""
    
    instructions = """
# INSTRUCCIONES PARA DESCARGAR MODELO DESDE RENDER

## Paso 1: Agregar endpoint temporal

1. Agrega este código a app/main.py o app/routes/analyses.py (temporalmente):

```python
# TEMPORAL: Endpoint para descargar archivos del modelo
from fastapi.responses import FileResponse
import os

@router.get("/api/download-model/{filename}")
async def download_model_file(filename: str):
    \"\"\"Descargar archivo del modelo (TEMPORAL)\"\"\"
    model_dir = 'app/ml_models'
    file_path = os.path.join(model_dir, filename)
    
    if not os.path.exists(file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type='application/octet-stream',
        filename=filename
    )
```

## Paso 2: Desplegar en Render

1. Haz commit y push del código
2. Espera a que Render despliegue
3. Espera a que el modelo se entrene (ver logs)

## Paso 3: Descargar archivos

Una vez que el modelo esté entrenado, descarga los archivos:

1. Ve a: https://tu-app-en-render.onrender.com/api/download-model/sentiment_model.h5
2. Ve a: https://tu-app-en-render.onrender.com/api/download-model/tokenizer.pkl
3. Ve a: https://tu-app-en-render.onrender.com/api/download-model/label_encoder.pkl

## Paso 4: Subir a GitHub Releases

1. Sigue los pasos en RESUMEN_PASOS_GITHUB.md
2. Sube los 3 archivos descargados a GitHub Releases

## Paso 5: Eliminar endpoint temporal

1. Elimina el endpoint temporal del código
2. Haz commit y push
3. Render ahora descargará desde GitHub Releases

## Alternativa: Usar Render Shell

Si tienes acceso a Render Shell:

1. Conéctate a Render Shell
2. Ve a: cd app/ml_models
3. Descarga los archivos usando scp o el método que Render proporcione
"""
    
    print("=" * 60)
    print("📥 INSTRUCCIONES PARA DESCARGAR MODELO DESDE RENDER")
    print("=" * 60)
    print()
    print(instructions)
    print()
    print("=" * 60)
    print("💡 RECOMENDACIÓN: Instala Visual C++ Redistributable")
    print("=" * 60)
    print("Es más rápido y fácil entrenar localmente:")
    print("1. Descarga: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Instala")
    print("3. Reinicia PowerShell")
    print("4. Ejecuta: python train_model_local.py")
    print()

if __name__ == "__main__":
    create_download_endpoint()


# ⚡ Solución Rápida: TensorFlow en Windows

## ❌ Problema
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

## ✅ Solución en 2 Minutos

### Paso 1: Descargar Visual C++ Redistributable
**URL directa:** https://aka.ms/vs/17/release/vc_redist.x64.exe

### Paso 2: Instalar
1. Ejecuta el archivo descargado
2. Acepta los términos
3. Clic en "Instalar"
4. Espera a que termine (30 segundos)

### Paso 3: Reiniciar PowerShell
1. Cierra PowerShell completamente
2. Ábrelo de nuevo
3. Ve a tu proyecto: `cd C:\Users\HP\Desktop\tareainteli`

### Paso 4: Activar entorno virtual y ejecutar
```powershell
.\venv\Scripts\Activate.ps1
python train_model_local.py
```

### Paso 5: Verificar
Deberías ver:
```
✅ MODELO ENTRENADO Y GUARDADO CORRECTAMENTE
✅ sentiment_model.h5
✅ tokenizer.pkl
✅ label_encoder.pkl
```

---

## 🎯 ¿Por Qué Esta Solución?

TensorFlow en Windows necesita las librerías de Visual C++ para funcionar. Sin ellas, no puede cargar las DLLs necesarias.

---

## ✅ Checklist

- [ ] Visual C++ Redistributable descargado
- [ ] Visual C++ Redistributable instalado
- [ ] PowerShell reiniciado
- [ ] Entorno virtual activado
- [ ] Script ejecutado: `python train_model_local.py`
- [ ] 3 archivos creados en `app/ml_models/`
- [ ] Archivos subidos a GitHub Releases

---

## 🚀 Siguiente Paso

Una vez que tengas los archivos:
1. Sigue los pasos en `RESUMEN_PASOS_GITHUB.md`
2. Sube los archivos a GitHub Releases
3. Render descargará automáticamente el modelo

---

## 💡 Alternativa

Si no puedes instalar Visual C++:
- Usa la Opción 2: Entrenar en Render
- Ver `download_model_from_render.py` para instrucciones


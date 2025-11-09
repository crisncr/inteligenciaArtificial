# 📋 Resumen: Pasos para Subir Modelo a GitHub Releases

## 🎯 Objetivo
El profesor dijo que la página web debe estar desplegada. Para que Render NO entrene el modelo cada vez (que consume tiempo y memoria), debemos subir el modelo pre-entrenado a GitHub Releases.

---

## ⚡ Pasos Rápidos (5 minutos)

### Paso 1: Entrenar el Modelo Localmente
```bash
python train_model_local.py
```

Esto crea 3 archivos:
- `app/ml_models/sentiment_model.h5`
- `app/ml_models/tokenizer.pkl`
- `app/ml_models/label_encoder.pkl`

---

### Paso 2: Ir a GitHub Releases

**URL Directa (MÁS FÁCIL):**
```
https://github.com/crisncr/inteligenciaArtificial/releases
```

**O desde el repositorio:**
1. Ve a: https://github.com/crisncr/inteligenciaArtificial
2. Busca el botón **"Releases"** en el menú superior (junto a "Code", "Issues")
3. O busca "Releases" en la barra lateral derecha

---

### Paso 3: Crear Nuevo Release

1. Haz clic en el botón verde **"Create a new release"** (arriba a la derecha)

---

### Paso 4: Configurar el Release

**Tag version:**
- Escribe: `v1.0.0`
- Si aparece "Create new tag: v1.0.0 on publish", selecciónalo

**Release title:**
- Escribe: `Modelo Pre-entrenado v1.0`

**Description:**
- Puedes dejarlo en blanco o escribir algo como: "Modelo de análisis de sentimientos"

---

### Paso 5: Subir los 3 Archivos

1. En la sección **"Attach binaries by dropping them here or selecting them"**:
   - **Arrastra** los 3 archivos desde `app/ml_models/`:
     - `sentiment_model.h5`
     - `tokenizer.pkl`
     - `label_encoder.pkl`
   
   **O** haz clic en **"selecting them"** y búscalos manualmente

2. Verifica que aparezcan los 3 archivos listados en la sección "Assets"

---

### Paso 6: Publicar el Release

1. Haz clic en el botón verde **"Publish release"** (abajo a la derecha)
2. Espera unos segundos hasta que GitHub procese el release

---

### Paso 7: Obtener las URLs de Descarga

Después de publicar, estarás en una página como:
```
https://github.com/crisncr/inteligenciaArtificial/releases/tag/v1.0.0
```

**Para obtener las URLs:**

1. Busca la sección **"Assets"** en la página (abajo)
2. Verás los 3 archivos listados:
   - `sentiment_model.h5`
   - `tokenizer.pkl`
   - `label_encoder.pkl`

3. **Para cada archivo:**
   - Haz **clic derecho** sobre el nombre del archivo
   - Selecciona **"Copy link address"** (Copiar dirección del enlace)
   - Guarda la URL

**Las URLs deberían verse así:**
```
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl
```

---

### Paso 8: Verificar las URLs

Abre cada URL en tu navegador. Deberías ver que:
- El archivo se descarga automáticamente
- O GitHub muestra información del archivo

**✅ Si el archivo se descarga = URL correcta**
**❌ Si ves una página de error = URL incorrecta**

---

## ✅ ¡Listo! No Necesitas Hacer Nada Más

**Las URLs ya están configuradas en el código por defecto**, así que:
- ✅ No necesitas cambiar el código
- ✅ No necesitas configurar variables de entorno (a menos que quieras)
- ✅ Render descargará automáticamente los archivos cuando la app inicie

---

## 📍 Dónde Están las URLs en el Código

Las URLs están en: `app/ml_models/sentiment_nn.py` (líneas 582-592)

```python
MODEL_URL = 'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5'
TOKENIZER_URL = 'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl'
LABEL_ENCODER_URL = 'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl'
```

---

## 🎨 Diagrama Visual de los Pasos

```
┌─────────────────────────────────────────────────────────┐
│ 1. Entrenar Modelo Localmente                           │
│    python train_model_local.py                          │
│    ↓                                                     │
│    Crea 3 archivos en app/ml_models/                   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Ir a GitHub Releases                                 │
│    https://github.com/crisncr/inteligenciaArtificial/   │
│    releases                                              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Crear Nuevo Release                                  │
│    Clic en "Create a new release"                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Configurar Release                                   │
│    Tag: v1.0.0                                          │
│    Title: Modelo Pre-entrenado v1.0                    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Subir 3 Archivos                                     │
│    Arrastra: sentiment_model.h5                         │
│             tokenizer.pkl                               │
│             label_encoder.pkl                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 6. Publicar Release                                     │
│    Clic en "Publish release"                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 7. Copiar URLs                                          │
│    Clic derecho en cada archivo →                       │
│    "Copy link address"                                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 8. ¡Listo!                                              │
│    Las URLs ya están en el código                       │
│    Render descargará automáticamente                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 Ubicación de "Releases" en GitHub (Visual)

### Opción 1: Menú Superior
```
┌─────────────────────────────────────────────────────────┐
│ [Code] [Issues] [Pull requests] [Releases] [Packages]  │
│                                    ↑                     │
│                            HAZ CLIC AQUÍ                │
└─────────────────────────────────────────────────────────┘
```

### Opción 2: Barra Lateral Derecha
```
┌─────────────────┐
│  About          │
│  Releases (2)   │  ← HAZ CLIC AQUÍ
│  Packages       │
│  Languages      │
│  ...            │
└─────────────────┘
```

### Opción 3: URL Directa (MÁS FÁCIL)
```
https://github.com/crisncr/inteligenciaArtificial/releases
```
**Solo copia y pega esta URL en tu navegador**

---

## 🐛 Problemas Comunes

### ❌ No encuentro el botón "Releases"
**Solución:** Ve directamente a:
```
https://github.com/crisncr/inteligenciaArtificial/releases
```

### ❌ No puedo arrastrar los archivos
**Solución:** 
1. Haz clic en "selecting them"
2. Busca los archivos en `app/ml_models/`
3. Selecciona los 3 archivos

### ❌ Las URLs no funcionan
**Solución:** Verifica que:
- El release esté publicado (no en draft)
- El tag sea `v1.0.0`
- Los archivos estén en la sección "Assets"
- La URL tenga el formato: `/releases/download/v1.0.0/ARCHIVO`

### ❌ No encuentro las URLs de descarga
**Solución:**
1. Ve a la página del release: `https://github.com/crisncr/inteligenciaArtificial/releases/tag/v1.0.0`
2. Busca la sección "Assets" (abajo)
3. Haz clic derecho en cada archivo → "Copy link address"

---

## ✅ Checklist Final

- [ ] Modelo entrenado (`python train_model_local.py`)
- [ ] 3 archivos creados en `app/ml_models/`
- [ ] Release creado en GitHub (tag v1.0.0)
- [ ] 3 archivos subidos al release
- [ ] Release publicado
- [ ] URLs copiadas (opcional, ya están en el código)
- [ ] URLs verificadas (descarga funciona)

---

## 🎉 Resultado Final

Después de seguir estos pasos:

1. ✅ El modelo estará en GitHub Releases
2. ✅ Render descargará automáticamente los archivos al iniciar
3. ✅ La aplicación NO entrenará el modelo cada vez
4. ✅ La aplicación cargará más rápido
5. ✅ Usará menos memoria

---

## 📚 Guías Adicionales

- **Guía completa:** `GUIA_GITHUB_RELEASES.md`
- **Pasos rápidos:** `PASOS_GITHUB_RELEASES_SIMPLE.md`
- **Este resumen:** `RESUMEN_PASOS_GITHUB.md`

---

## 💡 Por Qué Hacer Esto

**Sin GitHub Releases:**
- ❌ Render entrena el modelo cada vez que inicia
- ❌ Consume 30-60 segundos
- ❌ Usa mucha memoria
- ❌ Puede fallar o quedarse en "loading"

**Con GitHub Releases:**
- ✅ Render descarga el modelo (2-5 segundos)
- ✅ Usa menos memoria
- ✅ Carga más rápido
- ✅ Más confiable

---

## 🚀 Siguiente Paso

Una vez que hayas subido los archivos a GitHub Releases:
1. Haz commit y push del código
2. Render desplegará automáticamente
3. Verifica en los logs que dice "Descargando desde GitHub Releases" (no "Entrenando modelo")

---

¡Listo! Sigue estos pasos y tu aplicación estará desplegada correctamente. 🎉


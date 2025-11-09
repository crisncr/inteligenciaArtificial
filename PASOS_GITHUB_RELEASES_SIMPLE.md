# 🚀 Pasos Rápidos: Subir Modelo a GitHub Releases

## ⚡ Pasos en 5 Minutos

### 1️⃣ Entrenar el Modelo
```bash
python train_model_local.py
```
Esto crea 3 archivos en `app/ml_models/`:
- `sentiment_model.h5`
- `tokenizer.pkl`
- `label_encoder.pkl`

---

### 2️⃣ Ir a GitHub Releases

**Opción A: Desde el repositorio**
1. Ve a: https://github.com/crisncr/inteligenciaArtificial
2. Busca el botón **"Releases"** (arriba, junto a "Code", "Issues")
3. O busca "Releases" en el lado derecho de la página

**Opción B: URL Directa**
1. Ve directamente a: https://github.com/crisncr/inteligenciaArtificial/releases

---

### 3️⃣ Crear Nuevo Release

1. Haz clic en **"Create a new release"** (botón verde, arriba a la derecha)

---

### 4️⃣ Llenar los Campos

**Tag version:**
- Escribe: `v1.0.0`
- Si aparece "Create new tag: v1.0.0 on publish", selecciónalo

**Release title:**
- Escribe: `Modelo Pre-entrenado v1.0`

**Description (opcional):**
- Puedes dejar en blanco o escribir una descripción

---

### 5️⃣ Subir los 3 Archivos

1. En la sección **"Attach binaries"**:
   - Arrastra los 3 archivos desde `app/ml_models/`:
     - `sentiment_model.h5`
     - `tokenizer.pkl`
     - `label_encoder.pkl`
   
   **O** haz clic en "selecting them" y búscalos manualmente

2. Verifica que aparezcan los 3 archivos listados

---

### 6️⃣ Publicar

1. Haz clic en el botón verde **"Publish release"** (abajo a la derecha)
2. Espera unos segundos

---

### 7️⃣ Copiar las URLs

1. Después de publicar, estarás en la página del release
2. Busca la sección **"Assets"** (abajo)
3. Verás los 3 archivos listados

**Para cada archivo:**
1. Haz **clic derecho** sobre el nombre del archivo
2. Selecciona **"Copy link address"** (Copiar dirección del enlace)
3. Guarda la URL

**Las URLs deberían verse así:**
```
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl
```

---

### 8️⃣ Verificar las URLs

Abre cada URL en tu navegador. Deberías ver que:
- El archivo se descarga automáticamente
- O GitHub muestra información del archivo

---

## ✅ ¡Listo!

Las URLs ya están configuradas en el código por defecto, así que **no necesitas cambiar nada** en el código.

Render descargará automáticamente los archivos cuando la aplicación inicie.

---

## 🔍 Ubicación de "Releases" en GitHub

```
┌─────────────────────────────────────────┐
│  Code  Issues  Pull requests  Releases  │  ← Menú superior
└─────────────────────────────────────────┘

O

┌─────────────────┐
│  About          │
│  Releases (2)   │  ← Lado derecho
│  Packages       │
│  ...            │
└─────────────────┘
```

---

## 🐛 Si No Encuentras "Releases"

1. Ve directamente a: https://github.com/crisncr/inteligenciaArtificial/releases
2. O busca en la barra de búsqueda de GitHub: `crisncr/inteligenciaArtificial releases`

---

## 📝 Notas Importantes

- ✅ El release debe estar **publicado** (no en draft)
- ✅ Los archivos deben estar en la sección **"Assets"**
- ✅ Las URLs deben tener el formato: `/releases/download/v1.0.0/ARCHIVO`
- ✅ No subas los archivos al repositorio Git (solo al Release)

---

## 🎯 URLs Finales

Después de seguir estos pasos, deberías tener estas 3 URLs:

1. `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5`
2. `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl`
3. `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl`

**Estas URLs ya están en el código**, así que no necesitas hacer nada más. ✨


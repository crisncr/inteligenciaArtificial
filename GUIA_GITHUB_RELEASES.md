# 📦 Guía Paso a Paso: Subir Modelo a GitHub Releases

## 🎯 Objetivo
Subir los archivos del modelo entrenado a GitHub Releases para que Render los descargue automáticamente.

---

## 📋 Paso 1: Entrenar el Modelo Localmente

Primero, entrena el modelo en tu computadora:

```bash
python train_model_local.py
```

Esto creará 3 archivos en `app/ml_models/`:
- `sentiment_model.h5`
- `tokenizer.pkl`
- `label_encoder.pkl`

---

## 📤 Paso 2: Ir a tu Repositorio en GitHub

1. Abre tu navegador y ve a: **https://github.com/crisncr/inteligenciaArtificial**
2. Asegúrate de estar en la página principal de tu repositorio

---

## 🏷️ Paso 3: Crear un Nuevo Release

### 3.1. Encontrar la sección de Releases

En la página principal de tu repositorio, busca en el lado derecho (o arriba):

**Opción A: Barra lateral derecha**
- Busca la sección "Releases" en el lado derecho
- Haz clic en el número que aparece (ej: "2 releases" o "0 releases")

**Opción B: Menú superior**
- Haz clic en el botón **"Releases"** que aparece arriba (junto a "Code", "Issues", "Pull requests")

**Opción C: URL directa**
- Ve directamente a: **https://github.com/crisncr/inteligenciaArtificial/releases**

### 3.2. Crear el Release

1. Verás una página con los releases existentes (si hay alguno)
2. Haz clic en el botón **"Create a new release"** (o "Draft a new release")
   - Este botón está arriba a la derecha, en verde

---

## ✏️ Paso 4: Configurar el Release

En la página de creación del release, llena los siguientes campos:

### 4.1. Tag version (Etiqueta de versión)

1. Haz clic en el dropdown **"Choose a tag"**
2. Si no existe un tag v1.0.0, escribe: **`v1.0.0`**
3. GitHub te preguntará "Create new tag: v1.0.0 on publish"
   - Haz clic en **"Create new tag: v1.0.0 on publish"**

### 4.2. Release title (Título del Release)

Escribe un título, por ejemplo:
```
Modelo Pre-entrenado v1.0
```

### 4.3. Description (Descripción)

Escribe una descripción opcional, por ejemplo:
```
Modelo de análisis de sentimientos pre-entrenado para la aplicación de inteligencia artificial.

Archivos incluidos:
- sentiment_model.h5 (modelo entrenado)
- tokenizer.pkl (tokenizador)
- label_encoder.pkl (codificador de etiquetas)
```

### 4.4. Target (Rama objetivo)

Deja la opción por defecto: **`main`** (o `master`)

---

## 📎 Paso 5: Subir los Archivos

### 5.1. Arrastrar y Soltar los Archivos

1. En la sección **"Attach binaries by dropping them here or selecting them"**:
   - Arrastra los 3 archivos desde tu carpeta `app/ml_models/`:
     - `sentiment_model.h5`
     - `tokenizer.pkl`
     - `label_encoder.pkl`
   
   **O** haz clic en **"selecting them"** y busca los archivos manualmente

### 5.2. Verificar que los Archivos se Subieron

Después de subir, deberías ver los 3 archivos listados en la sección de binaries:
- ✅ sentiment_model.h5 (tamaño en KB/MB)
- ✅ tokenizer.pkl (tamaño en KB/MB)
- ✅ label_encoder.pkl (tamaño en KB/MB)

---

## 🚀 Paso 6: Publicar el Release

1. Haz clic en el botón verde **"Publish release"** (abajo a la derecha)
2. Espera a que GitHub procese y publique el release
3. Serás redirigido a la página del release publicado

---

## 🔗 Paso 7: Obtener las URLs de Descarga

### 7.1. Ir a la Página del Release

Después de publicar, estarás en una página como:
**https://github.com/crisncr/inteligenciaArtificial/releases/tag/v1.0.0**

### 7.2. Copiar las URLs de Descarga Directa

Para cada archivo, sigue estos pasos:

**Método 1: Clic derecho (RECOMENDADO)**
1. Busca la sección **"Assets"** en la página del release
2. Verás los 3 archivos listados:
   - `sentiment_model.h5`
   - `tokenizer.pkl`
   - `label_encoder.pkl`
3. Para cada archivo:
   - Haz **clic derecho** sobre el nombre del archivo
   - Selecciona **"Copy link address"** (o "Copiar dirección del enlace")
   - Pega la URL en un lugar seguro

**Método 2: Formato de URL Manual**

Las URLs deberían tener este formato:
```
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl
```

**Estructura de la URL:**
```
https://github.com/USUARIO/REPOSITORIO/releases/download/TAG/ARCHIVO
```

Donde:
- `USUARIO` = `crisncr`
- `REPOSITORIO` = `inteligenciaArtificial`
- `TAG` = `v1.0.0`
- `ARCHIVO` = nombre del archivo (ej: `sentiment_model.h5`)

---

## ✅ Paso 8: Verificar las URLs

### 8.1. Probar las URLs

Abre cada URL en tu navegador para verificar que funcionan:
- Deberías ver que el archivo se descarga automáticamente
- O ver información del archivo en GitHub

### 8.2. URLs Correctas vs Incorrectas

**✅ URL Correcta (descarga directa):**
```
https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5
```

**❌ URL Incorrecta (página del release):**
```
https://github.com/crisncr/inteligenciaArtificial/releases/tag/v1.0.0
```

**❌ URL Incorrecta (repositorio):**
```
https://github.com/crisncr/inteligenciaArtificial/blob/main/app/ml_models/sentiment_model.h5
```

---

## 🔧 Paso 9: Configurar las URLs en el Código (Opcional)

Las URLs ya están configuradas por defecto en `app/ml_models/sentiment_nn.py`:

```python
MODEL_URL = 'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5'
TOKENIZER_URL = 'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl'
LABEL_ENCODER_URL = 'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl'
```

**Si necesitas cambiar las URLs**, puedes:

**Opción A: Cambiar en el código**
- Edita `app/ml_models/sentiment_nn.py` líneas 582-592

**Opción B: Usar variables de entorno (RECOMENDADO para Render)**
- En Render, ve a tu servicio → Environment
- Agrega las variables:
  - `MODEL_URL` = `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.h5`
  - `TOKENIZER_URL` = `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl`
  - `LABEL_ENCODER_URL` = `https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl`

---

## 📸 Resumen Visual de los Pasos

```
1. GitHub Repositorio
   └── Clic en "Releases" (lado derecho o menú superior)

2. Página de Releases
   └── Clic en "Create a new release" (botón verde)

3. Crear Release
   ├── Tag: v1.0.0
   ├── Title: Modelo Pre-entrenado v1.0
   ├── Description: (opcional)
   └── Arrastrar 3 archivos:
       ├── sentiment_model.h5
       ├── tokenizer.pkl
       └── label_encoder.pkl

4. Publicar
   └── Clic en "Publish release"

5. Obtener URLs
   └── Clic derecho en cada archivo → "Copy link address"
```

---

## 🐛 Solución de Problemas

### Problema: No veo el botón "Releases"
- **Solución**: Ve directamente a: `https://github.com/crisncr/inteligenciaArtificial/releases`
- O busca en el menú superior junto a "Code", "Issues"

### Problema: No puedo arrastrar los archivos
- **Solución**: Haz clic en "selecting them" y busca los archivos manualmente
- Asegúrate de que los archivos existan en `app/ml_models/`

### Problema: Las URLs no funcionan
- **Solución**: Verifica que:
  - El release esté publicado (no en draft)
  - El tag sea correcto (v1.0.0)
  - Los archivos estén en la sección "Assets"
  - La URL tenga el formato: `/releases/download/TAG/ARCHIVO`

### Problema: No encuentro las URLs de descarga
- **Solución**: 
  - Ve a la página del release: `https://github.com/crisncr/inteligenciaArtificial/releases/tag/v1.0.0`
  - Busca la sección "Assets"
  - Haz clic derecho en cada archivo → "Copy link address"

---

## ✅ Checklist Final

- [ ] Modelo entrenado localmente (`train_model_local.py` ejecutado)
- [ ] 3 archivos creados en `app/ml_models/`
- [ ] Release creado en GitHub (tag v1.0.0)
- [ ] 3 archivos subidos al release
- [ ] Release publicado (no en draft)
- [ ] URLs de descarga copiadas y verificadas
- [ ] URLs probadas en el navegador (descarga funciona)
- [ ] Código actualizado (si es necesario)
- [ ] Variables de entorno configuradas en Render (opcional)

---

## 📞 Ayuda Adicional

Si tienes problemas:
1. Revisa los logs de Render para ver qué URLs está intentando descargar
2. Verifica que el release esté público (no privado)
3. Asegúrate de que los archivos no estén corruptos
4. Verifica que las URLs tengan el formato correcto

---

## 🎉 ¡Listo!

Una vez que hayas subido los archivos a GitHub Releases, Render los descargará automáticamente al iniciar la aplicación, evitando que tenga que entrenar el modelo cada vez.


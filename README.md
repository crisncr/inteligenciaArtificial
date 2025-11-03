# Motor de Inferencia de Sentimientos (ES)

Aplicación web simple con FastAPI para analizar el sentimiento de frases en español (positivo, negativo o moderado/neutral) usando un motor basado en reglas.

## Requisitos

- Python 3.9+
- PowerShell (Windows) o terminal equivalente

## Instalación

```bash
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Ejecución

```bash
uvicorn app.main:app --reload
```

Abre `http://127.0.0.1:8000` en tu navegador.

## Estructura

- `app/main.py`: servidor FastAPI, rutas y templates
- `app/sentiment.py`: motor de análisis de sentimiento (reglas en español)
- `app/templates/index.html`: interfaz web
- `app/static/styles.css`: estilos

## Despliegue en Render

1. **Sube tu código a GitHub** (si no lo has hecho):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/tu-usuario/tu-repo.git
   git push -u origin main
   ```

2. **En Render:**
   - Ve a https://render.com
   - Crea una cuenta o inicia sesión
   - Click en "New +" → "Web Service"
   - Conecta tu repositorio de GitHub
   - Render detectará automáticamente el `render.yaml` o puedes configurarlo manualmente:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT`
   - Selecciona "Free" tier
   - Click "Create Web Service"

3. **Espera el despliegue** (puede tomar 2-5 minutos)
   - Render construirá tu app automáticamente
   - Obtendrás una URL pública como: `https://motor-sentimientos.onrender.com`

## Notas

- El motor usa listas de palabras positivas/negativas, negaciones e intensificadores sencillos.
- Puedes ampliar los diccionarios en `app/sentiment.py` para mejorar la cobertura.


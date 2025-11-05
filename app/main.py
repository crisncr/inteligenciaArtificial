from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from app.sentiment import analyze_sentiment
from app.database import engine, Base
from app.routes import auth as auth_router
from app.routes import analyses as analyses_router

# Crear tablas en la base de datos
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Motor de Inferencia de Sentimientos", version="1.0.0")

# CORS para desarrollo local y pruebas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# En producción: servir React build
# Buscar el build en diferentes ubicaciones posibles
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Raíz del proyecto
app_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio app/

dist_paths = [
    os.path.join(app_dir, "static", "dist"),  # app/static/dist (absoluto)
    os.path.join(base_dir, "app", "static", "dist"),  # Raíz/app/static/dist
    "app/static/dist",  # Relativo desde cwd
    "./app/static/dist",  # Relativo con ./
]

dist_path = None
for path in dist_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path) and os.path.isdir(abs_path):
        dist_path = abs_path
        break

if dist_path:
    assets_path = os.path.join(dist_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
    @app.get("/")
    async def index():
        index_file = os.path.join(dist_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"error": "index.html no encontrado en build"})
    @app.get("/favicon.svg")
    async def favicon():
        favicon_path = "public/favicon.svg"
        if os.path.exists(favicon_path):
            return FileResponse(favicon_path)
        return JSONResponse({"error": "favicon no encontrado"})
else:
    # Desarrollo: mensaje informativo o fallback
    @app.get("/")
    async def index_dev():
        # Verificar si existe el directorio app/static
        static_exists = os.path.exists(os.path.join(app_dir, "static"))
        static_list = []
        if static_exists:
            try:
                static_list = os.listdir(os.path.join(app_dir, "static"))
            except:
                pass
        
        return JSONResponse({
            "message": "Frontend React no construido",
            "debug": {
                "checked_paths": [os.path.abspath(p) for p in dist_paths],
                "cwd": os.getcwd(),
                "app_dir": app_dir,
                "base_dir": base_dir,
                "static_exists": static_exists,
                "static_contents": static_list,
            },
            "instructions": "Para desarrollo: ejecuta 'npm run dev' en otra terminal (puerto 5173)",
            "instructions_prod": "Para producción: ejecuta 'npm run build' y luego reinicia el servidor",
            "api": "La API está disponible en /analyze",
            "health": "Health check en /health"
        })
    # Si no existe app/static, crear estructura básica
    if not os.path.exists("app/static"):
        os.makedirs("app/static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@app.get("/robots.txt")
async def robots() -> JSONResponse:
    content = "User-agent: *\nAllow: /\nSitemap: /sitemap.xml\n"
    return JSONResponse(content=content, media_type="text/plain; charset=utf-8")

@app.get("/sitemap.xml")
async def sitemap() -> JSONResponse:
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n"
        "  <url><loc>/</loc></url>\n"
        "</urlset>\n"
    )
    return JSONResponse(content=xml, media_type="application/xml; charset=utf-8")

# Incluir routers
app.include_router(auth_router.router)
app.include_router(analyses_router.router)

# Endpoint público para análisis (sin autenticación, límite de 3)
@app.post("/analyze")
async def analyze_public(payload: dict = Body(...)):
    """Endpoint público para análisis (máximo 3 análisis gratuitos)"""
    text = (payload.get("text") or "").strip()
    result = analyze_sentiment(text)
    return JSONResponse(result)


# Conveniencia para `uvicorn app.main:app --reload`
def get_app() -> FastAPI:
    return app

if __name__ == "__main__":
    # Permite iniciar el servicio con: python -m app.main
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



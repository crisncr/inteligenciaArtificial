from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from app.sentiment import analyze_sentiment

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
if os.path.exists("app/static/dist"):
    app.mount("/assets", StaticFiles(directory="app/static/dist/assets"), name="assets")
    @app.get("/")
    async def index():
        return FileResponse("app/static/dist/index.html")
    @app.get("/favicon.svg")
    async def favicon():
        return FileResponse("public/favicon.svg")
else:
    # Desarrollo: mensaje informativo o fallback
    @app.get("/")
    async def index_dev():
        return JSONResponse({
            "message": "Frontend React no construido",
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

@app.post("/analyze")
async def analyze(payload: dict = Body(...)):
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



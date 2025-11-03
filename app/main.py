from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

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

# Archivos estáticos y templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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



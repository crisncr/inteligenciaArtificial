from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from app.sentiment import analyze_sentiment
from app.database import engine, Base, SessionLocal
from app.routes import auth as auth_router
from app.routes import analyses as analyses_router
from app.routes import external_api as external_api_router
from app.routes import payments as payments_router
from app.routes import datasets as datasets_router
from app.routes import route_optimization as route_optimization_router
from app.routes import sales_prediction as sales_prediction_router

# Importar todos los modelos para que SQLAlchemy los registre
from app.models import User, Analysis, Plan, Payment, PasswordResetToken, EmailVerificationToken, ExternalAPI, Route, RoutePoint

# Ejecutar migraciones de Alembic al iniciar
def run_migrations():
    """Ejecutar migraciones de Alembic al iniciar la aplicación"""
    try:
        from alembic.config import Config
        from alembic import command
        import os
        
        # Obtener DATABASE_URL de variables de entorno
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/sentimetria")
        
        # Convertir formato de Render si es necesario (postgres:// -> postgresql://)
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        # Configurar Alembic
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        
        # Ejecutar migraciones
        print("🔄 Ejecutando migraciones de Alembic...")
        command.upgrade(alembic_cfg, "head")
        print("✅ Migraciones de Alembic ejecutadas correctamente")
    except Exception as e:
        print(f"⚠️ Error al ejecutar migraciones de Alembic: {e}")
        # Si falla, intentar crear las columnas manualmente
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                # Verificar si la columna source existe
                result = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='analyses' AND column_name='source'
                """))
                if result.fetchone() is None:
                    print("🔄 Agregando columnas source y external_api_id manualmente...")
                    # Agregar columna source
                    conn.execute(text("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'manual' NOT NULL"))
                    # Agregar columna external_api_id
                    conn.execute(text("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS external_api_id INTEGER"))
                    # Agregar foreign key si existe la tabla external_apis
                    try:
                        conn.execute(text("""
                            ALTER TABLE analyses 
                            ADD CONSTRAINT IF NOT EXISTS fk_analyses_external_api_id 
                            FOREIGN KEY (external_api_id) REFERENCES external_apis(id) ON DELETE SET NULL
                        """))
                    except Exception as fk_error:
                        print(f"⚠️ No se pudo agregar foreign key (puede que la tabla external_apis no exista aún): {fk_error}")
                    conn.commit()
                    print("✅ Columnas agregadas manualmente")
                else:
                    print("✅ Las columnas ya existen")
        except Exception as manual_error:
            print(f"⚠️ Error al agregar columnas manualmente: {manual_error}")

# Ejecutar migraciones al iniciar
try:
    run_migrations()
except Exception as e:
    print(f"⚠️ Error al ejecutar migraciones al iniciar: {e}")

# Crear tablas en la base de datos (después de importar los modelos)
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Tablas de base de datos creadas correctamente")
except Exception as e:
    print(f"⚠️ Error al crear tablas: {e}")

# Normalizar emails existentes a minúsculas (una sola vez al iniciar)
def normalize_existing_emails():
    """Normaliza todos los emails existentes en la base de datos a minúsculas"""
    db = SessionLocal()
    try:
        # Obtener todos los usuarios
        users = db.query(User).all()
        updated_count = 0
        for user in users:
            if user.email and user.email != user.email.lower():
                # Email tiene mayúsculas, normalizar a minúsculas
                old_email = user.email
                user.email = user.email.lower().strip()
                updated_count += 1
                print(f"📧 Normalizando email: {old_email} -> {user.email}")
        
        if updated_count > 0:
            db.commit()
            print(f"✅ {updated_count} email(s) normalizado(s) a minúsculas")
        else:
            print("✅ Todos los emails ya están normalizados")
    except Exception as e:
        print(f"⚠️ Error al normalizar emails: {e}")
        db.rollback()
    finally:
        db.close()

# Normalizar emails existentes al iniciar
try:
    normalize_existing_emails()
except Exception as e:
    print(f"⚠️ Error al normalizar emails al iniciar: {e}")

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
    
    # Servir imágenes del carrusel
    @app.get("/images/{filename}")
    async def serve_image(filename: str):
        image_path = os.path.join("public", "images", filename)
        if os.path.exists(image_path):
            return FileResponse(image_path)
        return JSONResponse({"error": f"Imagen {filename} no encontrada"}, status_code=404)
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
app.include_router(external_api_router.router)
app.include_router(payments_router.router)
app.include_router(datasets_router.router)
app.include_router(route_optimization_router.router)
app.include_router(sales_prediction_router.router)

# Configurar TensorFlow para modo ULTRA-LIGERO (512 MB limit en Render)
@app.on_event("startup")
async def startup_event():
    """Configurar TensorFlow para modo ULTRA-LIGERO (512 MB limit) y precargar modelo"""
    import os
    import threading
    
    # Configurar TensorFlow para usar MÍNIMA memoria (Render tiene 512 MB limit)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # Limitar uso de memoria
    os.environ['TF_MEMORY_ALLOCATION'] = '0.3'  # Usar solo 30% de memoria disponible
    
    try:
        import tensorflow as tf
        # Limitar threads para usar menos memoria
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        # Deshabilitar optimizaciones que usan más memoria
        tf.config.optimizer.set_jit(False)
        
        # Configurar límite de memoria para GPU (si existe)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Limitar crecimiento de memoria GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"⚠️ No se pudo configurar GPU: {e}")
        
        print("✅ TensorFlow configurado para modo ULTRA-LIGERO (512 MB limit)")
    except Exception as e:
        print(f"⚠️ No se pudo configurar TensorFlow: {e}")
    
    # Precargar modelo en thread separado (no bloquea el startup)
    def precargar_modelo():
        try:
            from app.sentiment import _train_model_async
            print("🚀 Iniciando precarga del modelo en background (versión ultra-ligera)...")
            _train_model_async()
            print("✅ Modelo precargado correctamente")
        except Exception as e:
            print(f"⚠️ Error al precargar modelo (se cargará en el primer request): {e}")
            import traceback
            traceback.print_exc()
    
    # Iniciar thread de precarga (daemon=True para que no bloquee el cierre)
    thread = threading.Thread(target=precargar_modelo, daemon=True, name="ModelPreloader")
    thread.start()
    print("✅ Thread de precarga iniciado (no bloquea el servidor)")


# Endpoint público para análisis (sin autenticación, límite de 3)
@app.post("/analyze")
async def analyze_public(payload: dict = Body(...)):
    """Endpoint público para análisis usando Red Neuronal LSTM (máximo 3 análisis gratuitos)"""
    try:
        text = (payload.get("text") or "").strip()
        if not text:
            return JSONResponse(
                {"error": "El texto a analizar no puede estar vacío"},
                status_code=400
            )
        
        # Usa analyze_sentiment que ahora SOLO usa red neuronal LSTM
        result = analyze_sentiment(text)
        return JSONResponse(result)
    except Exception as e:
        error_msg = str(e)
        # Retornar error en formato JSON
        return JSONResponse(
            {
                "error": error_msg,
                "sentiment": "error",
                "score": 0.0,
                "emoji": "⚠️"
            },
            status_code=500
        )


# Conveniencia para `uvicorn app.main:app --reload`
def get_app() -> FastAPI:
    return app

if __name__ == "__main__":
    # Permite iniciar el servicio con: python -m app.main
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



"""
Módulo de análisis de sentimientos usando SOLO red neuronal LSTM.

Este módulo proporciona análisis de sentimientos usando exclusivamente
una red neuronal LSTM que captura contexto y relaciones semánticas.

El método de diccionario ha sido completamente eliminado.
Todo el análisis ahora se realiza usando redes neuronales.
"""
from typing import Dict

# Instancia global del modelo para evitar reentrenarlo en cada request
_global_model = None
_model_lock = False

def _get_or_create_model():
    """Obtener o crear instancia global del modelo"""
    global _global_model, _model_lock
    
    if _global_model is not None and _global_model.is_trained:
        return _global_model
    
    # Si otro proceso está cargando el modelo, esperar (con timeout más corto)
    if _model_lock:
        import time
        max_wait = 180  # 3 minutos máximo (reducido de 5)
        waited = 0
        print("⏳ Esperando que el modelo termine de cargarse...")
        while _model_lock and waited < max_wait:
            time.sleep(2)  # Esperar 2 segundos entre checks
            waited += 2
            if _global_model is not None and _global_model.is_trained:
                print("✅ Modelo listo después de esperar")
                return _global_model
            if waited % 30 == 0:  # Log cada 30 segundos
                print(f"⏳ Todavía cargando modelo... ({waited}s / {max_wait}s)")
        
        # Si aún no está listo después del timeout, lanzar error
        if _global_model is None or not _global_model.is_trained:
            raise Exception(
                "El modelo está tardando demasiado en cargarse. "
                "Por favor, espera unos minutos e intenta de nuevo. "
                "El modelo se está entrenando por primera vez y esto puede tomar 2-3 minutos."
            )
    
    try:
        _model_lock = True
        from app.ml_models.sentiment_nn import SentimentNeuralNetwork
        print("🔄 Inicializando modelo de red neuronal...")
        _global_model = SentimentNeuralNetwork()
        _global_model.load_model()
        print("✅ Modelo de red neuronal listo y entrenado")
        return _global_model
    except Exception as e:
        print(f"❌ Error al cargar modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        _global_model = None
        raise
    finally:
        _model_lock = False


def analyze_sentiment(text: str) -> Dict[str, object]:
    """
    Analizar sentimiento usando SOLO red neuronal LSTM.
    
    Este es el método único y exclusivo para análisis de sentimientos.
    Usa una red neuronal LSTM que captura contexto y relaciones semánticas.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Dict con 'text', 'sentiment', 'score', 'emoji', 'method'
        
    Raises:
        Exception: Si no se puede cargar o usar el modelo de red neuronal
    """
    if not text or not text.strip():
        raise Exception("El texto a analizar no puede estar vacío")
    
    try:
        # Verificar si el modelo se está cargando/entrenando
        global _model_lock
        if _model_lock:
            raise Exception(
                "El modelo de red neuronal se está cargando o entrenando. "
                "Esto solo ocurre la primera vez que se inicia la aplicación y puede tomar 10-20 segundos. "
                "Por favor, espera unos momentos e intenta de nuevo."
            )
        
        # Usar modelo global para evitar reentrenarlo
        model = _get_or_create_model()
        
        if not model:
            raise Exception("No se pudo cargar el modelo de red neuronal")
        
        if not model.is_trained:
            raise Exception(
                "El modelo no está entrenado correctamente. "
                "El modelo se está entrenando por primera vez. Por favor, espera unos momentos e intenta de nuevo."
            )
        
        result = model.predict_single(text)
        # Marcar que se usó red neuronal
        result['method'] = 'neural_network'
        return result
    except ImportError as e:
        raise Exception(
            "Error: No se pudo importar el modelo de red neuronal. "
            "Asegúrate de que TensorFlow esté instalado correctamente. "
            f"Detalle: {str(e)}"
        )
    except ValueError as e:
        # Errores de validación del modelo
        error_msg = str(e)
        if "no está entrenado" in error_msg.lower():
            raise Exception(
                "El modelo de red neuronal está cargándose. Por favor, espera unos momentos e intenta de nuevo."
            )
        raise Exception(f"Error en el modelo: {error_msg}")
    except Exception as e:
        error_msg = str(e)
        # Mejorar mensajes de error
        if "no está entrenado" in error_msg.lower() or "no está disponible" in error_msg.lower():
            raise Exception(
                "El modelo de red neuronal está cargándose. Por favor, espera unos momentos e intenta de nuevo."
            )
        if "tardando demasiado" in error_msg.lower():
            raise Exception(
                "El modelo está tardando en cargarse. Por favor, intenta de nuevo en unos momentos."
            )
        raise Exception(
            f"Error al analizar sentimiento: {error_msg}"
        )

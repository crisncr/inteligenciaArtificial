"""
Módulo de análisis de sentimientos usando SOLO red neuronal LSTM.

Este módulo proporciona análisis de sentimientos usando exclusivamente
una red neuronal LSTM que captura contexto y relaciones semánticas.

El método de diccionario ha sido completamente eliminado.
Todo el análisis ahora se realiza usando redes neuronales.
"""
from typing import Dict
import threading

# Instancia global del modelo para evitar reentrenarlo en cada request
_global_model = None
_model_lock = False
_training_thread = None

def _train_model_async():
    """Entrenar modelo en un thread separado (no bloqueante)"""
    global _global_model, _model_lock
    
    try:
        _model_lock = True
        from app.ml_models.sentiment_nn import SentimentNeuralNetwork
        print("🔄 [Thread] Inicializando modelo de red neuronal...")
        _global_model = SentimentNeuralNetwork()
        _global_model.load_model()
        print("✅ [Thread] Modelo de red neuronal listo y entrenado")
    except Exception as e:
        print(f"❌ [Thread] Error al cargar modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        _global_model = None
    finally:
        _model_lock = False

def _get_or_create_model():
    """Obtener o crear instancia global del modelo - NO BLOQUEANTE"""
    global _global_model, _model_lock, _training_thread
    
    # Si el modelo ya está entrenado, devolverlo inmediatamente
    if _global_model is not None and _global_model.is_trained:
        return _global_model
    
    # Si el modelo se está entrenando, NO ESPERAR - lanzar error inmediatamente
    if _model_lock:
        raise Exception(
            "El modelo de red neuronal se está cargando o entrenando. "
            "Esto solo ocurre la primera vez que se inicia la aplicación y puede tomar 10-20 segundos. "
            "Por favor, espera unos momentos e intenta de nuevo."
        )
    
    # Si el modelo no existe y no se está entrenando, iniciar entrenamiento en thread separado
    if _global_model is None:
        print("🚀 Iniciando entrenamiento del modelo en thread separado (no bloqueante)...")
        _training_thread = threading.Thread(target=_train_model_async, daemon=True, name="ModelTrainer")
        _training_thread.start()
        # NO ESPERAR - lanzar error inmediatamente para que la request no se bloquee
        raise Exception(
            "El modelo de red neuronal se está cargando por primera vez. "
            "Esto puede tomar 10-20 segundos. Por favor, espera unos momentos e intenta de nuevo."
        )
    
    # Si el modelo existe pero no está entrenado, también lanzar error
    if not _global_model.is_trained:
        raise Exception(
            "El modelo de red neuronal aún se está entrenando. "
            "Por favor, espera unos momentos e intenta de nuevo."
        )
    
    return _global_model


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
        # Obtener modelo (puede lanzar excepción si no está listo - NO BLOQUEA)
        model = _get_or_create_model()
        
        # Si llegamos aquí, el modelo está listo y entrenado
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
    except Exception as e:
        # Re-lanzar excepciones del modelo (ya tienen mensajes informativos)
        error_msg = str(e)
        raise Exception(error_msg)

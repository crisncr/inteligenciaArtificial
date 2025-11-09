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
    
    # Si otro proceso está cargando el modelo, esperar
    if _model_lock:
        import time
        while _model_lock:
            time.sleep(0.1)
        if _global_model is not None and _global_model.is_trained:
            return _global_model
    
    try:
        _model_lock = True
        from app.ml_models.sentiment_nn import SentimentNeuralNetwork
        _global_model = SentimentNeuralNetwork()
        _global_model.load_model()
        return _global_model
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
    try:
        # Usar modelo global para evitar reentrenarlo
        model = _get_or_create_model()
        
        if not model or not model.is_trained:
            raise Exception("El modelo no está entrenado correctamente")
        
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
        error_msg = str(e)
        # Mejorar mensajes de error
        if "El modelo no está entrenado" in error_msg:
            raise Exception(
                "Error: El modelo de red neuronal no está disponible. "
                "Por favor, intenta de nuevo en unos momentos mientras se carga el modelo."
            )
        raise Exception(
            f"Error al analizar sentimiento: {error_msg}"
        )

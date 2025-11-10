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
        _global_model = SentimentNeuralNetwork()
        _global_model.load_model()
        
        # Validación mínima (sin predicción de prueba para mejor rendimiento)
        if not _global_model.is_trained or not _global_model.model:
            raise Exception("El modelo no se cargó correctamente")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {str(e)}")
        _global_model = None
        _model_lock = False
    finally:
        if _model_lock:
            _model_lock = False

def _get_or_create_model():
    """Obtener o crear instancia global del modelo - Espera razonable si se está entrenando"""
    global _global_model, _model_lock, _training_thread
    
    # Si el modelo ya está entrenado, devolverlo inmediatamente (sin logs para mejor rendimiento)
    if _global_model is not None and _global_model.is_trained:
        return _global_model
    
    # Si el modelo se está entrenando, esperar un poco (pero no bloquear mucho)
    if _model_lock:
        import time
        max_wait = 90  # Esperar máximo 90 segundos
        waited = 0
        while _model_lock and waited < max_wait:
            time.sleep(1)
            waited += 1
            if _global_model is not None and _global_model.is_trained:
                return _global_model
        
        # Si después de esperar no está listo, lanzar error
        raise Exception(
            f"El modelo se está cargando pero ha tardado más de {max_wait} segundos. "
            "Por favor, espera unos segundos e intenta de nuevo."
        )
    
    # Si el modelo no existe, iniciar carga
    if _global_model is None:
        _training_thread = threading.Thread(target=_train_model_async, daemon=True, name="ModelTrainer")
        _training_thread.start()
        raise Exception(
            "El modelo se está cargando por primera vez. Por favor, espera unos momentos e intenta de nuevo."
        )
    
    # Si el modelo existe pero no está entrenado
    if not _global_model.is_trained:
        raise Exception(
            "El modelo aún se está cargando. Por favor, espera unos momentos."
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
        # Obtener modelo (puede lanzar excepción si no está listo)
        model = _get_or_create_model()
        
        # Validación rápida (solo si es necesario)
        if model is None or not model.is_trained or not model.model:
            raise Exception(
                "El modelo de red neuronal no está disponible. "
                "Por favor, espera unos momentos e intenta de nuevo."
            )
        
        # Hacer predicción con la red neuronal LSTM (sin logs para mejor rendimiento)
        result = model.predict_single(text)
        
        # Validación mínima del resultado
        if not result or 'sentiment' not in result:
            raise Exception("El modelo no devolvió un resultado válido")
        
        # Marcar que se usó red neuronal (NO diccionario)
        result['method'] = 'neural_network'
        return result
        
    except ValueError as e:
        # Errores de validación del modelo
        error_msg = str(e)
        if "no está entrenado" in error_msg.lower() or "no está inicializado" in error_msg.lower():
            raise Exception(
                "El modelo de red neuronal se está cargando o entrenando. "
                "Por favor, espera unos momentos e intenta de nuevo."
            )
        raise Exception(f"Error en el modelo de red neuronal: {error_msg}")
    except ImportError as e:
        raise Exception(
            "Error: No se pudo importar TensorFlow. "
            f"Detalle: {str(e)}"
        )
    except Exception as e:
        error_msg = str(e)
        # Mejorar mensajes de error
        if "cargando" in error_msg.lower() or "entrenando" in error_msg.lower():
            raise Exception(error_msg)
        raise Exception(f"Error al analizar con red neuronal: {error_msg}")

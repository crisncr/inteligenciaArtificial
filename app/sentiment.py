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
        import os
        # En producción (Render), esperar menos tiempo para evitar timeout
        is_production = os.getenv('RENDER') == 'true' or os.getenv('ENVIRONMENT') == 'production'
        max_wait = 10 if is_production else 90  # En producción solo 10 segundos
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
        import os
        is_production = os.getenv('RENDER') == 'true' or os.getenv('ENVIRONMENT') == 'production'
        # En producción, no esperar - devolver error inmediatamente
        if is_production:
            raise Exception(
                "El modelo no está disponible. El servidor puede estar iniciando. "
                "Por favor, espera 30-60 segundos e intenta de nuevo."
            )
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
    import time
    start_time = time.time()
    
    # Reducir logs en producción para ahorrar memoria
    import os
    is_production = os.getenv('RENDER') == 'true' or os.getenv('ENVIRONMENT') == 'production'
    
    if not is_production:
        print(f"🔍 [SENTIMENT] Iniciando análisis individual - Texto: '{text[:50]}...'")
    
    if not text or not text.strip():
        if not is_production:
            print(f"❌ [SENTIMENT] Error: Texto vacío")
        raise Exception("El texto a analizar no puede estar vacío")
    
    try:
        if not is_production:
            print(f"⏳ [SENTIMENT] Obteniendo modelo...")
        model_start = time.time()
        # Obtener modelo (puede lanzar excepción si no está listo)
        model = _get_or_create_model()
        model_time = time.time() - model_start
        if not is_production:
            print(f"✅ [SENTIMENT] Modelo obtenido en {model_time:.2f}s")
        
        # Validación rápida (solo si es necesario)
        if model is None or not model.is_trained or not model.model:
            if not is_production:
                print(f"❌ [SENTIMENT] Error: Modelo no disponible")
            raise Exception(
                "El modelo de red neuronal no está disponible. "
                "Por favor, espera unos momentos e intenta de nuevo."
            )
        
        if not is_production:
            print(f"✅ [SENTIMENT] Modelo validado - is_trained={model.is_trained}, model_exists={model.model is not None}")
        
        # Hacer predicción con la red neuronal LSTM
        if not is_production:
            print(f"🧠 [SENTIMENT] Iniciando predicción con LSTM...")
        predict_start = time.time()
        result = model.predict_single(text)
        predict_time = time.time() - predict_start
        if not is_production:
            print(f"✅ [SENTIMENT] Predicción completada en {predict_time:.2f}s")
        
        # Validación mínima del resultado
        if not result or 'sentiment' not in result:
            if not is_production:
                print(f"❌ [SENTIMENT] Error: Resultado inválido")
            raise Exception("El modelo no devolvió un resultado válido")
        
        total_time = time.time() - start_time
        sentiment = result.get('sentiment', 'unknown')
        confidence = result.get('confidence', 0.0)
        if not is_production:
            print(f"✅ [SENTIMENT] Análisis completado en {total_time:.2f}s - Sentimiento: {sentiment}, Confianza: {confidence:.3f}")
        
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

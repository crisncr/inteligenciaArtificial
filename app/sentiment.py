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
        print("🔍 [DEBUG] [Thread] _model_lock = True")
        from app.ml_models.sentiment_nn import SentimentNeuralNetwork
        print("✅ [DEBUG] [Thread] SentimentNeuralNetwork importado correctamente")
        print("🔄 [Thread] Inicializando modelo de red neuronal...")
        _global_model = SentimentNeuralNetwork()
        print(f"🔍 [DEBUG] [Thread] Modelo creado: is_trained={_global_model.is_trained}")
        print("🔄 [Thread] Cargando modelo...")
        _global_model.load_model()
        print(f"🔍 [DEBUG] [Thread] Modelo cargado: is_trained={_global_model.is_trained}")
        
        # Validación adicional
        if not _global_model.is_trained:
            raise Exception("El modelo no se marcó como entrenado después de load_model()")
        if not _global_model.model:
            raise Exception("El modelo no tiene el atributo model después de load_model()")
        
        print("✅ [Thread] Modelo de red neuronal listo y entrenado")
    except Exception as e:
        print(f"❌ [Thread] Error al cargar modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        _global_model = None
    finally:
        _model_lock = False
        print("🔍 [DEBUG] [Thread] _model_lock = False")

def _get_or_create_model():
    """Obtener o crear instancia global del modelo - Espera razonable si se está entrenando"""
    global _global_model, _model_lock, _training_thread
    
    print(f"🔍 [DEBUG] _get_or_create_model() llamado")
    print(f"🔍 [DEBUG] Estado: _global_model={_global_model is not None}, _model_lock={_model_lock}")
    
    # Si el modelo ya está entrenado, devolverlo inmediatamente
    if _global_model is not None and _global_model.is_trained:
        print("✅ [DEBUG] Modelo ya está entrenado, devolviendo...")
        return _global_model
    
    # Si el modelo se está entrenando, esperar un poco (pero no bloquear mucho)
    if _model_lock:
        import time
        max_wait = 60  # Esperar máximo 60 segundos
        waited = 0
        print("⏳ [DEBUG] Esperando que el modelo termine de cargarse...")
        while _model_lock and waited < max_wait:
            time.sleep(1)
            waited += 1
            if waited % 5 == 0:  # Log cada 5 segundos
                print(f"⏳ [DEBUG] Esperando... {waited}s / {max_wait}s")
            if _global_model is not None and _global_model.is_trained:
                print("✅ [DEBUG] Modelo listo después de esperar")
                return _global_model
        
        # Si después de esperar no está listo, lanzar error
        print(f"❌ [DEBUG] Timeout esperando modelo: {waited}s")
        raise Exception(
            f"El modelo se está cargando pero ha tardado más de {max_wait} segundos. "
            "Por favor, espera unos segundos e intenta de nuevo."
        )
    
    # Si el modelo no existe, iniciar entrenamiento
    if _global_model is None:
        print("🚀 [DEBUG] Iniciando entrenamiento del modelo en thread separado...")
        _training_thread = threading.Thread(target=_train_model_async, daemon=True, name="ModelTrainer")
        _training_thread.start()
        print("🚀 [DEBUG] Thread de entrenamiento iniciado")
        raise Exception(
            "El modelo se está cargando por primera vez. Esto tomará 15-30 segundos. "
            "Por favor, espera unos momentos e intenta de nuevo."
        )
    
    # Si el modelo existe pero no está entrenado
    if not _global_model.is_trained:
        print(f"❌ [DEBUG] Modelo existe pero no está entrenado: is_trained={_global_model.is_trained}")
        raise Exception(
            "El modelo aún se está entrenando. Por favor, espera unos momentos."
        )
    
    print("✅ [DEBUG] Devolviendo modelo")
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
    print(f"🔍 [DEBUG] analyze_sentiment() llamado con texto: '{text[:50]}...'")
    
    if not text or not text.strip():
        print("❌ [DEBUG] Error: texto vacío")
        raise Exception("El texto a analizar no puede estar vacío")
    
    try:
        print("🔍 [DEBUG] Obteniendo modelo...")
        # Obtener modelo (puede lanzar excepción si no está listo)
        model = _get_or_create_model()
        print(f"🔍 [DEBUG] Modelo obtenido: is_trained={model.is_trained if model else None}, model={model is not None}")
        
        # Validar que el modelo esté completamente listo
        if model is None:
            print("❌ [DEBUG] Error: modelo es None")
            raise Exception(
                "El modelo de red neuronal no está disponible. "
                "El modelo se está cargando. Por favor, espera unos momentos e intenta de nuevo."
            )
        
        if not model.is_trained:
            print("❌ [DEBUG] Error: modelo no está entrenado")
            raise Exception(
                "El modelo de red neuronal aún no está entrenado. "
                "El modelo se está entrenando. Por favor, espera unos momentos e intenta de nuevo."
            )
        
        if not model.model:
            print("❌ [DEBUG] Error: model.model es None")
            raise Exception(
                "El modelo de red neuronal no está inicializado correctamente. "
                "Por favor, espera unos momentos e intenta de nuevo."
            )
        
        print("🔍 [DEBUG] Llamando a model.predict_single()...")
        # Hacer predicción con la red neuronal LSTM
        result = model.predict_single(text)
        print(f"🔍 [DEBUG] Resultado recibido: {result}")
        
        # Validar resultado
        if not result:
            print("❌ [DEBUG] Error: resultado es None o vacío")
            raise Exception("El modelo no devolvió un resultado válido")
        
        if 'sentiment' not in result:
            print(f"❌ [DEBUG] Error: resultado no tiene 'sentiment'. Keys: {result.keys() if result else 'None'}")
            raise Exception("El modelo no devolvió un sentimiento válido")
        
        # Marcar que se usó red neuronal (NO diccionario)
        result['method'] = 'neural_network'
        print(f"✅ [DEBUG] Análisis completado: sentiment={result.get('sentiment')}, score={result.get('score')}")
        return result
        
    except ValueError as e:
        # Errores de validación del modelo
        error_msg = str(e)
        print(f"❌ [DEBUG] ValueError en analyze_sentiment: {error_msg}")
        import traceback
        traceback.print_exc()
        if "no está entrenado" in error_msg.lower() or "no está inicializado" in error_msg.lower():
            raise Exception(
                "El modelo de red neuronal se está cargando o entrenando. "
                "Esto toma 15-30 segundos la primera vez. Por favor, espera unos momentos e intenta de nuevo."
            )
        raise Exception(f"Error en el modelo de red neuronal: {error_msg}")
    except ImportError as e:
        print(f"❌ [DEBUG] ImportError: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(
            "Error: No se pudo importar TensorFlow. "
            "Asegúrate de que TensorFlow esté instalado correctamente. "
            f"Detalle: {str(e)}"
        )
    except Exception as e:
        error_msg = str(e)
        print(f"❌ [DEBUG] Exception en analyze_sentiment: {error_msg}")
        import traceback
        traceback.print_exc()
        # Mejorar mensajes de error
        if "cargando" in error_msg.lower() or "entrenando" in error_msg.lower():
            raise Exception(error_msg)
        raise Exception(f"Error al analizar con red neuronal: {error_msg}")

"""
Módulo de análisis de sentimientos usando SOLO red neuronal LSTM.

Este módulo proporciona análisis de sentimientos usando exclusivamente
una red neuronal LSTM que captura contexto y relaciones semánticas.

El método de diccionario ha sido completamente eliminado.
Todo el análisis ahora se realiza usando redes neuronales.
"""
from typing import Dict


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
        from app.ml_models.sentiment_nn import SentimentNeuralNetwork
        model = SentimentNeuralNetwork()
        model.load_model()
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
        raise Exception(
            f"Error al analizar sentimiento con red neuronal: {str(e)}. "
            "Asegúrate de que el modelo esté correctamente entrenado y disponible."
        )

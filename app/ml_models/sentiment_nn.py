import re
import numpy as np
from typing import Dict, List, Tuple
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentNeuralNetwork:
    def __init__(self, max_words=1000, max_len=50):
        # Versión ULTRA-LIGERA para Render (512 MB limit)
        # max_words: 3000 -> 1000 (menos palabras = menos memoria)
        # max_len: 100 -> 50 (secuencias más cortas = menos memoria)
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        
    def clean_text(self, text: str) -> str:
        """Limpieza de texto - Parte 1"""
        if not text:
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar menciones y hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Eliminar caracteres especiales excepto letras, números y espacios
        text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def prepare_data(self, texts: List[str], labels: List[str] = None) -> Tuple:
        """Preparar datos para entrenamiento o predicción"""
        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")
        
        # Limpiar textos
        cleaned_texts = [self.clean_text(text) if text else "" for text in texts]
        
        # Tokenizar
        if labels:
            # Si hay etiquetas, estamos entrenando, ajustar tokenizer
            self.tokenizer.fit_on_texts(cleaned_texts)
        elif not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
            # Si no hay tokenizer entrenado y no estamos entrenando, error
            raise ValueError("El tokenizer no está entrenado. Debe entrenar el modelo primero.")
        
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # Asegurar que todas las secuencias tengan al menos un elemento (OOV token)
        # Si una secuencia está vacía, agregar el token OOV (índice 1 generalmente)
        sequences = [seq if seq else [1] for seq in sequences]
        
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        if labels:
            encoded_labels = self.label_encoder.fit_transform(labels)
            return padded_sequences, encoded_labels
        
        return padded_sequences
    
    def build_model(self, vocab_size: int, num_classes: int):
        """Construir modelo ULTRA-LIGERO para Render (512 MB limit) - Mantiene funcionalidad"""
        # Modelo mínimo pero funcional: embedding pequeño + LSTM pequeña
        # Aunque es pequeño, sigue siendo una red neuronal LSTM completamente funcional
        model = Sequential([
            Embedding(vocab_size + 1, 32),  # Reducido de 64 a 32 para menos memoria
            LSTM(16, dropout=0.2),  # Reducido de 32 a 16 neuronas
            Dense(8, activation='relu'),  # Reducido de 16 a 8
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        # Usar optimizador estándar (Adam es eficiente en memoria)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, texts: List[str], labels: List[str], epochs=10, batch_size=32):
        """Entrenar modelo - Versión ULTRA-LIGERA para Render (512 MB limit)"""
        import tensorflow as tf
        import gc  # Para limpiar memoria
        
        print(f"📊 Preparando datos: {len(texts)} textos, {len(set(labels))} clases")
        X, y = self.prepare_data(texts, labels)
        
        # Limitar tamaño de datos si es muy grande (para ahorrar memoria)
        max_samples = 200  # Máximo 200 muestras para entrenamiento
        if len(X) > max_samples:
            print(f"⚠️ Reduciendo datos de {len(X)} a {max_samples} para ahorrar memoria...")
            X = X[:max_samples]
            y = y[:max_samples]
        
        # Si hay pocos datos, usar todos para entrenamiento (sin validación)
        if len(X) < 50:
            X_train, y_train = X, y
            X_val, y_val = X, y
            use_validation = False
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            use_validation = True
        
        vocab_size = len(self.tokenizer.word_index)
        num_classes = len(self.label_encoder.classes_)
        print(f"📊 Vocabulario: {vocab_size} palabras, Clases: {num_classes}")
        print(f"📊 Datos entrenamiento: {len(X_train)}, Validación: {len(X_val) if use_validation else 'N/A'}")
        
        # Limpiar memoria antes de construir modelo
        gc.collect()
        
        self.model = self.build_model(vocab_size, num_classes)
        
        print(f"🚀 Iniciando entrenamiento: {epochs} épocas, batch_size={batch_size}")
        
        # Entrenar con batch size pequeño para usar menos memoria
        fit_kwargs = {
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': 0,  # Sin logs para acelerar y ahorrar memoria
        }
        
        if use_validation:
            fit_kwargs['validation_data'] = (X_val, y_val)
            history = self.model.fit(X_train, y_train, **fit_kwargs)
            # Evaluar modelo
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            print(f"✅ Entrenamiento completado - Precisión validación: {val_accuracy:.2%}")
        else:
            history = self.model.fit(X_train, y_train, **fit_kwargs)
            print(f"✅ Entrenamiento completado (sin validación por datos limitados)")
        
        # Limpiar memoria después de entrenar
        gc.collect()
        
        self.is_trained = True
        return history
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Predecir sentimiento"""
        if not self.is_trained or not self.model:
            raise ValueError("El modelo no está entrenado")
        
        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")
        
        try:
            X = self.prepare_data(texts)
            
            # Verificar que tenemos datos válidos
            if X.shape[0] == 0:
                raise ValueError("No se pudieron preparar los datos para predicción")
            
            predictions = self.model.predict(X, verbose=0)
            
            # Manejar caso donde el modelo devuelve predicciones vacías
            if predictions is None or len(predictions) == 0:
                raise ValueError("El modelo no devolvió predicciones válidas")
            
            predicted_classes = np.argmax(predictions, axis=1)
            predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
            confidence = np.max(predictions, axis=1)
            
            results = []
            for i, text in enumerate(texts):
                if i >= len(predicted_labels):
                    # Si hay menos predicciones que textos, usar neutral por defecto
                    label = 'neutral'
                    score = 0.5
                else:
                    label = predicted_labels[i]
                    score = float(confidence[i])
                
                if label == 'positivo':
                    sentiment = 'positivo'
                    emoji = '🟢'
                    score_value = score
                elif label == 'negativo':
                    sentiment = 'negativo'
                    emoji = '🔴'
                    score_value = -score
                else:
                    sentiment = 'neutral'
                    emoji = '🟡'
                    score_value = 0.0
                
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'score': round(score_value, 3),
                    'emoji': emoji,
                    'confidence': round(score, 3)
                })
            
            return results
        except Exception as e:
            error_msg = f"Error en predicción: {str(e)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
    
    def predict_single(self, text: str) -> Dict:
        """Predecir sentimiento de un solo texto"""
        results = self.predict([text])
        return results[0]
    
    def load_model(self, model_path: str = 'app/ml_models/sentiment_model.h5'):
        """Cargar modelo pre-entrenado"""
        # En Render, el sistema de archivos es efímero, así que siempre creamos el modelo en memoria
        # pero solo lo entrenamos una vez por instancia de la aplicación
        
        # Asegurar que el directorio existe
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        # Intentar cargar modelo existente (puede no existir en Render)
        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
            try:
                print("🔄 Cargando modelo de red neuronal pre-entrenado...")
                # Intentar cargar con compile=True primero (más seguro)
                try:
                    self.model = load_model(model_path)
                    print("✅ Modelo cargado con compilación automática")
                except Exception as compile_error:
                    print(f"⚠️ Error al cargar con compilación automática: {compile_error}")
                    print("🔄 Intentando cargar sin compilación y recompilando manualmente...")
                    self.model = load_model(model_path, compile=False)
                    # Recompilar el modelo
                    self.model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    print("✅ Modelo recompilado correctamente")
                
                # Cargar tokenizer y label encoder
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                # Verificar que el modelo está correctamente cargado
                if self.model is None:
                    raise ValueError("El modelo no se cargó correctamente")
                if not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
                    raise ValueError("El tokenizer no se cargó correctamente")
                if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                    raise ValueError("El label encoder no se cargó correctamente")
                
                self.is_trained = True
                print("✅ Modelo de red neuronal cargado y verificado correctamente")
                return
            except Exception as e:
                print(f"⚠️ Error al cargar modelo pre-entrenado: {e}")
                import traceback
                traceback.print_exc()
                print("🔄 Se creará un nuevo modelo...")
                # Limpiar archivos corruptos si existen
                try:
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    if os.path.exists(tokenizer_path):
                        os.remove(tokenizer_path)
                    if os.path.exists(label_encoder_path):
                        os.remove(label_encoder_path)
                except:
                    pass
        
        # Si no existe o falló cargar, crear y entrenar modelo
        print("🔄 Creando y entrenando modelo de red neuronal (versión rápida, ~10-20 segundos)...")
        try:
            self._create_pretrained_model()
            print("✅ Modelo de red neuronal entrenado y guardado correctamente")
        except Exception as e:
            print(f"❌ Error al crear modelo de red neuronal: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_pretrained_model(self):
        """Crear modelo pre-entrenado - Versión ULTRA-LIGERA para Render (512 MB) pero FUNCIONAL"""
        # Datos de entrenamiento balanceados pero reducidos para ahorrar memoria
        # Mantenemos suficiente variedad para que el modelo aprenda correctamente
        positive_texts = [
            "excelente producto muy bueno", "me encanta este servicio", "muy satisfecho con la compra",
            "recomiendo totalmente", "calidad superior", "atención perfecta", "rápido y eficiente",
            "super contento", "vale la pena", "muy recomendado", "increíble experiencia", "servicio de primera",
            "muy buena calidad", "excelente atención", "producto genial", "muy bien hecho",
            "súper recomendable", "calidad excelente", "muy profesional", "servicio impecable",
            "excelente servicio al cliente", "muy buena experiencia", "producto de calidad",
            "muy satisfecho", "recomiendo este producto", "muy bueno", "excelente calidad",
            "muy rápido", "muy eficiente", "muy bien", "excelente", "genial", "perfecto"
        ] * 2  # Reducido de 5 a 2 para ahorrar memoria (pero mantiene variedad)
        
        negative_texts = [
            "pésimo servicio muy malo", "no recomiendo para nada", "calidad terrible",
            "muy decepcionado", "atención horrible", "lento e ineficiente", "no vale la pena",
            "muy insatisfecho", "problema grave", "no cumplió expectativas", "servicio pésimo",
            "muy mala calidad", "no funciona bien", "muy decepcionante", "producto defectuoso",
            "atención pésima", "muy caro para lo que es", "no lo recomiendo", "muy mal servicio",
            "problemas constantes", "muy malo", "terrible", "pésimo", "horrible", "decepcionante",
            "no funciona", "defectuoso", "mala calidad", "mal servicio", "no recomiendo",
            "insatisfecho", "problemas", "muy mal", "no vale", "terrible experiencia"
        ] * 2  # Reducido de 5 a 2
        
        neutral_texts = [
            "producto regular", "ni bueno ni malo", "aceptable", "normal", "sin comentarios",
            "básico", "estándar", "cumple su función", "nada especial", "producto común",
            "servicio estándar", "normal como cualquier otro", "ni destacable ni malo",
            "producto promedio", "servicio básico", "regular", "aceptable", "normal",
            "estándar", "básico", "promedio", "común", "sin destacar"
        ] * 2  # Reducido de 5 a 2
        
        texts = positive_texts + negative_texts + neutral_texts
        labels = (['positivo'] * len(positive_texts) + 
                 ['negativo'] * len(negative_texts) + 
                 ['neutral'] * len(neutral_texts))
        
        print("🔄 Entrenando modelo ULTRA-LIGERO (optimizado para 512 MB, pero completamente funcional)...")
        print(f"📊 Total de textos: {len(texts)}, Clases: {len(set(labels))}")
        # Entrenamiento con batch pequeño para usar menos memoria
        # Aunque es pequeño, el modelo sigue siendo una red neuronal LSTM funcional
        self.train(texts, labels, epochs=3, batch_size=16)  # 3 épocas, batch pequeño = menos memoria
        self.save_model()
        print("✅ Modelo entrenado y guardado correctamente (red neuronal LSTM funcional)")
    
    def save_model(self, model_path: str = 'app/ml_models/sentiment_model.h5'):
        """Guardar modelo"""
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model:
            self.model.save(model_path)
        
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)


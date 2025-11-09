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
    def __init__(self, max_words=800, max_len=35):
        # Red neuronal LSTM basada en texto - Soporta comentarios de hasta 25 palabras
        # max_words: 800 (vocabulario suficiente para comentarios)
        # max_len: 35 (soporta cómodamente hasta 25 palabras)
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
        print(f"🔍 [DEBUG] prepare_data() llamado con {len(texts)} texto(s), labels={labels is not None}")
        
        if not texts:
            print("❌ [DEBUG] Error: lista de textos vacía en prepare_data")
            raise ValueError("La lista de textos no puede estar vacía")
        
        # Limpiar textos
        print("🔍 [DEBUG] Limpiando textos...")
        cleaned_texts = [self.clean_text(text) if text else "" for text in texts]
        print(f"🔍 [DEBUG] Textos limpiados: {[t[:30] + '...' if len(t) > 30 else t for t in cleaned_texts[:3]]}")
        
        # Tokenizar
        if labels:
            # Si hay etiquetas, estamos entrenando, ajustar tokenizer
            print("🔍 [DEBUG] Entrenando tokenizer...")
            self.tokenizer.fit_on_texts(cleaned_texts)
            print(f"🔍 [DEBUG] Tokenizer entrenado: vocab_size={len(self.tokenizer.word_index)}")
        elif not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
            # Si no hay tokenizer entrenado y no estamos entrenando, error
            print("❌ [DEBUG] Error: tokenizer no está entrenado")
            raise ValueError("El tokenizer no está entrenado. Debe entrenar el modelo primero.")
        else:
            print(f"🔍 [DEBUG] Usando tokenizer existente: vocab_size={len(self.tokenizer.word_index)}")
        
        print("🔍 [DEBUG] Convirtiendo textos a secuencias...")
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        print(f"🔍 [DEBUG] Secuencias creadas: {[seq[:5] for seq in sequences[:3]]}")
        
        # Asegurar que todas las secuencias tengan al menos un elemento (OOV token)
        # Si una secuencia está vacía, agregar el token OOV (índice 1 generalmente)
        sequences = [seq if seq else [1] for seq in sequences]
        print(f"🔍 [DEBUG] Secuencias después de OOV: {[seq[:5] for seq in sequences[:3]]}")
        
        print(f"🔍 [DEBUG] Haciendo padding: maxlen={self.max_len}")
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        print(f"🔍 [DEBUG] Secuencias con padding: shape={padded_sequences.shape}")
        
        if labels:
            print("🔍 [DEBUG] Codificando etiquetas...")
            encoded_labels = self.label_encoder.fit_transform(labels)
            print(f"🔍 [DEBUG] Etiquetas codificadas: {encoded_labels[:5]}")
            return padded_sequences, encoded_labels
        
        return padded_sequences
    
    def build_model(self, vocab_size: int, num_classes: int):
        """Construir red neuronal LSTM basada en texto para comentarios de hasta 25 palabras"""
        # Red neuronal LSTM real - suficiente capacidad para aprender patrones de texto
        model = Sequential([
            Embedding(vocab_size + 1, 24),  # Embedding layer (vectores de palabras)
            LSTM(16, dropout=0.2),        # LSTM layer (16 neuronas - aprende patrones de texto)
            Dense(8, activation='relu'),   # Dense layer (red neuronal)
            Dropout(0.2),
            Dense(num_classes, activation='softmax')  # Salida (probabilidades: positivo/negativo/neutral)
        ])
        
        # Compilar modelo neuronal
        model.compile(
            optimizer='adam',  # Optimizador de red neuronal
            loss='sparse_categorical_crossentropy',  # Función de pérdida
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
        max_samples = 150  # Máximo 150 muestras para entrenamiento (suficiente para comentarios)
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
        """Predecir sentimiento usando red neuronal LSTM"""
        print(f"🔍 [DEBUG] predict() llamado con {len(texts)} texto(s)")
        
        # Validar que el modelo esté completamente entrenado y listo
        print(f"🔍 [DEBUG] Validando modelo: is_trained={self.is_trained}, model={self.model is not None}")
        if not self.is_trained:
            print("❌ [DEBUG] Error: modelo no está entrenado")
            raise ValueError(
                "El modelo de red neuronal no está entrenado. "
                "El modelo se está cargando o entrenando. Por favor, espera unos momentos."
            )
        
        if not self.model:
            print("❌ [DEBUG] Error: modelo no está inicializado")
            raise ValueError(
                "El modelo de red neuronal no está inicializado. "
                "El modelo se está cargando. Por favor, espera unos momentos."
            )
        
        # Validar que el tokenizer esté entrenado
        print(f"🔍 [DEBUG] Validando tokenizer: has word_index={hasattr(self.tokenizer, 'word_index')}")
        if not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
            print("❌ [DEBUG] Error: tokenizer no tiene word_index")
            raise ValueError(
                "El tokenizer del modelo no está entrenado. "
                "El modelo se está cargando. Por favor, espera unos momentos."
            )
        
        # Validar que el label encoder esté entrenado
        print(f"🔍 [DEBUG] Validando label_encoder: has classes={hasattr(self.label_encoder, 'classes_')}")
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            print("❌ [DEBUG] Error: label_encoder no tiene classes")
            raise ValueError(
                "El label encoder del modelo no está entrenado. "
                "El modelo se está cargando. Por favor, espera unos momentos."
            )
        
        if not texts:
            print("❌ [DEBUG] Error: lista de textos vacía")
            raise ValueError("La lista de textos no puede estar vacía")
        
        try:
            print(f"🔍 [DEBUG] Preparando datos para {len(texts)} texto(s)...")
            # Preparar datos para predicción
            X = self.prepare_data(texts)
            print(f"🔍 [DEBUG] Datos preparados: shape={X.shape}")
            
            # Verificar que tenemos datos válidos
            if X.shape[0] == 0:
                print("❌ [DEBUG] Error: X.shape[0] == 0")
                raise ValueError("No se pudieron preparar los datos para predicción")
            
            print("🔍 [DEBUG] Haciendo predicción con modelo neuronal...")
            # Hacer predicción con la red neuronal
            predictions = self.model.predict(X, verbose=0)
            print(f"🔍 [DEBUG] Predicciones recibidas: shape={predictions.shape if predictions is not None else None}")
            
            # Validar predicciones
            if predictions is None:
                print("❌ [DEBUG] Error: predictions es None")
                raise ValueError("El modelo no devolvió predicciones (None)")
            
            if len(predictions) == 0:
                print("❌ [DEBUG] Error: predictions está vacío")
                raise ValueError("El modelo no devolvió predicciones (vacío)")
            
            print(f"🔍 [DEBUG] Procesando predicciones...")
            # Procesar predicciones de la red neuronal
            predicted_classes = np.argmax(predictions, axis=1)
            print(f"🔍 [DEBUG] predicted_classes: {predicted_classes}")
            
            predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
            print(f"🔍 [DEBUG] predicted_labels: {predicted_labels}")
            
            confidence = np.max(predictions, axis=1)
            print(f"🔍 [DEBUG] confidence: {confidence}")
            
            results = []
            for i, text in enumerate(texts):
                if i >= len(predicted_labels):
                    print(f"⚠️ [DEBUG] Advertencia: índice {i} fuera de rango para predicciones")
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
            
            print(f"✅ [DEBUG] Predicción completada: {len(results)} resultado(s)")
            return results
            
        except ValueError as e:
            # Re-lanzar ValueError con mensaje claro
            error_msg = str(e)
            print(f"❌ [DEBUG] ValueError en predict: {error_msg}")
            import traceback
            traceback.print_exc()
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error en predicción de red neuronal: {str(e)}"
            print(f"❌ [DEBUG] Exception en predict: {error_msg}")
            import traceback
            traceback.print_exc()
            raise ValueError(error_msg)
    
    def predict_single(self, text: str) -> Dict:
        """Predecir sentimiento de un solo texto"""
        print(f"🔍 [DEBUG] predict_single() llamado con texto: '{text[:50]}...'")
        try:
            results = self.predict([text])
            if not results or len(results) == 0:
                print("❌ [DEBUG] Error: predict() no devolvió resultados")
                raise ValueError("No se obtuvieron resultados de la predicción")
            result = results[0]
            print(f"🔍 [DEBUG] predict_single() resultado: sentiment={result.get('sentiment')}, score={result.get('score')}")
            return result
        except Exception as e:
            print(f"❌ [DEBUG] Error en predict_single: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
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
                
                # Validación final: asegurar que el modelo puede hacer una predicción de prueba
                print("🔍 [DEBUG] Validando modelo con predicción de prueba...")
                try:
                    # Hacer una predicción de prueba para validar que el modelo funciona
                    test_text = "excelente"
                    print(f"🔍 [DEBUG] Texto de prueba: '{test_text}'")
                    test_X = self.prepare_data([test_text])
                    print(f"🔍 [DEBUG] Datos de prueba preparados: shape={test_X.shape}")
                    test_pred = self.model.predict(test_X, verbose=0)
                    print(f"🔍 [DEBUG] Predicción de prueba: {test_pred}")
                    if test_pred is None or len(test_pred) == 0:
                        raise ValueError("El modelo no puede hacer predicciones válidas")
                    print("✅ [DEBUG] Modelo validado correctamente con predicción de prueba")
                except Exception as e:
                    print(f"⚠️ [DEBUG] Error al validar modelo: {e}")
                    import traceback
                    traceback.print_exc()
                    # Si falla la validación, marcar como no entrenado
                    self.is_trained = False
                    raise ValueError(f"El modelo no está funcionando correctamente: {str(e)}")
                
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
        print("🔍 [DEBUG] Iniciando _create_pretrained_model()...")
        try:
            self._create_pretrained_model()
            print("✅ Modelo de red neuronal entrenado y guardado correctamente")
            
            # Validar que el modelo esté completamente listo después del entrenamiento
            print("🔍 [DEBUG] Validando modelo después del entrenamiento...")
            if not self.is_trained:
                raise ValueError("El modelo no se marcó como entrenado después de _create_pretrained_model()")
            if not self.model:
                raise ValueError("El modelo no se creó después de _create_pretrained_model()")
            if not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
                raise ValueError("El tokenizer no se entrenó correctamente")
            if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                raise ValueError("El label encoder no se entrenó correctamente")
            
            print("✅ [DEBUG] Modelo completamente validado después del entrenamiento")
        except Exception as e:
            print(f"❌ Error al crear modelo de red neuronal: {e}")
            import traceback
            traceback.print_exc()
            self.is_trained = False
            raise
    
    def _create_pretrained_model(self):
        """Entrenar red neuronal LSTM con comentarios de hasta 25 palabras"""
        print("🔍 [DEBUG] _create_pretrained_model() iniciado")
        # Datos de entrenamiento con comentarios completos (hasta 25 palabras)
        # Incluir frases cortas Y comentarios completos para mejor aprendizaje
        
        # Comentarios POSITIVOS (hasta 25 palabras)
        positive_texts = [
            # Frases cortas
            "excelente producto muy bueno", "me encanta este servicio", "muy satisfecho",
            "recomiendo totalmente", "calidad superior", "atención perfecta",
            "super contento", "vale la pena", "muy recomendado", "increíble experiencia",
            "excelente servicio", "muy buena calidad", "excelente atención", 
            "producto genial", "muy bien hecho", "súper recomendable",
            
            # Comentarios completos (10-25 palabras)
            "excelente servicio al cliente muy atento y profesional la atención fue rápida y eficiente",
            "me encantó este producto la calidad es superior y el precio es muy razonable lo recomiendo totalmente",
            "muy buena experiencia de compra el producto llegó rápido y en perfecto estado estoy muy satisfecho",
            "servicio impecable desde el primer contacto hasta la entrega todo fue perfecto muy recomendado",
            "calidad excelente el producto superó mis expectativas y el servicio fue muy profesional y amable",
            "increíble experiencia el producto es de muy buena calidad y la atención al cliente fue excepcional",
            "muy contento con la compra el servicio fue rápido y el producto es de excelente calidad",
            "recomiendo totalmente este producto la calidad es superior y el precio es muy justo",
        ]
        
        # Comentarios NEGATIVOS (hasta 25 palabras)
        negative_texts = [
            # Frases cortas
            "pésimo servicio muy malo", "no recomiendo para nada", "calidad terrible",
            "muy decepcionado", "atención horrible", "lento e ineficiente", "no vale la pena",
            "muy insatisfecho", "problema grave", "no cumplió expectativas", "servicio pésimo",
            "mal servicio", "muy mala calidad", "no funciona bien", "muy decepcionante",
            
            # Comentarios completos (10-25 palabras)
            "pésimo servicio al cliente muy lento y desatento la atención fue horrible y no resolvieron mi problema",
            "muy decepcionado con este producto la calidad es terrible y no funciona como se esperaba no lo recomiendo",
            "servicio muy malo el producto llegó tarde y en mal estado estoy muy insatisfecho con la compra",
            "no recomiendo para nada este producto tiene muchos defectos y el servicio al cliente es pésimo",
            "muy mala experiencia el producto no cumple con lo prometido y la atención fue horrible",
            "calidad terrible el producto se rompió al poco tiempo y el servicio no respondió a mis quejas",
            "problema grave con este producto no funciona correctamente y el servicio al cliente fue ineficiente",
            "muy insatisfecho con la compra el producto es de mala calidad y el servicio fue pésimo",
        ]
        
        # Comentarios NEUTRALES (hasta 25 palabras)
        neutral_texts = [
            # Frases cortas
            "producto regular", "ni bueno ni malo", "aceptable", "normal", "sin comentarios",
            "básico", "estándar", "cumple su función", "nada especial", "producto común",
            "servicio estándar", "normal como cualquier otro", "ni destacable ni malo",
            "producto promedio", "servicio básico",
            
            # Comentarios completos (10-25 palabras)
            "producto regular que cumple su función básica nada especial pero tampoco tiene problemas mayores",
            "servicio estándar normal como cualquier otro no destacó ni positivo ni negativo simplemente aceptable",
            "producto promedio que funciona como se espera sin nada que destacar pero tampoco con problemas",
            "experiencia normal el producto es básico y cumple su función sin sorpresas positivas ni negativas",
            "servicio básico que funciona correctamente sin problemas pero tampoco con características especiales",
            "producto común que cumple con lo mínimo esperado ni bueno ni malo simplemente aceptable",
        ]
        
        texts = positive_texts + negative_texts + neutral_texts
        labels = (['positivo'] * len(positive_texts) + 
                 ['negativo'] * len(negative_texts) + 
                 ['neutral'] * len(neutral_texts))
        
        print("🔄 Entrenando red neuronal LSTM para comentarios de hasta 25 palabras...")
        print(f"📊 Total de textos: {len(texts)}, Clases: {len(set(labels))}")
        print(f"🔍 [DEBUG] Textos positivos: {len(positive_texts)}, negativos: {len(negative_texts)}, neutrales: {len(neutral_texts)}")
        
        # Entrenamiento con más épocas para mejor aprendizaje de comentarios completos
        print("🔍 [DEBUG] Iniciando entrenamiento...")
        try:
            self.train(texts, labels, epochs=5, batch_size=12)  # 5 épocas para mejor aprendizaje
            print("✅ [DEBUG] Entrenamiento completado")
            
            # Validar que el modelo está entrenado
            if not self.is_trained:
                raise ValueError("El modelo no se marcó como entrenado después del entrenamiento")
            if not self.model:
                raise ValueError("El modelo no existe después del entrenamiento")
            
            print("🔍 [DEBUG] Guardando modelo...")
            self.save_model()
            print("✅ [DEBUG] Modelo guardado correctamente")
            
            # Validación final: hacer una predicción de prueba
            print("🔍 [DEBUG] Haciendo predicción de prueba después del entrenamiento...")
            test_result = self.predict_single("excelente servicio")
            print(f"🔍 [DEBUG] Predicción de prueba: {test_result}")
            
        except Exception as e:
            print(f"❌ [DEBUG] Error en _create_pretrained_model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.is_trained = False
            raise
        
        print("✅ Red neuronal LSTM entrenada correctamente (soporta comentarios de hasta 25 palabras)")
    
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


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
from tensorflow.keras.callbacks import Callback
import time

class TrainingProgressCallback(Callback):
    """Callback para monitorear el progreso del entrenamiento"""
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        import sys
        self.stdout = sys.stdout
    
    def _print_and_flush(self, message):
        """Imprimir y hacer flush inmediatamente"""
        print(message)
        self.stdout.flush()
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self._print_and_flush(f"⏱️ [DEBUG] Entrenamiento comenzó a las {time.strftime('%H:%M:%S')}")
        self._print_and_flush(f"🔍 [DEBUG] Callback on_train_begin ejecutado correctamente")
    
    def on_epoch_begin(self, epoch, logs=None):
        epoch_start = time.time()
        self._print_and_flush(f"🔄 [DEBUG] Época {epoch + 1} comenzando a las {time.strftime('%H:%M:%S')}...")
        self.current_epoch_start = epoch_start
    
    def on_batch_begin(self, batch, logs=None):
        # Log cada 5 batches para no saturar, pero ver progreso
        if batch % 5 == 0 or batch == 0:
            self._print_and_flush(f"🔍 [DEBUG] Batch {batch} comenzando...")
    
    def on_batch_end(self, batch, logs=None):
        # Log cada 5 batches para no saturar, pero ver progreso
        if batch % 5 == 0 or batch == 0:
            loss = logs.get('loss', 'N/A')
            acc = logs.get('accuracy', 'N/A')
            self._print_and_flush(f"🔍 [DEBUG] Batch {batch} completado - loss: {loss}, accuracy: {acc}")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.current_epoch_start
        self.epoch_times.append(epoch_time)
        loss = logs.get('loss', 'N/A')
        accuracy = logs.get('accuracy', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')
        val_accuracy = logs.get('val_accuracy', 'N/A')
        self._print_and_flush(f"✅ [DEBUG] Época {epoch + 1} completada en {epoch_time:.2f}s - loss: {loss:.4f}, accuracy: {accuracy:.4f}, val_loss: {val_loss}, val_accuracy: {val_accuracy}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        self._print_and_flush(f"⏱️ [DEBUG] Entrenamiento terminado en {total_time:.2f}s total")
        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            self._print_and_flush(f"📊 [DEBUG] Tiempo promedio por época: {avg_time:.2f}s")
        else:
            self._print_and_flush(f"⚠️ [DEBUG] No se registraron épocas completadas")

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
        """Limpieza de texto mejorada con normalización"""
        if not text:
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Normalizar tildes y caracteres especiales (esto ayuda a que "atención" y "atencion" se traten igual)
        # Reemplazar tildes por versiones sin tilde para normalizar
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ñ': 'n', 'ü': 'u'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar menciones y hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Eliminar caracteres especiales excepto letras, números y espacios
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
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
            # Mostrar distribución completa de etiquetas codificadas
            unique_encoded, counts_encoded = np.unique(encoded_labels, return_counts=True)
            label_names_encoded = self.label_encoder.inverse_transform(unique_encoded)
            print(f"🔍 [DEBUG] Distribución de etiquetas codificadas:")
            for label_name, label_code, count in zip(label_names_encoded, unique_encoded, counts_encoded):
                print(f"   - {label_name} (código {label_code}): {count} muestras")
            print(f"🔍 [DEBUG] Primeras 10 etiquetas codificadas: {encoded_labels[:10]}")
            return padded_sequences, encoded_labels
        
        return padded_sequences
    
    def build_model(self, vocab_size: int, num_classes: int):
        """Construir red neuronal LSTM basada en texto para comentarios de hasta 25 palabras"""
        print(f"🔍 [DEBUG] Construyendo modelo: vocab_size={vocab_size}, num_classes={num_classes}")
        print(f"🔍 [DEBUG] Parámetros del modelo: max_words={self.max_words}, max_len={self.max_len}")
        
        # Red neuronal LSTM optimizada para entrenamiento RÁPIDO pero efectivo
        # Modelo balanceado: suficientemente grande para aprender, pero pequeño para entrenar rápido
        from tensorflow.keras.initializers import GlorotUniform
        
        model = Sequential([
            Embedding(vocab_size + 1, 8, embeddings_initializer=GlorotUniform()),  # 8 dimensiones (balance entre velocidad y capacidad)
            LSTM(4, dropout=0.1, kernel_initializer=GlorotUniform()),        # 4 unidades (rápido pero efectivo)
            Dense(4, activation='relu', kernel_initializer=GlorotUniform()),   # 4 unidades
            Dense(num_classes, activation='softmax', kernel_initializer=GlorotUniform())  # Salida
        ])
        
        print(f"🔍 [DEBUG] Modelo construido, compilando...")
        # Compilar modelo neuronal con learning rate optimizado
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=0.005)  # Learning rate balanceado (0.005) para aprender bien sin overshooting
        model.compile(
            optimizer=optimizer,  # Optimizador con learning rate configurado
            loss='sparse_categorical_crossentropy',  # Función de pérdida
            metrics=['accuracy'],
            run_eagerly=True  # Ejecutar en modo eager para evitar bloqueos durante compilación
        )
        
        # NO contar parámetros aquí - el modelo aún no está "built"
        # Los parámetros se contarán después del primer fit() cuando el modelo se construya automáticamente
        print(f"🔍 [DEBUG] Modelo compilado correctamente (run_eagerly=True)")
        
        return model
    
    def train(self, texts: List[str], labels: List[str], epochs=10, batch_size=32):
        """Entrenar modelo - Versión ULTRA-LIGERA para Render (512 MB limit)"""
        import tensorflow as tf
        import gc  # Para limpiar memoria
        
        print(f"📊 Preparando datos: {len(texts)} textos, {len(set(labels))} clases")
        X, y = self.prepare_data(texts, labels)
        
        # Mostrar distribución de etiquetas ANTES de reducir
        unique_labels, counts = np.unique(y, return_counts=True)
        label_names = self.label_encoder.inverse_transform(unique_labels)
        print(f"🔍 [DEBUG] Distribución de etiquetas ANTES de reducir:")
        for label_name, count in zip(label_names, counts):
            print(f"   - {label_name}: {count} muestras")
        
        # NO reducir datos - usar TODOS para mejor aprendizaje
        # Con modelo más pequeño y menos épocas, podemos usar más datos sin tardar mucho
        max_samples = 1000  # Usar todos los datos disponibles (no reducir)
        if len(X) > max_samples:
            print(f"⚠️ Reduciendo datos de {len(X)} a {max_samples} para ahorrar memoria y velocidad...")
            
            # CRÍTICO: Mezclar datos ANTES de reducir para mantener balance de clases
            # Esto asegura que no tomemos solo los primeros elementos que pueden ser de la misma clase
            indices = np.arange(len(X))
            np.random.seed(42)  # Semilla fija para reproducibilidad
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Intentar mantener balance de clases al reducir
            # Asegurar que haya al menos algunas muestras de cada clase
            unique_labels_all = np.unique(y_shuffled)
            num_classes_available = len(unique_labels_all)
            samples_per_class = max_samples // num_classes_available
            min_samples_per_class = max(1, samples_per_class - 1)  # Al menos 1 por clase
            
            print(f"🔍 [DEBUG] Intentando balancear: {min_samples_per_class} muestras mínimas por clase de {num_classes_available} clases")
            
            # Recopilar muestras balanceadas
            X_balanced = []
            y_balanced = []
            samples_taken_per_class = {int(label): 0 for label in unique_labels_all}
            used_indices = set()
            
            # Primero, tomar al menos min_samples_per_class de cada clase
            for label in unique_labels_all:
                label_int = int(label)
                label_indices = np.where(y_shuffled == label)[0]
                np.random.shuffle(label_indices)
                
                for idx in label_indices[:min_samples_per_class]:
                    if len(X_balanced) >= max_samples:
                        break
                    if idx not in used_indices:
                        X_balanced.append(X_shuffled[idx])
                        y_balanced.append(y_shuffled[idx])
                        used_indices.add(idx)
                        samples_taken_per_class[label_int] += 1
                
                if len(X_balanced) >= max_samples:
                    break
            
            # Si aún hay espacio, tomar muestras adicionales de manera aleatoria
            remaining_indices = [i for i in range(len(X_shuffled)) if i not in used_indices]
            np.random.shuffle(remaining_indices)
            
            for idx in remaining_indices:
                if len(X_balanced) >= max_samples:
                    break
                X_balanced.append(X_shuffled[idx])
                y_balanced.append(y_shuffled[idx])
                used_indices.add(idx)
            
            X = np.array(X_balanced)
            y = np.array(y_balanced)
            
            # Validar balance de clases DESPUÉS de reducir
            unique_labels_reduced, counts_reduced = np.unique(y, return_counts=True)
            label_names_reduced = self.label_encoder.inverse_transform(unique_labels_reduced)
            print(f"🔍 [DEBUG] Distribución de etiquetas DESPUÉS de reducir (balanceado):")
            for label_name, count in zip(label_names_reduced, counts_reduced):
                print(f"   - {label_name}: {count} muestras")
        
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
        
        # Validar balance de clases en datos de entrenamiento
        unique_labels_train, counts_train = np.unique(y_train, return_counts=True)
        label_names_train = self.label_encoder.inverse_transform(unique_labels_train)
        print(f"🔍 [DEBUG] Distribución de etiquetas en datos de ENTRENAMIENTO:")
        for label_name, count in zip(label_names_train, counts_train):
            print(f"   - {label_name}: {count} muestras")
        
        # Verificar que haya al menos una muestra de cada clase en entrenamiento
        if len(unique_labels_train) < num_classes:
            print(f"⚠️ [DEBUG] ADVERTENCIA: Solo hay {len(unique_labels_train)} clases en entrenamiento, esperadas {num_classes}")
            print(f"⚠️ [DEBUG] Esto puede afectar la precisión del modelo")
        else:
            print(f"✅ [DEBUG] Todas las clases ({num_classes}) están representadas en los datos de entrenamiento")
        
        # Limpiar memoria antes de construir modelo
        print("🔍 [DEBUG] Limpiando memoria antes de construir modelo...")
        gc.collect()
        
        print("🔍 [DEBUG] Construyendo modelo...")
        build_start = time.time()
        self.model = self.build_model(vocab_size, num_classes)
        build_time = time.time() - build_start
        print(f"✅ [DEBUG] Modelo construido en {build_time:.2f}s")
        
        # Optimizar épocas: suficiente para aprender, pero rápido
        actual_epochs = min(epochs, 2)  # Solo 2 épocas para entrenamiento rápido (con más datos y mejor LR)
        # Batch size debe ser menor o igual al número de muestras
        # Si hay 15 muestras, usar batch_size=15 (entrenar todas a la vez es más rápido)
        actual_batch_size = min(batch_size, len(X_train))  # No puede ser mayor que las muestras disponibles
        if actual_batch_size > len(X_train):
            actual_batch_size = len(X_train)  # Usar todas las muestras en un solo batch
        print(f"🔍 [DEBUG] Batch size ajustado: {actual_batch_size} (muestras disponibles: {len(X_train)})")
        
        print(f"🚀 Iniciando entrenamiento: {actual_epochs} épocas (reducido de {epochs}), batch_size={actual_batch_size} (ajustado de {batch_size})")
        print(f"📊 Datos de entrenamiento: {len(X_train)} muestras")
        print(f"📊 Shape de X_train: {X_train.shape}, Shape de y_train: {y_train.shape}")
        
        # Crear callback de progreso
        progress_callback = TrainingProgressCallback()
        
        # Entrenar con batch size más grande para más velocidad
        # run_eagerly ya está configurado en compile() para evitar bloqueos
        fit_kwargs = {
            'epochs': actual_epochs,
            'batch_size': actual_batch_size,
            'verbose': 0,  # Sin logs de TensorFlow (usamos nuestro callback)
            'callbacks': [progress_callback]  # Agregar callback de progreso
        }
        
        try:
            if use_validation:
                fit_kwargs['validation_data'] = (X_val, y_val)
                print("🔍 [DEBUG] Llamando a model.fit() con validación...")
                print(f"🔍 [DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"🔍 [DEBUG] X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
                print(f"🔍 [DEBUG] Callback creado: {progress_callback}")
                print(f"🔍 [DEBUG] fit_kwargs: {fit_kwargs}")
                
                # Flush stdout para asegurar que los logs se muestren
                import sys
                sys.stdout.flush()
                
                print("🚀 [DEBUG] INICIANDO model.fit() CON VALIDACIÓN AHORA...")
                sys.stdout.flush()
                
                fit_start = time.time()
                try:
                    history = self.model.fit(X_train, y_train, **fit_kwargs)
                    fit_time = time.time() - fit_start
                    print(f"✅ [DEBUG] model.fit() completado en {fit_time:.2f}s")
                except Exception as fit_error:
                    fit_time = time.time() - fit_start
                    print(f"❌ [DEBUG] ERROR en model.fit() después de {fit_time:.2f}s: {str(fit_error)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Ahora sí podemos contar los parámetros (el modelo ya está "built" después del fit)
                try:
                    total_params = self.model.count_params()
                    print(f"📊 [DEBUG] Modelo entrenado - Total de parámetros: {total_params:,}")
                except Exception as e:
                    print(f"⚠️ [DEBUG] No se pudo contar parámetros: {e}")
                
                # Evaluar modelo
                print("🔍 [DEBUG] Evaluando modelo...")
                eval_start = time.time()
                val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
                eval_time = time.time() - eval_start
                print(f"✅ [DEBUG] Evaluación completada en {eval_time:.2f}s")
                print(f"✅ Entrenamiento completado - Precisión validación: {val_accuracy:.2%}")
            else:
                print("🔍 [DEBUG] Llamando a model.fit() sin validación...")
                print(f"🔍 [DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"🔍 [DEBUG] Callback creado: {progress_callback}")
                print(f"🔍 [DEBUG] fit_kwargs: {fit_kwargs}")
                print(f"🔍 [DEBUG] Modelo antes de fit: {self.model}")
                print(f"🔍 [DEBUG] Verificando que el modelo esté compilado...")
                print(f"🔍 [DEBUG] Optimizer: {self.model.optimizer}")
                
                # Flush stdout para asegurar que los logs se muestren
                import sys
                sys.stdout.flush()
                
                print("🚀 [DEBUG] INICIANDO model.fit() AHORA...")
                print(f"🔍 [DEBUG] Parámetros: epochs={actual_epochs}, batch_size={actual_batch_size}, samples={len(X_train)}")
                sys.stdout.flush()
                
                fit_start = time.time()
                try:
                    # Agregar logging periódico durante el entrenamiento
                    print(f"🔍 [DEBUG] Llamando a model.fit() - esto puede tomar 10-30 segundos...")
                    sys.stdout.flush()
                    
                    history = self.model.fit(X_train, y_train, **fit_kwargs)
                    fit_time = time.time() - fit_start
                    print(f"✅ [DEBUG] model.fit() completado en {fit_time:.2f}s")
                    sys.stdout.flush()
                except Exception as fit_error:
                    fit_time = time.time() - fit_start
                    print(f"❌ [DEBUG] ERROR en model.fit() después de {fit_time:.2f}s: {str(fit_error)}")
                    print(f"🔍 [DEBUG] Tipo de error: {type(fit_error).__name__}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise
                
                # Ahora sí podemos contar los parámetros (el modelo ya está "built" después del fit)
                try:
                    total_params = self.model.count_params()
                    print(f"📊 [DEBUG] Modelo entrenado - Total de parámetros: {total_params:,}")
                except Exception as e:
                    print(f"⚠️ [DEBUG] No se pudo contar parámetros: {e}")
                
                print(f"✅ Entrenamiento completado (sin validación por datos limitados)")
        except Exception as e:
            print(f"❌ [DEBUG] ERROR durante model.fit(): {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Limpiar memoria después de entrenar
        print("🔍 [DEBUG] Limpiando memoria después de entrenar...")
        gc.collect()
        
        # Validar que el modelo esté correctamente entrenado
        print("🔍 [DEBUG] Validando modelo después del entrenamiento...")
        if self.model is None:
            raise ValueError("El modelo no existe después del entrenamiento")
        
        print("🔍 [DEBUG] Marcando modelo como entrenado...")
        self.is_trained = True
        print(f"✅ [DEBUG] Modelo marcado como entrenado: is_trained={self.is_trained}")
        print(f"🔍 [DEBUG] Modelo existe: {self.model is not None}")
        print(f"🔍 [DEBUG] Tokenizer tiene word_index: {hasattr(self.tokenizer, 'word_index') and len(self.tokenizer.word_index) > 0}")
        print(f"🔍 [DEBUG] Label encoder tiene classes: {hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0}")
        
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
            # Mostrar probabilidades completas para diagnóstico
            print(f"🔍 [DEBUG] Probabilidades completas (primeras 3 predicciones):")
            for i in range(min(3, len(predictions))):
                probs = predictions[i]
                label_names = self.label_encoder.classes_
                print(f"   Predicción {i}: {dict(zip(label_names, probs))}")
            
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
        
        # Datos de entrenamiento MEJORADOS con muchas más palabras clave y ejemplos
        # Incluir variaciones de palabras comunes en español
        
        # Comentarios POSITIVOS (palabras clave: excelente, bueno, buena, genial, etc.)
        positive_texts = [
            # Palabras clave simples
            "excelente", "bueno", "buena", "genial", "perfecto", "perfecta",
            "increíble", "maravilloso", "fantástico", "súper", "súper bien",
            # Frases comunes positivas
            "excelente producto", "buen producto", "muy buen producto", "producto excelente",
            "excelente servicio", "buen servicio", "muy buen servicio", "servicio excelente",
            "excelente atención", "buena atención", "muy buena atención", "atención excelente",
            "excelente calidad", "buena calidad", "muy buena calidad", "calidad excelente",
            # Frases completas positivas
            "excelente producto muy bueno", "me encanta este servicio", "muy satisfecho",
            "recomiendo totalmente", "calidad superior", "atención perfecta",
            "super contento", "vale la pena", "muy recomendado", "increíble experiencia",
            "producto genial", "muy bien hecho", "súper recomendable",
            "excelente servicio al cliente", "servicio excelente",
            "muy buena experiencia", "experiencia excelente", "experiencia positiva",
            "altamente recomendado", "muy recomendable", "totalmente recomendado",
            "muy contento", "satisfecho completamente", "me gustó mucho",
            "funciona perfecto", "cumple expectativas", "supera expectativas",
        ]
        
        # Comentarios NEGATIVOS (palabras clave: mal, malo, pésimo, insultos, etc.)
        negative_texts = [
            # Palabras clave simples negativas
            "mal", "malo", "mala", "pésimo", "pésima", "terrible", "horrible",
            "basura", "ruin", "decepcionante", "decepcionado",
            # Insultos y expresiones negativas comunes
            "esta cagada", "es una mierda", "una porquería", "es basura",
            "no sirve", "no funciona", "no vale", "no recomiendo",
            # Frases comunes negativas
            "pésimo servicio", "mal servicio", "muy mal servicio", "servicio pésimo",
            "pésimo producto", "mal producto", "muy mal producto", "producto pésimo",
            "pésima atención", "mal atención", "muy mal atención", "atención pésima",
            "pésima calidad", "mal calidad", "muy mal calidad", "calidad pésima",
            # Frases completas negativas
            "pésimo servicio muy malo", "no recomiendo para nada", "calidad terrible",
            "muy decepcionado", "atención horrible", "lento e ineficiente", "no vale la pena",
            "muy insatisfecho", "problema grave", "no cumplió expectativas", "servicio pésimo",
            "muy mala calidad", "no funciona bien", "muy decepcionante",
            "muy mal", "horrible experiencia", "pésima experiencia", "experiencia negativa",
            "no lo recomiendo", "no vale nada", "totalmente insatisfecho",
            "funciona mal", "no cumple expectativas", "muy por debajo de lo esperado",
        ]
        
        # Comentarios NEUTRALES (palabras clave: normal, regular, aceptable, etc.)
        neutral_texts = [
            # Palabras clave simples neutrales
            "normal", "regular", "aceptable", "básico", "estándar", "común",
            "ni bueno ni malo", "ni mal ni bien", "sin más", "nada especial",
            # Frases comunes neutrales
            "producto regular", "servicio regular", "atención regular", "calidad regular",
            "producto normal", "servicio normal", "atención normal", "calidad normal",
            "producto aceptable", "servicio aceptable", "atención aceptable", "calidad aceptable",
            # Frases completas neutrales
            "ni bueno ni malo", "aceptable", "sin comentarios",
            "básico", "estándar", "cumple su función", "nada especial", "producto común",
            "servicio estándar", "normal como cualquier otro", "ni destacable ni malo",
            "producto promedio", "servicio básico", "cumple con lo básico",
            "ni destacable ni malo", "regular nada más", "como se esperaba",
            "sin sorpresas", "ni bueno ni mal", "está bien",
        ]
        
        texts = positive_texts + negative_texts + neutral_texts
        labels = (['positivo'] * len(positive_texts) + 
                 ['negativo'] * len(negative_texts) + 
                 ['neutral'] * len(neutral_texts))
        
        print("🔄 Entrenando red neuronal LSTM para comentarios de hasta 25 palabras...")
        print(f"📊 Total de textos: {len(texts)}, Clases: {len(set(labels))}")
        print(f"🔍 [DEBUG] Textos positivos: {len(positive_texts)}, negativos: {len(negative_texts)}, neutrales: {len(neutral_texts)}")
        
        # Entrenamiento con más épocas para mejor aprendizaje
        print("🔍 [DEBUG] Iniciando entrenamiento...")
        try:
            # Entrenamiento rápido pero efectivo: 2 épocas con más datos
            history = self.train(texts, labels, epochs=2, batch_size=32)  # 2 épocas, batch más grande para velocidad
            print("✅ [DEBUG] Método train() completado")
            
            # Validar que el modelo está entrenado
            print(f"🔍 [DEBUG] Verificando estado del modelo después del entrenamiento...")
            print(f"🔍 [DEBUG] is_trained: {self.is_trained}")
            print(f"🔍 [DEBUG] model existe: {self.model is not None}")
            print(f"🔍 [DEBUG] tokenizer tiene word_index: {hasattr(self.tokenizer, 'word_index') and self.tokenizer.word_index is not None}")
            print(f"🔍 [DEBUG] label_encoder tiene classes: {hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0}")
            
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
            print(f"✅ [DEBUG] Predicción de prueba exitosa: sentiment={test_result.get('sentiment')}, score={test_result.get('score')}")
            
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


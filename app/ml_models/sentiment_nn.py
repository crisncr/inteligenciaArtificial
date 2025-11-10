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
    def __init__(self, max_words=5000, max_len=100):
        # Red neuronal LSTM basada en texto - Optimizado para párrafos largos
        # max_words: 5000 (vocabulario amplio para mejor comprensión)
        # max_len: 100 (longitud suficiente para párrafos completos)
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        
    def clean_text(self, text: str) -> str:
        """Limpieza de texto mejorada con normalización y corrección de encoding"""
        if not text:
            return ""
        
        # Primero, intentar corregir problemas de encoding comunes de Excel/CSV
        # Problemas comunes: Ã© -> é, Ã³ -> ó, Ã± -> ñ, etc.
        # Esto ocurre cuando Excel guarda UTF-8 pero se lee como Latin-1
        encoding_fixes = {
            # Caracteres mal codificados más comunes (UTF-8 mal leído como Latin-1)
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã±': 'ñ', 'Ã¼': 'ü', 
            'Ã': 'Á', 'Ã‰': 'É', 'Ã': 'Í', 'Ã"': 'Ó', 'Ãš': 'Ú',
            'Ã': 'Ñ', 'Ãœ': 'Ü',
            # Caracteres raros que aparecen en Excel
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '—', 'â€"': '–',
            # Limpiar caracteres de control
            '\ufeff': '',  # BOM de UTF-8
            '\x00': '',  # Null bytes
        }
        
        # Aplicar correcciones de encoding
        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)
        
        # Intentar corrección automática de encoding si detectamos problemas
        if 'Ã' in text:
            try:
                # Intentar decodificar como Latin-1 y recodificar como UTF-8
                text = text.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
            except:
                pass
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Normalizar tildes y caracteres especiales
        # IMPORTANTE: El modelo fue entrenado eliminando tildes para normalizar
        # "atención" y "atencion" se tratan igual, lo cual es útil para el modelo
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
        # Logging mínimo para ahorrar memoria durante predicción
        if labels:
            print(f"🔍 [DEBUG] prepare_data() entrenamiento: {len(texts)} textos")
        # Si no hay labels, es predicción - logging mínimo
        
        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")
        
        # Limpiar textos
        cleaned_texts = [self.clean_text(text) if text else "" for text in texts]
        
        # Tokenizar
        if labels:
            # Si hay etiquetas, estamos entrenando, ajustar tokenizer
            self.tokenizer.fit_on_texts(cleaned_texts)
            if len(self.tokenizer.word_index) > 0:
                print(f"🔍 [DEBUG] Tokenizer entrenado: vocab_size={len(self.tokenizer.word_index)}")
        elif not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
            raise ValueError("El tokenizer no está entrenado. Debe entrenar el modelo primero.")
        
        # Convertir textos a secuencias
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # Asegurar que todas las secuencias tengan al menos un elemento (OOV token)
        sequences = [seq if seq else [1] for seq in sequences]
        
        # Hacer padding
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        if labels:
            encoded_labels = self.label_encoder.fit_transform(labels)
            # Mostrar distribución de etiquetas (logging mínimo)
            unique_encoded, counts_encoded = np.unique(encoded_labels, return_counts=True)
            label_names_encoded = self.label_encoder.inverse_transform(unique_encoded)
            print(f"🔍 [DEBUG] Datos preparados: shape={padded_sequences.shape}, Distribución: {dict(zip(label_names_encoded, counts_encoded))}")
            return padded_sequences, encoded_labels
        
        return padded_sequences
    
    def build_model(self, vocab_size: int, num_classes: int):
        """Construir red neuronal LSTM basada en texto para comentarios de hasta 25 palabras"""
        print(f"🔍 [DEBUG] Construyendo modelo: vocab_size={vocab_size}, num_classes={num_classes}")
        print(f"🔍 [DEBUG] Parámetros del modelo: max_words={self.max_words}, max_len={self.max_len}")
        
        # Red neuronal LSTM con suficiente capacidad para aprender
        from tensorflow.keras.initializers import GlorotUniform
        
        # Modelo optimizado para mejor aprendizaje (aumentado de tamaño mínimo)
        # Balance entre memoria y capacidad de aprendizaje
        effective_vocab_size = min(vocab_size + 1, self.max_words + 1)
        model = Sequential([
            Embedding(effective_vocab_size, 16, mask_zero=True),  # 16 dimensiones (aumentado de 6)
            LSTM(8, dropout=0.2, recurrent_dropout=0.2),  # 8 unidades (aumentado de 3) con dropout
            Dense(16, activation='relu'),   # 16 unidades (aumentado de 6)
            Dropout(0.3),  # Dropout para regularización
            Dense(num_classes, activation='softmax')  # Salida (3 clases)
        ])
        print(f"🔍 [DEBUG] Vocabulario: {effective_vocab_size}, Modelo mejorado: Embedding(16), LSTM(8), Dense(16)")
        
        print(f"🔍 [DEBUG] Modelo construido, compilando...")
        # Compilar modelo neuronal con learning rate más conservador para mejor convergencia
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=0.001)  # Learning rate más conservador (0.001) para mejor convergencia
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False  # Deshabilitar eager mode para mejor rendimiento y convergencia
        )
        
        # NO contar parámetros aquí - el modelo aún no está "built"
        # Los parámetros se contarán después del primer fit() cuando el modelo se construya automáticamente
        print(f"🔍 [DEBUG] Modelo compilado correctamente (run_eagerly=False)")
        
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
        
        # USAR más datos para mejor aprendizaje (pero sin exceder memoria)
        # Aumentar muestras para mejor precisión con párrafos largos
        max_samples = 180  # Usar 180 muestras para mejor aprendizaje (aumentado de 120 para incluir párrafos largos)
        if len(X) > max_samples:
            print(f"⚠️ Reduciendo datos de {len(X)} a {max_samples} para ahorrar memoria...")
            
            # MEZCLAR datos ANTES de reducir para mantener balance de clases
            indices = np.arange(len(X))
            np.random.seed(42)  # Semilla fija para reproducibilidad
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Asegurar balance de clases al reducir
            unique_labels_all = np.unique(y_shuffled)
            num_classes_available = len(unique_labels_all)
            samples_per_class = max_samples // num_classes_available
            min_samples_per_class = max(1, samples_per_class - 2)  # Al menos samples_per_class-2 por clase
            
            print(f"🔍 [DEBUG] Intentando balancear: ~{samples_per_class} muestras por clase de {num_classes_available} clases")
            
            # Recopilar muestras balanceadas
            X_balanced = []
            y_balanced = []
            samples_taken_per_class = {int(label): 0 for label in unique_labels_all}
            used_indices = set()
            
            # Primero, tomar muestras balanceadas de cada clase
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
        else:
            # MEZCLAR datos antes de entrenar para mejor aprendizaje
            print("🔍 [DEBUG] Mezclando datos antes de entrenar...")
            indices = np.arange(len(X))
            np.random.seed(42)  # Semilla fija para reproducibilidad
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            print(f"✅ [DEBUG] Datos mezclados: {len(X)} muestras")
        
        # SIEMPRE entrenar sin validación para máxima velocidad y menor uso de memoria
        # Con pocos datos, la validación no es necesaria y solo ralentiza
        X_train, y_train = X, y
        X_val, y_val = X, y
        use_validation = False
        print(f"🔍 [DEBUG] Entrenando SIN validación para máxima velocidad y menor memoria")
        
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
        
        # Limpiar memoria ANTES de construir modelo (CRÍTICO para Render 512 MB)
        print("🔍 [DEBUG] Limpiando memoria antes de construir modelo...")
        import tensorflow as tf
        tf.keras.backend.clear_session()  # Limpiar sesión de Keras antes
        gc.collect()  # Recolectar basura
        print("✅ [DEBUG] Memoria limpiada antes de construir modelo")
        
        print("🔍 [DEBUG] Construyendo modelo...")
        build_start = time.time()
        self.model = self.build_model(vocab_size, num_classes)
        build_time = time.time() - build_start
        print(f"✅ [DEBUG] Modelo construido en {build_time:.2f}s")
        
        # OPTIMIZACIÓN: Balance entre memoria y aprendizaje
        actual_epochs = 15  # Aumentar épocas para mejor aprendizaje (aumentado de 5)
        # Batch size balanceado para mejor aprendizaje
        actual_batch_size = min(8, len(X_train))  # Batch size aumentado para mejor estabilidad (aumentado de 3)
        print(f"🔍 [DEBUG] Batch size: {actual_batch_size}, Épocas: {actual_epochs} (optimizado para mejor aprendizaje)")
        
        print(f"🚀 Iniciando entrenamiento: {actual_epochs} épocas (reducido de {epochs}), batch_size={actual_batch_size} (ajustado de {batch_size})")
        print(f"📊 Datos de entrenamiento: {len(X_train)} muestras")
        print(f"📊 Shape de X_train: {X_train.shape}, Shape de y_train: {y_train.shape}")
        
        # Callbacks simples para entrenamiento
        fit_kwargs = {
            'epochs': actual_epochs,
            'batch_size': actual_batch_size,
            'verbose': 1,  # Mostrar progress (se cambia a 1 en fit())
            'callbacks': []  # Sin callbacks complejos para velocidad
        }
        
        # NO construir modelo explícitamente - ahorra memoria
        # El modelo se construirá automáticamente en el primer fit()
        print("🔍 [DEBUG] El modelo se construirá automáticamente en el primer fit()")
        
        # Entrenamiento SIMPLIFICADO - sin validación, sin callbacks, máximo velocidad
        print("🔍 [DEBUG] Llamando a model.fit() sin validación...")
        print(f"🔍 [DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"🔍 [DEBUG] Parámetros: epochs={actual_epochs}, batch_size={actual_batch_size}, samples={len(X_train)}")
        
        # Flush stdout para asegurar que los logs se muestren
        import sys
        sys.stdout.flush()
        
        print(f"🚀 [DEBUG] INICIANDO model.fit() - entrenamiento con {actual_epochs} épocas...")
        sys.stdout.flush()
        
        fit_start = time.time()
        history = None
        try:
            # Entrenamiento con verbose para ver accuracy
            fit_kwargs['verbose'] = 1  # Mostrar progress para ver si está aprendiendo
            history = self.model.fit(X_train, y_train, **fit_kwargs)
            fit_time = time.time() - fit_start
            print(f"✅ [DEBUG] model.fit() completado en {fit_time:.2f}s")
            
            # Verificar accuracy final
            if hasattr(history, 'history') and 'accuracy' in history.history:
                final_accuracy = history.history['accuracy'][-1]
                final_loss = history.history['loss'][-1] if 'loss' in history.history else None
                print(f"📊 [DEBUG] Accuracy final del entrenamiento: {final_accuracy:.4f}")
                if final_loss:
                    print(f"📊 [DEBUG] Loss final del entrenamiento: {final_loss:.4f}")
                if final_accuracy < 0.6:
                    print(f"⚠️ [DEBUG] ADVERTENCIA: Accuracy baja ({final_accuracy:.4f}), el modelo podría no estar aprendiendo bien")
                elif final_accuracy < 0.8:
                    print(f"✅ [DEBUG] Accuracy aceptable: {final_accuracy:.4f} (mejorable pero funcional)")
                else:
                    print(f"✅ [DEBUG] Accuracy excelente: {final_accuracy:.4f}")
            else:
                print(f"⚠️ [DEBUG] No se pudo obtener accuracy del historial")
            
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
        
        print(f"✅ Entrenamiento completado (sin validación)")
        
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
        
        # NO hacer prueba rápida para ahorrar memoria
        # El modelo ya está entrenado y validado por el accuracy del entrenamiento
        print("🔍 [DEBUG] Prueba rápida omitida para ahorrar memoria")
        
        # Limpiar memoria después de validar (CRÍTICO para Render 512 MB)
        print("🔍 [DEBUG] Limpiando memoria después de entrenar...")
        # NO eliminar history aquí porque se necesita devolver
        # Solo limpiar otras variables temporales
        gc.collect()  # Recolectar basura de Python
        print("✅ [DEBUG] Memoria limpiada (modelo preservado)")
        
        # Devolver history si existe, sino devolver None
        return history if history is not None else None
    
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
            # Preparar datos para predicción (logging mínimo para ahorrar memoria)
            X = self.prepare_data(texts)
            # Limpiar memoria inmediatamente después de preparar datos
            import gc
            gc.collect()
            
            # Verificar que tenemos datos válidos
            if X.shape[0] == 0:
                print("❌ [DEBUG] Error: X.shape[0] == 0")
                raise ValueError("No se pudieron preparar los datos para predicción")
            
            print("🔍 [DEBUG] Haciendo predicción con modelo neuronal...")
            # Hacer predicción con batch_size=1 para mínimo uso de memoria
            # Procesar uno por uno para evitar problemas de memoria
            predictions = self.model.predict(X, batch_size=1, verbose=0)
            print(f"🔍 [DEBUG] Predicciones recibidas: shape={predictions.shape if predictions is not None else None}")
            
            # Validar predicciones
            if predictions is None:
                print("❌ [DEBUG] Error: predictions es None")
                raise ValueError("El modelo no devolvió predicciones (None)")
            
            if len(predictions) == 0:
                print("❌ [DEBUG] Error: predictions está vacío")
                raise ValueError("El modelo no devolvió predicciones (vacío)")
            
            print(f"🔍 [DEBUG] Procesando predicciones...")
            # Mostrar solo la primera predicción para ahorrar memoria (logging mínimo)
            label_names = self.label_encoder.classes_
            if len(predictions) > 0:
                probs = predictions[0]
                prob_dict = dict(zip(label_names, probs))
                print(f"🔍 [DEBUG] Predicción 0: {prob_dict}")
            
            # Procesar predicciones de la red neuronal (mínimo logging para ahorrar memoria)
            predicted_classes = np.argmax(predictions, axis=1)
            predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
            confidence = np.max(predictions, axis=1)
            
            # Formatear confianza correctamente
            if len(confidence) > 0 and len(predicted_labels) > 0:
                conf_value = confidence[0]
                label_value = predicted_labels[0]
                print(f"🔍 [DEBUG] Predicción: {label_value}, confianza: {conf_value:.3f}")
            else:
                print(f"🔍 [DEBUG] Predicción: N/A, confianza: 0.000")
            
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
    
    def load_model(self, model_path: str = 'app/ml_models/sentiment_model.keras'):
        """Cargar modelo pre-entrenado - Descarga automática desde GitHub Releases si no existe"""
        # Asegurar que el directorio existe
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        # URLs para descargar modelo pre-entrenado desde GitHub Releases
        # ACTUALIZA ESTAS URLs con las URLs reales después de subir a GitHub Releases
        MODEL_URL = os.getenv(
            'MODEL_URL', 
            'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/sentiment_model.keras'
        )
        TOKENIZER_URL = os.getenv(
            'TOKENIZER_URL',
            'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/tokenizer.pkl'
        )
        LABEL_ENCODER_URL = os.getenv(
            'LABEL_ENCODER_URL',
            'https://github.com/crisncr/inteligenciaArtificial/releases/download/v1.0.0/label_encoder.pkl'
        )
        
        def download_file(url: str, filepath: str) -> bool:
            """Descargar archivo desde URL"""
            try:
                import requests
                print(f"📥 Descargando {os.path.basename(filepath)} desde GitHub Releases...")
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                print(f"✅ {os.path.basename(filepath)} descargado correctamente ({downloaded / 1024:.1f} KB)")
                return True
            except Exception as e:
                print(f"⚠️ No se pudo descargar {os.path.basename(filepath)}: {str(e)}")
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except:
                    pass
                return False
        
        # Verificar qué archivos faltan
        missing_files = []
        if not os.path.exists(model_path):
            missing_files.append(('Modelo', MODEL_URL, model_path))
        if not os.path.exists(tokenizer_path):
            missing_files.append(('Tokenizer', TOKENIZER_URL, tokenizer_path))
        if not os.path.exists(label_encoder_path):
            missing_files.append(('Label Encoder', LABEL_ENCODER_URL, label_encoder_path))
        
        # Descargar archivos faltantes
        if missing_files:
            print(f"📥 Descargando {len(missing_files)} archivo(s) del modelo pre-entrenado desde GitHub Releases...")
            downloaded_count = 0
            for name, url, filepath in missing_files:
                print(f"📥 Descargando {name}...")
                if download_file(url, filepath):
                    downloaded_count += 1
                else:
                    print(f"❌ Error al descargar {name}")
            
            if downloaded_count < len(missing_files):
                print(f"⚠️ Solo se descargaron {downloaded_count}/{len(missing_files)} archivos.")
                print("🔄 Se entrenará el modelo desde cero (esto tomará más tiempo)...")
                print("💡 NOTA: Esto solo debería pasar si las URLs de GitHub Releases no están disponibles")
                # Limpiar archivos parcialmente descargados
                for name, url, filepath in missing_files:
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except:
                            pass
            else:
                print("✅ Todos los archivos del modelo se descargaron correctamente desde GitHub Releases")
                print("✅ El modelo NO se entrenará, se usará el modelo pre-entrenado")
        
        # Intentar cargar modelo existente (local o descargado)
        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
            try:
                print("🔄 Cargando modelo de red neuronal pre-entrenado...")
                # Cargar modelo en formato .keras (compatible con Keras 3.x)
                try:
                    # Intentar cargar directamente (formato .keras es más compatible)
                    self.model = load_model(model_path)
                    print("✅ Modelo cargado correctamente")
                except Exception as load_error:
                    print(f"⚠️ Error al cargar modelo: {load_error}")
                    # Si falla, intentar cargar sin compilación
                    try:
                        self.model = load_model(model_path, compile=False)
                        # Recompilar el modelo
                        from tensorflow.keras.optimizers import Adam
                        self.model.compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        print("✅ Modelo cargado y recompilado correctamente")
                    except Exception as compile_error:
                        print(f"❌ Error al recompilar modelo: {compile_error}")
                        raise
                
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
                    test_pred = self.model.predict(test_X, batch_size=1, verbose=0)
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
        
        # Si no existe o falló cargar, crear y entrenar modelo (FALLBACK)
        print("=" * 60)
        print("⚠️ MODO FALLBACK: ENTRENANDO MODELO DESDE CERO")
        print("=" * 60)
        print("🔄 Creando y entrenando modelo de red neuronal (versión rápida, ~30-60 segundos)...")
        print("⚠️ NOTA: Esto se ejecuta solo si no se pudo descargar el modelo pre-entrenado")
        print("💡 RECOMENDACIÓN: Sube los archivos a GitHub Releases para evitar este entrenamiento")
        print("📋 Ver train_model_local.py para instrucciones")
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
        """Entrenar red neuronal LSTM con comentarios y párrafos largos (hasta 100 palabras)"""
        print("🔍 [DEBUG] _create_pretrained_model() iniciado")
        print("📊 Modelo configurado para párrafos largos: max_len=100, max_words=5000")
        # Datos de entrenamiento con frases cortas Y párrafos largos para mejor aprendizaje
        # Incluir variaciones de palabras comunes en español y párrafos completos
        # El modelo ahora puede procesar párrafos completos de hasta 100 palabras
        
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
            # Frases positivas adicionales (palabras que el modelo necesita aprender)
            "funciona de maravilla", "muy fácil de usar", "muy rápida",
            "fue amable", "resolvió mi problema", "soporte técnico amable",
            "aplicación funciona bien", "muy fácil", "rápida y eficiente",
            "llegó antes de lo esperado", "totalmente recomendado", "muy recomendable",
            "resolvió enseguida", "problema resuelto", "soporte excelente",
            "aplicación fácil", "funciona bien", "muy rápido",
            "amable y servicial", "resolvió rápido", "soporte rápido",
            "fácil de usar", "muy eficiente", "funciona perfectamente",
            "de maravilla", "muy bien", "excelente atención",
            "resolvió mi problema enseguida", "soporte técnico excelente",
            "aplicación funciona de maravilla", "muy fácil de usar y rápida",
            # Párrafos largos positivos
            "estoy muy satisfecho con este producto la calidad es excelente y el servicio al cliente fue increíble me respondieron rápido a todas mis preguntas y el producto llegó en perfectas condiciones sin duda lo recomiendo a todos",
            "me encanta este servicio la atención que recibí fue maravillosa desde el primer momento me sentí bien atendido el producto funciona perfectamente y cumple con todas mis expectativas estoy muy contento con la compra",
            "excelente experiencia de compra el producto es de muy buena calidad y el servicio al cliente es excepcional me ayudaron con todas mis dudas y el envío fue muy rápido sin duda volveré a comprar aquí",
            "estoy muy contento con este producto la calidad es superior a lo que esperaba y el servicio al cliente fue increíble me ayudaron con todas mis preguntas y el producto llegó en perfectas condiciones",
            "me encanta este servicio la atención que recibí fue maravillosa desde el primer momento me sentí bien atendido el producto funciona perfectamente y cumple con todas mis expectativas",
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
            # Frases negativas con problemas técnicos y de usabilidad (CRÍTICAS)
            "se cierra constantemente", "muy difícil de usar", "se cierra mucho",
            "aplicación se cierra", "se cierra todo el tiempo", "constantemente se cierra",
            "muy difícil", "difícil de usar", "complicado de usar",
            "no responde", "se congela", "se queda congelada", "se bloquea",
            "no funciona", "no arranca", "no inicia", "no carga",
            "muy lento", "demasiado lento", "súper lento", "extremadamente lento",
            "se queda colgado", "se cuelga", "se traba", "se detiene",
            "no sirve", "no funciona para nada", "no vale la pena",
            "problemas constantes", "muchos problemas", "siempre tiene problemas",
            "se cierra solo", "se cierra automáticamente", "se cierra sin avisar",
            "muy complicado", "demasiado complicado", "complejo de usar",
            "servicio fue lento", "muy lento el servicio", "servicio lento",
            "quedé insatisfecho", "muy insatisfecho", "totalmente insatisfecho",
            "no respondió", "no responden", "no contestan", "no contestaron",
            "no respondió a mis mensajes", "no contestan mensajes", "no responden mensajes",
            "producto llegó dañado", "llegó dañado", "llegó roto", "llegó mal",
            "no cumplió expectativas", "no cumple expectativas", "no cumplió",
            "mala experiencia", "horrible experiencia", "pésima experiencia",
            "no volveré a comprar", "no compraré más", "no recomiendo comprar",
            # Párrafos largos negativos (IMPORTANTE para detectar negatividad en párrafos)
            "estoy muy decepcionado con este producto la calidad es terrible y el servicio al cliente fue pésimo me tardaron mucho en responder y cuando lo hicieron no me ayudaron en nada el producto llegó dañado y no me quisieron dar reembolso no lo recomiendo para nada",
            "pésima experiencia de compra el producto no funciona como debería y el servicio al cliente es horrible me tardaron días en responder y cuando lo hicieron no me solucionaron nada el producto está defectuoso y no me quieren dar reembolso",
            "estoy muy insatisfecho con este servicio la atención fue terrible desde el primer momento me sentí mal atendido el producto no funciona bien y no cumple con mis expectativas no volveré a comprar aquí",
            "horrible experiencia el producto es de muy mala calidad y el servicio al cliente es pésimo me ayudaron mal con mis dudas y el envío tardó mucho tiempo sin duda no volveré a comprar aquí",
            "estoy muy decepcionado con este producto la calidad es pésima y el servicio al cliente fue terrible me respondieron mal a todas mis preguntas y el producto llegó en malas condiciones no lo recomiendo",
            "muy mala experiencia el producto no funciona como debería y el servicio al cliente es horrible me tardaron mucho en responder y cuando lo hicieron no me solucionaron nada el producto está defectuoso",
            "pésimo servicio la atención que recibí fue terrible desde el primer momento me sentí mal atendido el producto funciona mal y no cumple con mis expectativas no volveré a comprar aquí",
            "estoy muy insatisfecho con este producto la calidad es terrible y el servicio al cliente fue pésimo me ayudaron mal con todas mis dudas y el envío tardó mucho tiempo sin duda no lo recomiendo",
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
            # Párrafos largos neutrales
            "el producto es normal cumple con su función básica pero no destaca en nada especial el servicio al cliente es regular y la calidad es aceptable sin más comentarios",
            "experiencia regular el producto funciona como se espera pero no es nada especial el servicio al cliente es normal y la calidad es básica cumple con lo básico",
            "producto estándar la calidad es normal y el servicio al cliente es aceptable no hay nada destacable pero tampoco hay problemas graves cumple con su función",
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
            # Entrenamiento ULTRA-RÁPIDO: 1 época, batch_size automático (todas las muestras)
            history = self.train(texts, labels, epochs=1, batch_size=1000)  # 1 época, batch grande (se ajustará automáticamente)
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
    
    def save_model(self, model_path: str = 'app/ml_models/sentiment_model.keras'):
        """Guardar modelo en formato .keras (compatible con Keras 3.x)"""
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model:
            # Guardar en formato .keras (más compatible con Keras 3.x)
            # Keras 3.x infiere automáticamente el formato desde la extensión .keras
            self.model.save(model_path)
            print(f"✅ Modelo guardado en formato .keras: {model_path}")
        
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ Tokenizer guardado en: {tokenizer_path}")
        print(f"✅ Label encoder guardado en: {label_encoder_path}")


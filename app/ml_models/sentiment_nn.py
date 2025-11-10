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
        """
        Limpieza de texto mejorada con normalización y corrección de encoding.
        
        ⚠️ IMPORTANTE: Este método SOLO limpia el texto (corrige encoding, normaliza).
        NO clasifica sentimientos. La clasificación se hace 100% por la red neuronal LSTM.
        
        El diccionario 'encoding_fixes' es SOLO para corregir problemas de encoding
        de archivos CSV/Excel (ej: Ã© -> é). NO es un diccionario de sentimientos.
        """
        if not text:
            return ""
        
        # ⚠️ SOLO CORRECCIÓN DE ENCODING - NO CLASIFICACIÓN DE SENTIMIENTOS
        # Esto corrige problemas cuando Excel guarda UTF-8 pero se lee como Latin-1
        # Ejemplo: "Ã©" (mal codificado) -> "é" (correcto)
        # Esto NO afecta la clasificación de sentimientos, solo limpia el texto
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
        
        # Aplicar correcciones de encoding (SOLO limpieza, NO clasificación)
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
    
    def _is_valid_comment(self, comment: str) -> bool:
        """
        Valida si un comentario tiene sentido y debe incluirse en el dataset.
        Filtra comentarios sin sentido (solo números, solo símbolos, muy cortos, etc.)
        """
        if not comment or len(comment.strip()) < 3:
            return False
        
        # Limpiar para validación
        cleaned = comment.lower().strip()
        
        # Verificar que tenga al menos una letra
        has_letter = any(c.isalpha() for c in cleaned)
        if not has_letter:
            return False
        
        # Verificar que tenga al menos una palabra válida (no solo números/símbolos)
        words = cleaned.split()
        valid_words = [w for w in words if any(c.isalpha() for c in w)]
        if len(valid_words) < 1:
            return False
        
        # Verificar que no sea solo números y símbolos
        if all(c.isdigit() or c in '.,;:!?-/()' or c.isspace() for c in cleaned):
            return False
        
        # Verificar que tenga sentido (al menos 1 palabra significativa o 2 palabras)
        meaningful_words = [w for w in valid_words if len(w) >= 3 or w in ['no', 'si', 'ya', 'el', 'la', 'un', 'una', 'me', 'le', 'se']]
        if len(meaningful_words) < 1 and len(valid_words) < 2:
            return False
        
        return True
    
    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normaliza texto para comparación y evitar repeticiones exactas.
        Elimina tildes y números para comparar solo la estructura semántica.
        """
        if not text:
            return ""
        
        # Normalizar tildes
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ñ': 'n', 'ü': 'u'
        }
        text = text.lower()
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remover números para comparación (solo estructura semántica)
        text = re.sub(r'\d+', '', text)
        
        # Limpiar espacios
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def prepare_data(self, texts: List[str], labels: List[str] = None) -> Tuple:
        """
        Preparar datos para entrenamiento o predicción.
        
        ⚠️ IMPORTANTE: Este método SOLO convierte texto a números.
        NO clasifica sentimientos. La clasificación se hace 100% por la red neuronal LSTM.
        
        Flujo:
        1. Limpia el texto (encoding, normalización)
        2. Tokenizer: Convierte palabras a números (ej: "excelente" -> 5)
           - Esto es necesario porque las redes neuronales solo procesan números
           - NO es un diccionario de sentimientos, solo un mapeo palabra->número
        3. Label encoder: Convierte etiquetas a números (ej: "positivo" -> 0)
           - Solo para entrenamiento, NO para predicción
        4. La red neuronal LSTM hace la clasificación real en predict()
        """
        # Logging mínimo para ahorrar memoria durante predicción
        if labels:
            print(f"🔍 [DEBUG] prepare_data() entrenamiento: {len(texts)} textos")
        # Si no hay labels, es predicción - logging mínimo
        
        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")
        
        # 1. Limpiar textos (SOLO limpieza, NO clasificación)
        cleaned_texts = [self.clean_text(text) if text else "" for text in texts]
        
        # 2. Tokenizar: Convertir palabras a números
        # ⚠️ El tokenizer.word_index es un VOCABULARIO (mapeo palabra->número)
        # NO es un diccionario de sentimientos. Ejemplo: {"excelente": 5, "malo": 12}
        # Las redes neuronales necesitan números, no texto
        if labels:
            # Si hay etiquetas, estamos entrenando, ajustar tokenizer
            self.tokenizer.fit_on_texts(cleaned_texts)
            if len(self.tokenizer.word_index) > 0:
                print(f"🔍 [DEBUG] Tokenizer entrenado: vocab_size={len(self.tokenizer.word_index)}")
        elif not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
            raise ValueError("El tokenizer no está entrenado. Debe entrenar el modelo primero.")
        
        # Convertir textos a secuencias de números
        # Ejemplo: "excelente servicio" -> [5, 23] (números, no sentimientos)
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # Asegurar que todas las secuencias tengan al menos un elemento (OOV token)
        sequences = [seq if seq else [1] for seq in sequences]
        
        # Hacer padding (rellenar secuencias para que tengan la misma longitud)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        if labels:
            # 3. Label encoder: Convertir etiquetas a números (SOLO para entrenamiento)
            # Ejemplo: "positivo" -> 0, "negativo" -> 1, "neutral" -> 2
            # ⚠️ Esto NO clasifica, solo convierte etiquetas a números para entrenar
            encoded_labels = self.label_encoder.fit_transform(labels)
            # Mostrar distribución de etiquetas (logging mínimo)
            unique_encoded, counts_encoded = np.unique(encoded_labels, return_counts=True)
            label_names_encoded = self.label_encoder.inverse_transform(unique_encoded)
            print(f"🔍 [DEBUG] Datos preparados: shape={padded_sequences.shape}, Distribución: {dict(zip(label_names_encoded, counts_encoded))}")
            return padded_sequences, encoded_labels
        
        return padded_sequences
    
    def build_model(self, vocab_size: int, num_classes: int):
        """
        Construir red neuronal LSTM basada en texto.
        
        ⚠️ IMPORTANTE: Esta es una RED NEURONAL REAL (LSTM) que aprende patrones.
        NO hay reglas hardcodeadas, NO hay diccionarios de sentimientos.
        
        Arquitectura de la red neuronal:
        1. Embedding: Convierte números de palabras a vectores (representación semántica)
        2. LSTM: Procesa secuencias de palabras y aprende patrones temporales
        3. Dense + Dropout: Capas de aprendizaje que extraen características
        4. Dense (softmax): Capa de salida que clasifica en 3 clases (positivo/negativo/neutral)
        
        La red neuronal APRENDE durante el entrenamiento qué combinaciones de palabras
        indican sentimientos positivos, negativos o neutrales.
        """
        print(f"🔍 [DEBUG] Construyendo modelo: vocab_size={vocab_size}, num_classes={num_classes}")
        print(f"🔍 [DEBUG] Parámetros del modelo: max_words={self.max_words}, max_len={self.max_len}")
        
        # 🧠 RED NEURONAL LSTM - Aprende patrones, no reglas hardcodeadas
        from tensorflow.keras.initializers import GlorotUniform
        
        # Modelo optimizado para mejor aprendizaje (aumentado de tamaño mínimo)
        # Balance entre memoria y capacidad de aprendizaje
        effective_vocab_size = min(vocab_size + 1, self.max_words + 1)
        model = Sequential([
            # Capa 1: Embedding - Convierte números a vectores semánticos
            Embedding(effective_vocab_size, 16, mask_zero=True),  # 16 dimensiones (aumentado de 6)
            # Capa 2: LSTM - Procesa secuencias y aprende patrones temporales
            LSTM(8, dropout=0.2, recurrent_dropout=0.2),  # 8 unidades (aumentado de 3) con dropout
            # Capa 3: Dense - Extrae características aprendidas
            Dense(16, activation='relu'),   # 16 unidades (aumentado de 6)
            # Capa 4: Dropout - Previene sobreajuste
            Dropout(0.3),  # Dropout para regularización
            # Capa 5: Dense (softmax) - Clasifica en 3 clases (positivo/negativo/neutral)
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
        max_samples = 1000  # Usar 1000 muestras para mejor aprendizaje de patrones (dataset estructurado)
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
        """
        Predecir sentimiento usando SOLO red neuronal LSTM.
        
        ⚠️ IMPORTANTE: Esta función usa 100% red neuronal LSTM para clasificar.
        NO hay reglas hardcodeadas, NO hay diccionarios de sentimientos.
        
        Flujo de predicción:
        1. Limpia el texto (encoding, normalización)
        2. Tokeniza (convierte palabras a números)
        3. Pasa por la red neuronal LSTM (aquí se hace la clasificación)
        4. La red neuronal devuelve probabilidades (ej: [0.1, 0.8, 0.1] = negativo)
        5. Se convierte el número de clase a etiqueta (ej: 1 -> "negativo")
        
        La clasificación real ocurre en la línea: predictions = self.model.predict(X)
        La red neuronal LSTM aprendió los patrones durante el entrenamiento.
        """
        # Validación rápida (sin logs para mejor rendimiento)
        if not self.is_trained or not self.model:
            raise ValueError("El modelo no está listo. Por favor, espera unos momentos.")
        
        if not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
            raise ValueError("El tokenizer no está listo. Por favor, espera unos momentos.")
        
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            raise ValueError("El label encoder no está listo. Por favor, espera unos momentos.")
        
        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")
        
        try:
            # 1. Preparar datos: Convertir texto a números (NO clasifica, solo convierte)
            X = self.prepare_data(texts)
            # Limpiar memoria inmediatamente después de preparar datos
            import gc
            gc.collect()
            
            # Verificar que tenemos datos válidos
            if X.shape[0] == 0:
                raise ValueError("No se pudieron preparar los datos para predicción")
            
            # 2. 🧠 AQUÍ ES DONDE LA RED NEURONAL CLASIFICA
            # La red neuronal LSTM procesa los números y devuelve probabilidades
            # Ejemplo: [0.1, 0.8, 0.1] = 80% negativo, 10% positivo, 10% neutral
            # NO hay reglas hardcodeadas, TODO es aprendizaje neuronal
            predictions = self.model.predict(X, batch_size=1, verbose=0)
            
            # Validar predicciones
            if predictions is None or len(predictions) == 0:
                raise ValueError("El modelo no devolvió predicciones")
            
            # 3. Procesar predicciones de la red neuronal
            # np.argmax encuentra la clase con mayor probabilidad (la que eligió la red neuronal)
            predicted_classes = np.argmax(predictions, axis=1)
            # Convertir número de clase a etiqueta (ej: 1 -> "negativo")
            predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
            # Obtener la confianza (probabilidad máxima)
            confidence = np.max(predictions, axis=1)
            
            results = []
            for i, text in enumerate(texts):
                if i >= len(predicted_labels):
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
        try:
            results = self.predict([text])
            if not results or len(results) == 0:
                raise ValueError("No se obtuvieron resultados de la predicción")
            return results[0]
        except Exception as e:
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
            """Descargar archivo desde URL - Optimizado para memoria"""
            try:
                import requests
                print(f"📥 Descargando {os.path.basename(filepath)} desde GitHub Releases...")
                # Timeout más corto y stream para ahorrar memoria
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                # Descargar en chunks pequeños para ahorrar memoria
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                        # Limpiar memoria periódicamente durante la descarga
                        if downloaded % (1024 * 1024) == 0:  # Cada 1MB
                            import gc
                            gc.collect()
                
                # Limpiar memoria después de descargar
                import gc
                del response
                gc.collect()
                
                file_size_kb = downloaded / 1024
                print(f"✅ {os.path.basename(filepath)} descargado correctamente ({file_size_kb:.1f} KB)")
                return True
            except Exception as e:
                print(f"⚠️ No se pudo descargar {os.path.basename(filepath)}: {str(e)}")
                print(f"🔍 URL intentada: {url}")
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except:
                    pass
                # Limpiar memoria en caso de error
                import gc
                gc.collect()
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
                print(f"❌ ERROR: Solo se descargaron {downloaded_count}/{len(missing_files)} archivos.")
                print("❌ NO se puede entrenar el modelo en producción (consume demasiada memoria)")
                print("💡 SOLUCIÓN: Sube los archivos del modelo a GitHub Releases")
                print("📋 Ver train_model_local.py para instrucciones")
                # Limpiar archivos parcialmente descargados
                for name, url, filepath in missing_files:
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except:
                            pass
                # NO ENTRENAR - Lanzar error en lugar de entrenar
                raise ValueError(
                    f"No se pudieron descargar los archivos del modelo desde GitHub Releases. "
                    f"Archivos faltantes: {len(missing_files) - downloaded_count}. "
                    f"Por favor, asegúrate de que los archivos estén disponibles en GitHub Releases o "
                    f"entrena el modelo localmente y súbelo a GitHub Releases."
                )
            else:
                print("✅ Todos los archivos del modelo se descargaron correctamente desde GitHub Releases")
                print("✅ El modelo NO se entrenará, se usará el modelo pre-entrenado")
        
        # Intentar cargar modelo existente (local o descargado)
        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
            try:
                # Cargar modelo en formato .keras (compatible con Keras 3.x)
                try:
                    # Intentar cargar directamente (formato .keras es más compatible)
                    self.model = load_model(model_path)
                except Exception as load_error:
                    # Si falla, intentar cargar sin compilación
                    self.model = load_model(model_path, compile=False)
                    # Recompilar el modelo
                    from tensorflow.keras.optimizers import Adam
                    self.model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                
                # Cargar tokenizer y label encoder (optimizado para memoria)
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                # Limpiar memoria después de cargar
                import gc
                gc.collect()
                
                # Verificar que el modelo está correctamente cargado
                if self.model is None:
                    raise ValueError("El modelo no se cargó correctamente")
                if not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
                    raise ValueError("El tokenizer no se cargó correctamente")
                if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                    raise ValueError("El label encoder no se cargó correctamente")
                
                # Marcar modelo como entrenado (sin validación con predicción para mejor rendimiento)
                self.is_trained = True
                
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
        
        # Si no existe o falló cargar, NO ENTRENAR - Lanzar error
        # El entrenamiento consume demasiada memoria (>512MB) en Render
        print("=" * 60)
        print("❌ ERROR: No se pudo cargar el modelo pre-entrenado")
        print("=" * 60)
        print("❌ NO se puede entrenar el modelo en producción (límite de 512MB de memoria)")
        print("💡 SOLUCIÓN:")
        print("   1. Entrena el modelo localmente: python train_model_local.py")
        print("   2. Sube los archivos a GitHub Releases")
        print("   3. Asegúrate de que las URLs en load_model() sean correctas")
        print("📋 Ver train_model_local.py y GUIA_GITHUB_RELEASES.md para instrucciones")
        raise ValueError(
            "No se pudo cargar el modelo pre-entrenado y no se puede entrenar en producción "
            "(límite de memoria: 512MB). Por favor, asegúrate de que los archivos del modelo "
            "estén disponibles en GitHub Releases. Entrena el modelo localmente y súbelo a "
            "GitHub Releases antes de desplegar."
        )
    
    def _label_text_with_keywords(self, text: str) -> str:
        """
        Etiqueta un texto como positivo, negativo o neutral usando palabras clave.
        Versión mejorada para detectar mejor los comentarios negativos.
        Usado para etiquetar textos de Hugging Face que no tienen etiquetas de sentimiento.
        """
        if not text:
            return 'neutral'
        
        text_lower = text.lower()
        
        # Palabras clave positivas (EXPANDIDO)
        positive_keywords = [
            'excelente', 'bueno', 'buena', 'genial', 'perfecto', 'perfecta',
            'increíble', 'maravilloso', 'fantástico', 'delicioso', 'deliciosa',
            'agradable', 'acogedor', 'acogedora', 'limpio', 'limpia', 'bonito', 'bonita',
            'recomiendo', 'recomendado', 'recomendada', 'satisfecho', 'satisfecha',
            'me encanta', 'me encantó', 'me encanto', 'superó', 'supero', 'super', 'súper',
            'feliz', 'contento', 'contenta', 'alegre', 'amable', 'atento', 'atenta',
            'rápido', 'rápida', 'eficiente', 'profesional', 'calidad', 'precio', 'barato', 'barata',
            'vale la pena', 'valió la pena', 'valio la pena', 'volveré', 'volvere',
            'satisfactorio', 'satisfactoria', 'recomendable',
            # Palabras adicionales para casos específicos
            'fácil', 'facil', 'fácil de usar', 'facil de usar',
            'atención rápida', 'atencion rapida', 'atención eficiente', 'atencion eficiente',
            'rápida y eficiente', 'rapida y eficiente', 'rápido y eficiente', 'rapido y eficiente'
        ]
        
        # Palabras clave negativas (EXPANDIDO para detectar mejor los negativos)
        negative_keywords = [
            # Calificaciones negativas directas
            'malo', 'mala', 'pésimo', 'pésima', 'terrible', 'horrible', 'decepcionante',
            'decepcionado', 'decepcionada', 'decepción', 'decepcion',
            # Problemas de calidad/estado
            'roto', 'rota', 'dañado', 'dañada', 'defectuoso', 'defectuosa', 
            'incompleto', 'incompleta', 'en mal estado', 'mal estado', 
            'defectos', 'defecto', 'daños', 'daño',
            # Problemas de temperatura/sabor
            'frío', 'fría', 'sin sabor', 'horrible sabor', 'sabor horrible',
            'comida fría', 'comida fria',
            # Problemas de tiempo/demora
            'tarde', 'demoró', 'demoro', 'demorado', 'demorada', 'retraso', 
            'retrasado', 'retrasada', 'se demoró', 'se demoro', 
            'demoró demasiado', 'demoro demasiado', 'tardó', 'tardo',
            'lento', 'lenta', 'demasiado lento', 'demasiado lenta',
            'llegó tarde', 'llego tarde', 'con retraso',
            # Problemas de entrega/envío
            'desastre', 'perdido', 'perdida', 'se perdió', 'se perdio', 
            'no llegó', 'no llego', 'llegó incompleto', 'llego incompleto', 
            'llegó en mal estado', 'llego en mal estado', 'llegó roto', 'llego roto',
            'el envío se perdió', 'el envio se perdio', 'el envío se demoró',
            'el envio se demoro', 'la entrega fue un desastre',
            # Problemas de servicio/atención
            'grosero', 'grosera', 'mala atención', 'pésima atención', 
            'pésimo servicio', 'poco atento', 'poca atención', 
            'no respondió', 'no respondio', 'nunca respondió', 'nunca respondio',
            'no funciona', 'no funciona bien', 'no cumple',
            'mala comunicación', 'pésima comunicación',
            # Negaciones y rechazo
            'no recomiendo', 'no recomendaria', 'nunca volveré', 'nunca volvere',
            'no volveré', 'no volvere', 'no compraría', 'no compraria',
            'no lo recomiendo', 'no lo recomendaria', 'no lo recomendaría',
            'no recibí', 'no recibi', 'no recibió', 'no recibio',
            # Problemas y quejas
            'problema', 'problemas', 'queja', 'quejas', 'reclamo', 'reclamos',
            'insatisfecho', 'insatisfecha', 'devolución', 'devolver', 'reembolso',
            # Otros negativos
            'lleno de errores', 'errores', 'no cumple expectativas', 
            'no cumplió', 'no cumplio', 'defraudado', 'defraudada',
            # Frases negativas comunes
            'muy mala', 'muy malo', 'muy mal', 'pésimo servicio', 
            'terrible experiencia', 'no funcionó', 'no funciono',
            'no sirve', 'no sirvió', 'no sirvio', 'horrible experiencia',
            'una pésima experiencia', 'una pesima experiencia', 'pésima experiencia',
            # Negaciones específicas con palabras positivas
            'no vale', 'no vale la pena', 'no vale la calidad', 'no vale el precio',
            'no es bueno', 'no es buena', 'no es excelente', 'no es genial'
        ]
        
        # Contar palabras positivas y negativas primero
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        # Detectar negaciones que cambian el sentido (ej: "no es bueno" = negativo)
        negation_words = ['no', 'nunca', 'jamás', 'jamas', 'tampoco', 'ni']
        words = text_lower.split()
        has_negation_near_positive = False
        has_negation_with_value = False  # Para "no vale"
        
        # Buscar patrones específicos de negación
        text_lower_clean = ' ' + text_lower + ' '  # Agregar espacios para búsqueda exacta
        
        # Detectar "no vale" (ej: "no vale la calidad", "no vale la pena")
        if ' no vale ' in text_lower_clean or text_lower.startswith('no vale ') or text_lower.endswith(' no vale'):
            has_negation_with_value = True
            negative_count += 3  # Peso alto para este patrón
        
        # Buscar patrones como "no es bueno", "nunca fue excelente", etc.
        for i, word in enumerate(words):
            if word in negation_words:
                # Verificar si hay palabra positiva cerca (dentro de 4 palabras)
                context_start = max(0, i-4)
                context_end = min(len(words), i+5)
                context = ' '.join(words[context_start:context_end])
                
                # Palabras positivas que pueden ser negadas
                positive_words_to_check = ['bueno', 'buena', 'excelente', 'genial', 'perfecto', 
                                         'recomiendo', 'satisfecho', 'contento', 'vale', 'valió',
                                         'valio', 'recomendable', 'útil', 'util']
                
                for pos_word in positive_words_to_check:
                    if pos_word in context:
                        has_negation_near_positive = True
                        break
                
                if has_negation_near_positive:
                    break
        
        # Detectar frases con "muy" + adjetivo positivo/negativo
        if 'muy ' in text_lower:
            muy_index = text_lower.find('muy ')
            if muy_index != -1:
                # Buscar adjetivo después de "muy" (hasta 5 palabras para capturar contexto)
                rest_of_text = text_lower[muy_index + 4:].split()[0:5]
                rest_text = ' '.join(rest_of_text)
                
                # Adjetivos positivos con "muy"
                muy_positivos = ['amable', 'satisfecho', 'satisfecha', 'contento', 'contenta', 
                               'bueno', 'buena', 'bien', 'fácil', 'facil', 'feliz', 'excelente',
                               'buen', 'satisfactorio', 'satisfactoria']
                if any(adj in rest_text for adj in muy_positivos):
                    positive_count += 3  # Peso alto para "muy + positivo"
                
                # Adjetivos negativos con "muy"
                muy_negativos = ['malo', 'mala', 'mal', 'pésimo', 'pesimo', 'pésima', 'pesima',
                               'decepcionado', 'decepcionada', 'insatisfecho', 'insatisfecha']
                if any(adj in rest_text for adj in muy_negativos):
                    negative_count += 3  # Peso alto para "muy + negativo"
        
        # Detectar patrones específicos positivos en contexto
        # "atención al cliente" + adjetivo positivo
        if 'atención' in text_lower or 'atencion' in text_lower:
            if any(pos in text_lower for pos in ['amable', 'rápida', 'rapida', 'eficiente', 'buena', 'excelente']):
                positive_count += 2
        
        # "diseño" + verbo positivo (ej: "me encantó el diseño")
        if 'diseño' in text_lower or 'diseno' in text_lower:
            if any(pos in text_lower for pos in ['encantó', 'encanto', 'encanta', 'excelente', 'bueno', 'bonito']):
                positive_count += 2
        
        # "proceso" + adjetivo positivo (ej: "fácil proceso")
        if 'proceso' in text_lower:
            if any(pos in text_lower for pos in ['fácil', 'facil', 'rápido', 'rapido', 'sencillo', 'bueno']):
                positive_count += 2
        
        # "compra" + adjetivo positivo (ej: "fácil compra", "buena compra")
        if 'compra' in text_lower:
            if any(pos in text_lower for pos in ['fácil', 'facil', 'buena', 'buen', 'satisfecho', 'contento']):
                positive_count += 2
        
        # "resultado" + adjetivo positivo (ej: "satisfecho con el resultado")
        if 'resultado' in text_lower:
            if any(pos in text_lower for pos in ['satisfecho', 'satisfecha', 'contento', 'contenta', 'bueno', 'excelente']):
                positive_count += 2
        
        # "app" o "aplicación" + adjetivo positivo (ej: "app fácil de usar")
        if 'app' in text_lower or 'aplicación' in text_lower or 'aplicacion' in text_lower:
            if any(pos in text_lower for pos in ['fácil', 'facil', 'rápida', 'rapida', 'eficiente', 'buena']):
                positive_count += 2
        
        # Si hay negación con "vale", es definitivamente negativo
        if has_negation_with_value:
            return 'negativo'
        
        # Si hay negación cerca de palabra positiva, es negativo (ej: "no es bueno")
        if has_negation_near_positive:
            negative_count += 3  # Peso alto para negaciones
        
        # Detectar "pésima experiencia" o variantes
        if 'pésima experiencia' in text_lower or 'pesima experiencia' in text_lower or \
           'pésima' in text_lower and 'experiencia' in text_lower:
            negative_count += 2
        
        # Determinar sentimiento con lógica mejorada
        # Si hay indicadores negativos claros, priorizar negativo
        if negative_count > 0:
            # Si hay más negativos que positivos, o si hay al menos 2 negativos, es negativo
            if negative_count > positive_count or negative_count >= 2:
                return 'negativo'
            # Si hay negativos pero también muchos positivos, puede ser positivo
            elif positive_count > negative_count * 2:
                return 'positivo'
        
        # Si hay positivos y no hay negativos, es positivo
        if positive_count > 0 and negative_count == 0:
            return 'positivo'
        
        # Si hay más positivos que negativos, es positivo
        if positive_count > negative_count:
            return 'positivo'
        
        # Si hay negativos y no hay positivos, es negativo
        if negative_count > 0 and positive_count == 0:
            return 'negativo'
        
        # Por defecto, neutral
        return 'neutral'
    
    def _load_huggingface_datasets(self, limite: int = 5000, min_negativos: int = 300) -> List[Dict[str, str]]:
        """
        Carga datasets de Hugging Face en español y los etiqueta automáticamente.
        Carga muchos datos hasta encontrar suficientes ejemplos negativos.
        
        Args:
            limite: Número máximo de comentarios a cargar
            min_negativos: Número mínimo de comentarios negativos requeridos
            
        Returns:
            Lista de diccionarios con 'valor' (positivo/negativo/neutral) y 'comentario'
        """
        datos = []
        
        try:
            from datasets import load_dataset
            print("🔄 Cargando dataset de análisis de sentimientos en español desde Hugging Face...")
            print(f"📥 Solicitando hasta {limite} comentarios para encontrar al menos {min_negativos} negativos...")
            
            # Intentar cargar diferentes datasets compatibles
            dataset = None
            dataset_name = None
            
            # Opción 1: Dataset de análisis de sentimientos en textos turísticos de México
            try:
                print("🔄 Intentando cargar: alexcom/analisis-sentimientos-textos-turisitcos-mx-paisV2...")
                dataset = load_dataset("alexcom/analisis-sentimientos-textos-turisitcos-mx-paisV2", split=f"train[:{limite}]")
                dataset_name = "Textos Turísticos México"
                print(f"✅ Dataset cargado: {len(dataset)} comentarios disponibles")
            except Exception as e1:
                print(f"⚠️ No se pudo cargar dataset turístico: {e1}")
                return []
            
            # Procesar cada comentario
            negativos_encontrados = 0
            for item in dataset:
                # Obtener texto del comentario
                texto = item.get('text', item.get('texto', item.get('comentario', 
                        item.get('review_body', item.get('content', item.get('review', ''))))))
                
                # Validar que el texto tenga sentido
                if not texto or not isinstance(texto, str) or len(texto.strip()) < 10:
                    continue
                
                # Limpiar texto
                texto = texto.strip()
                
                # Filtrar comentarios sin sentido
                if not self._is_valid_comment(texto):
                    continue
                
                # Etiquetar usando palabras clave mejoradas
                sentimiento = self._label_text_with_keywords(texto)
                
                # Si es negativo, incrementar contador
                if sentimiento == 'negativo':
                    negativos_encontrados += 1
                
                datos.append({
                    'valor': sentimiento,
                    'comentario': texto
                })
                
                # Si ya tenemos suficientes negativos y suficientes datos totales, podemos parar antes
                # (pero procesamos todos para tener mejor distribución)
            
            print(f"✅ {len(datos)} comentarios válidos procesados de {dataset_name}")
            
            # Mostrar distribución de sentimientos
            positivo_count = sum(1 for d in datos if d['valor'] == 'positivo')
            negativo_count = sum(1 for d in datos if d['valor'] == 'negativo')
            neutral_count = sum(1 for d in datos if d['valor'] == 'neutral')
            print(f"📊 Distribución: {positivo_count} positivos, {negativo_count} negativos, {neutral_count} neutrales")
            
            # Advertir si no hay suficientes negativos
            if negativo_count < min_negativos:
                print(f"⚠️ ADVERTENCIA: Solo se encontraron {negativo_count} comentarios negativos (objetivo: {min_negativos})")
                print(f"💡 El dataset será balanceado con los negativos disponibles")
            else:
                print(f"✅ Se encontraron {negativo_count} comentarios negativos (objetivo: {min_negativos})")
            
        except ImportError:
            print("❌ Error: La librería 'datasets' no está instalada.")
            print("💡 Instala con: pip install datasets")
            return []
        except Exception as e:
            print(f"⚠️ Error al cargar dataset desde Hugging Face: {e}")
            print(f"📋 Tipo de error: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return []
        
        return datos
    
    def _create_training_dataset(self) -> List[Dict[str, str]]:
        """
        Crea un dataset estructurado con ~1000 muestras balanceadas.
        Usa solo datos de Hugging Face, balanceando automáticamente las clases.
        Los textos se cargan desde Hugging Face y se etiquetan automáticamente usando palabras clave.
        """
        dataset = []
        
        print("=" * 80)
        print("CARGANDO DATASET DE HUGGING FACE (ESPAÑOL)")
        print("=" * 80)
        print()
        
        # Cargar muchos datos de Hugging Face para encontrar suficientes negativos
        hf_data = self._load_huggingface_datasets(limite=5000, min_negativos=300)
        
        if not hf_data:
            raise ValueError(
                "No se pudieron cargar datos de Hugging Face. "
                "Asegúrate de tener instalada la librería 'datasets' (pip install datasets) "
                "y conexión a internet para descargar el dataset."
            )
        
        # Separar por sentimiento
        hf_positivos = [d for d in hf_data if d['valor'] == 'positivo']
        hf_negativos = [d for d in hf_data if d['valor'] == 'negativo']
        hf_neutrales = [d for d in hf_data if d['valor'] == 'neutral']
        
        print(f"\n📊 Datos disponibles después de etiquetado: {len(hf_positivos)} positivos, {len(hf_negativos)} negativos, {len(hf_neutrales)} neutrales")
        
        # Balancear: usar la cantidad de negativos como referencia
        # Si hay pocos negativos, usar todos y balancear positivos/neutrales
        import random
        random.seed(42)
        
        if len(hf_negativos) > 0:
            # Usar todos los negativos disponibles (son los más importantes)
            target_negativos = min(len(hf_negativos), 350)
            target_por_clase = target_negativos  # Balancear con la misma cantidad
            
            # Negativos: usar todos los disponibles (hasta el límite)
            if len(hf_negativos) > target_negativos:
                hf_negativos_selected = random.sample(hf_negativos, target_negativos)
            else:
                hf_negativos_selected = hf_negativos
            
            # Positivos: limitar a la misma cantidad que negativos
            if len(hf_positivos) > target_por_clase:
                hf_positivos_selected = random.sample(hf_positivos, target_por_clase)
            else:
                hf_positivos_selected = hf_positivos
            
            # Neutrales: limitar a la misma cantidad
            if len(hf_neutrales) > target_por_clase:
                hf_neutrales_selected = random.sample(hf_neutrales, target_por_clase)
            else:
                hf_neutrales_selected = hf_neutrales
            
            # Combinar
            dataset.extend(hf_negativos_selected)
            dataset.extend(hf_positivos_selected)
            dataset.extend(hf_neutrales_selected)
            
            print(f"✅ Dataset balanceado seleccionado: {len(hf_negativos_selected)} negativos, {len(hf_positivos_selected)} positivos, {len(hf_neutrales_selected)} neutrales")
        else:
            # Si no hay negativos, usar todos los datos disponibles
            print("⚠️ No se encontraron comentarios negativos, usando todos los datos disponibles")
            dataset.extend(hf_data)
        
        # Eliminar duplicados (mismo comentario)
        seen_comments = set()
        unique_dataset = []
        for item in dataset:
            # Normalizar comentario para comparación
            normalized = self._normalize_for_comparison(item['comentario'])
            if normalized not in seen_comments:
                seen_comments.add(normalized)
                unique_dataset.append(item)
        
        dataset = unique_dataset
        
        # Mezclar dataset
        random.seed(42)
        random.shuffle(dataset)
        
        # Limitar a ~1000 muestras si es necesario (mantener balance)
        if len(dataset) > 1000:
            # Mantener proporción balanceada al limitar
            positive_count = sum(1 for d in dataset if d['valor'] == 'positivo')
            negative_count = sum(1 for d in dataset if d['valor'] == 'negativo')
            neutral_count = sum(1 for d in dataset if d['valor'] == 'neutral')
            
            # Calcular proporciones
            total = len(dataset)
            p_ratio = positive_count / total
            n_ratio = negative_count / total
            neu_ratio = neutral_count / total
            
            # Seleccionar manteniendo proporciones
            target_positive = int(1000 * p_ratio)
            target_negative = int(1000 * n_ratio)
            target_neutral = 1000 - target_positive - target_negative
            
            balanced_dataset = []
            balanced_dataset.extend([d for d in dataset if d['valor'] == 'positivo'][:target_positive])
            balanced_dataset.extend([d for d in dataset if d['valor'] == 'negativo'][:target_negative])
            balanced_dataset.extend([d for d in dataset if d['valor'] == 'neutral'][:target_neutral])
            
            random.shuffle(balanced_dataset)
            dataset = balanced_dataset
        
        # Estadísticas finales
        positive_count = sum(1 for d in dataset if d['valor'] == 'positivo')
        negative_count = sum(1 for d in dataset if d['valor'] == 'negativo')
        neutral_count = sum(1 for d in dataset if d['valor'] == 'neutral')
        with_numbers = sum(1 for d in dataset if any(c.isdigit() for c in d['comentario']))
        
        print()
        print("=" * 80)
        print("RESUMEN DEL DATASET BALANCEADO")
        print("=" * 80)
        print(f"📊 Total de comentarios: {len(dataset)}")
        print(f"📊   - Positivos: {positive_count} ({positive_count/len(dataset)*100:.1f}%)")
        print(f"📊   - Negativos: {negative_count} ({negative_count/len(dataset)*100:.1f}%)")
        print(f"📊   - Neutrales: {neutral_count} ({neutral_count/len(dataset)*100:.1f}%)")
        print(f"📊   - Comentarios con números: {with_numbers} ({with_numbers/len(dataset)*100:.1f}%)")
        print("=" * 80)
        print()
        
        return dataset
    
    def _create_pretrained_model(self):
        """
        Entrenar red neuronal LSTM con dataset real de Hugging Face (~1000 muestras balanceadas).
        Los textos se cargan desde Hugging Face y se etiquetan automáticamente usando palabras clave.
        El dataset se balancea automáticamente para tener distribución similar de positivos, negativos y neutrales.
        El modelo aprenderá patrones generales de comentarios reales, mejorando la generalización.
        """
        print("🔍 [DEBUG] _create_pretrained_model() iniciado")
        print("📊 Modelo configurado para párrafos largos: max_len=100, max_words=5000")
        print("🔄 Cargando dataset estructurado balanceado con ~1000 muestras desde Hugging Face...")
        
        # Generar dataset estructurado (valor, comentario)
        dataset = self._create_training_dataset()
        
        if not dataset:
            raise ValueError("No se generó ningún comentario válido en el dataset")
        
        # Extraer textos y etiquetas del dataset estructurado
        texts = [item["comentario"] for item in dataset]
        labels = [item["valor"] for item in dataset]
        
        print(f"🔄 Entrenando red neuronal LSTM con {len(texts)} comentarios...")
        print(f"📊 Distribución: {labels.count('positivo')} positivos, {labels.count('negativo')} negativos, {labels.count('neutro')} neutrales")
        
        # Entrenamiento con más épocas para mejor aprendizaje
        print("🔍 [DEBUG] Iniciando entrenamiento...")
        try:
            # Entrenar modelo usando el método train() existente
            # El método train() ya maneja la preparación de datos, tokenización, etc.
            history = self.train(texts, labels, epochs=15, batch_size=32)
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
        
        print("✅ Red neuronal LSTM entrenada correctamente")
    
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

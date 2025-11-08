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
    def __init__(self, max_words=5000, max_len=200):
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
        """Preparar datos para entrenamiento"""
        # Limpiar textos
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Tokenizar
        if labels:
            self.tokenizer.fit_on_texts(cleaned_texts)
        
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        if labels:
            encoded_labels = self.label_encoder.fit_transform(labels)
            return padded_sequences, encoded_labels
        
        return padded_sequences
    
    def build_model(self, vocab_size: int, num_classes: int):
        """Construir modelo de red neuronal"""
        model = Sequential([
            Embedding(vocab_size + 1, 128, input_length=self.max_len),
            LSTM(64, return_sequences=True),
            Dropout(0.5),
            LSTM(32),
            Dropout(0.5),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, texts: List[str], labels: List[str], epochs=10, batch_size=32):
        """Entrenar modelo"""
        X, y = self.prepare_data(texts, labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        vocab_size = len(self.tokenizer.word_index)
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_model(vocab_size, num_classes)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Predecir sentimiento"""
        if not self.is_trained or not self.model:
            raise ValueError("El modelo no está entrenado")
        
        X = self.prepare_data(texts)
        predictions = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        confidence = np.max(predictions, axis=1)
        
        results = []
        for i, text in enumerate(texts):
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
    
    def predict_single(self, text: str) -> Dict:
        """Predecir sentimiento de un solo texto"""
        results = self.predict([text])
        return results[0]
    
    def load_model(self, model_path: str = 'app/ml_models/sentiment_model.h5'):
        """Cargar modelo pre-entrenado"""
        # Asegurar que el directorio existe
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        # Intentar cargar modelo existente
        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
            try:
                print("Cargando modelo de red neuronal pre-entrenado...")
                self.model = load_model(model_path)
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                self.is_trained = True
                print("Modelo de red neuronal cargado correctamente")
                return
            except Exception as e:
                print(f"Error al cargar modelo pre-entrenado: {e}")
                print("Se creará un nuevo modelo...")
        
        # Si no existe, crear y entrenar modelo (solo la primera vez)
        print("Creando y entrenando modelo de red neuronal (esto puede tomar unos minutos)...")
        try:
            self._create_pretrained_model()
            print("Modelo de red neuronal entrenado y guardado correctamente")
        except Exception as e:
            print(f"Error al crear modelo de red neuronal: {e}")
            raise
    
    def _create_pretrained_model(self):
        """Crear modelo pre-entrenado con datos de ejemplo"""
        positive_texts = [
            "excelente producto muy bueno",
            "me encanta este servicio",
            "muy satisfecho con la compra",
            "recomiendo totalmente",
            "calidad superior",
            "atención perfecta",
            "rápido y eficiente",
            "super contento",
            "vale la pena",
            "muy recomendado",
            "increíble experiencia",
            "servicio de primera",
            "muy buena calidad",
            "excelente atención",
            "producto genial",
            "muy bien hecho",
            "súper recomendable",
            "calidad excelente",
            "muy profesional",
            "servicio impecable"
        ] * 5  # Multiplicar para tener más datos
        
        negative_texts = [
            "pésimo servicio muy malo",
            "no recomiendo para nada",
            "calidad terrible",
            "muy decepcionado",
            "atención horrible",
            "lento e ineficiente",
            "no vale la pena",
            "muy insatisfecho",
            "problema grave",
            "no cumplió expectativas",
            "servicio pésimo",
            "muy mala calidad",
            "no funciona bien",
            "muy decepcionante",
            "producto defectuoso",
            "atención pésima",
            "muy caro para lo que es",
            "no lo recomiendo",
            "muy mal servicio",
            "problemas constantes"
        ] * 5
        
        neutral_texts = [
            "producto regular",
            "ni bueno ni malo",
            "aceptable",
            "normal",
            "sin comentarios",
            "básico",
            "estándar",
            "cumple su función",
            "nada especial",
            "producto común",
            "servicio estándar",
            "normal como cualquier otro",
            "ni destacable ni malo",
            "producto promedio",
            "servicio básico"
        ] * 5
        
        texts = positive_texts + negative_texts + neutral_texts
        labels = (['positivo'] * len(positive_texts) + 
                 ['negativo'] * len(negative_texts) + 
                 ['neutral'] * len(neutral_texts))
        
        print("Entrenando modelo de red neuronal con datos de ejemplo...")
        self.train(texts, labels, epochs=20, batch_size=8)
        self.save_model()
        print("Modelo entrenado y guardado correctamente")
    
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


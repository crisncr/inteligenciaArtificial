import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from datetime import datetime, timedelta

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = None
        self.is_trained = False
    
    def prepare_data(self, df: pd.DataFrame, region: str = None) -> Tuple:
        """Preparar datos para entrenamiento"""
        if region:
            df = df[df['region'] == region].copy()
        
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['dia_mes'] = df['fecha'].dt.day
        df['mes'] = df['fecha'].dt.month
        df['año'] = df['fecha'].dt.year
        df['dia_semana'] = df['fecha'].dt.dayofweek
        
        df = df.sort_values('fecha')
        
        X = df[['dia_mes', 'mes', 'año', 'dia_semana']].values
        y = df['ventas'].values
        
        return X, y, df
    
    def train_linear_regression(self, df: pd.DataFrame, region: str = None):
        """Entrenar modelo de regresión lineal - Parte 3"""
        X, y, _ = self.prepare_data(df, region)
        
        if len(X) == 0:
            raise ValueError("No hay datos para la región especificada")
        
        X = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        test_predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        
        self.model_type = 'linear_regression'
        self.is_trained = True
        
        return {
            "train_score": float(train_score),
            "test_score": float(test_score),
            "mse": float(mse),
            "r2_score": float(r2),
            "model_type": "Regresión Lineal",
            "justification": "La regresión lineal es eficiente para datos con tendencias lineales y proporciona interpretabilidad clara de las relaciones entre variables."
        }
    
    def train_neural_network(self, df: pd.DataFrame, region: str = None, epochs=50):
        """Entrenar modelo de red neuronal - Parte 3"""
        X, y, _ = self.prepare_data(df, region)
        
        if len(X) == 0:
            raise ValueError("No hay datos para la región especificada")
        
        X = self.scaler.fit_transform(X)
        y = y.reshape(-1, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        test_predictions = self.model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        
        self.model_type = 'neural_network'
        self.is_trained = True
        
        return {
            "train_loss": float(train_loss[0]),
            "test_loss": float(test_loss[0]),
            "mse": float(mse),
            "r2_score": float(r2),
            "model_type": "Red Neuronal",
            "justification": "Las redes neuronales capturan relaciones no lineales complejas y patrones temporales en los datos de ventas, mejorando la precisión en predicciones."
        }
    
    def predict(self, start_date: str, days: int = 30, region: str = None) -> List[Dict]:
        """Predecir ventas futuras"""
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        dates = [start + timedelta(days=i) for i in range(days)]
        
        X = np.array([[
            d.day,
            d.month,
            d.year,
            d.weekday()
        ] for d in dates])
        
        X = self.scaler.transform(X)
        
        if self.model_type == 'linear_regression':
            predictions = self.model.predict(X)
        else:
            predictions = self.model.predict(X, verbose=0).flatten()
        
        results = []
        for i, date in enumerate(dates):
            results.append({
                "fecha": date.strftime('%Y-%m-%d'),
                "ventas_predichas": float(predictions[i]),
                "region": region
            })
        
        return results


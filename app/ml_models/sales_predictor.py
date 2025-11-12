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
        self.models = {}  # Diccionario: clave = (producto, region), valor = modelo
        self.scalers = {}  # Diccionario: clave = (producto, region), valor = scaler
        self.model_type = None
        self.is_trained = False
        self.products = []  # Lista de productos entrenados
        self.regions = []  # Lista de regiones entrenadas
        self.product_region_combinations = []  # Lista de combinaciones (producto, region)
    
    def prepare_data(self, df: pd.DataFrame, region: str = None, producto: str = None) -> Tuple:
        """Preparar datos para entrenamiento - Filtra por región y producto"""
        # Filtrar por región si se especifica
        if region and region.strip():
            df = df[df['region'] == region].copy()
            if len(df) == 0:
                raise ValueError(f"No hay datos para la región '{region}'")
        
        # Filtrar por producto si se especifica
        if producto and producto.strip():
            df = df[df['producto'] == producto].copy()
            if len(df) == 0:
                raise ValueError(f"No hay datos para el producto '{producto}'")
        
        if len(df) == 0:
            raise ValueError("No hay datos para procesar")
        
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
        """Entrenar modelo de regresión lineal - UN MODELO POR CADA PRODUCTO-REGION"""
        # Validar que existe la columna 'producto'
        if 'producto' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'producto'")
        
        # Obtener todas las combinaciones únicas de producto-región
        if region and region.strip():
            df_filtered = df[df['region'] == region].copy()
        else:
            df_filtered = df.copy()
        
        if len(df_filtered) == 0:
            raise ValueError("No hay datos para entrenar")
        
        # Obtener combinaciones únicas de producto-región
        combinations = df_filtered[['producto', 'region']].drop_duplicates()
        self.product_region_combinations = [
            (row['producto'], row['region']) 
            for _, row in combinations.iterrows()
        ]
        
        # Obtener listas únicas
        self.products = df_filtered['producto'].unique().tolist()
        self.regions = df_filtered['region'].unique().tolist()
        
        results = {}
        trained_count = 0
        
        # Entrenar un modelo por cada combinación producto-región
        for producto, reg in self.product_region_combinations:
            try:
                X, y, df_subset = self.prepare_data(df, reg, producto)
                
                # Validar que hay suficientes datos (mínimo 5 para train/test split)
                if len(X) < 5:
                    print(f"⚠️ Saltando {producto} - {reg}: muy pocos datos ({len(X)})")
                    continue
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Si hay muy pocos datos, usar todos para entrenamiento
                if len(X_scaled) < 10:
                    X_train, y_train = X_scaled, y
                    X_test, y_test = X_scaled, y
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42
                    )
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                test_predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, test_predictions)
                r2 = r2_score(y_test, test_predictions)
                train_score = model.score(X_train, y_train)
                
                # Guardar modelo y scaler
                key = (producto, reg)
                self.models[key] = model
                self.scalers[key] = scaler
                
                results[f"{producto} - {reg}"] = {
                    "mse": float(mse),
                    "r2_score": float(r2),
                    "train_score": float(train_score),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "total_samples": len(X)
                }
                
                trained_count += 1
                
            except Exception as e:
                print(f"❌ Error entrenando modelo para '{producto} - {reg}': {str(e)}")
                continue
        
        if trained_count == 0:
            raise ValueError("No se pudo entrenar ningún modelo. Verifica que hay suficientes datos.")
        
        self.model_type = 'linear_regression'
        self.is_trained = True
        
        # Calcular promedios
        if results:
            avg_r2 = sum(r['r2_score'] for r in results.values()) / len(results)
            avg_mse = sum(r['mse'] for r in results.values()) / len(results)
        else:
            avg_r2 = 0.0
            avg_mse = 0.0
        
        return {
            "model_type": "Regresión Lineal",
            "products_trained": trained_count,
            "products": sorted(self.products),
            "regions": sorted(self.regions),
            "combinations": [f"{p} - {r}" for p, r in self.product_region_combinations],
            "average_r2_score": float(avg_r2),
            "average_mse": float(avg_mse),
            "product_results": results,
            "justification": "Modelo de regresión lineal entrenado por producto-región, considerando fecha y ventas históricas."
        }
    
    def train_neural_network(self, df: pd.DataFrame, region: str = None, epochs=50):
        """Entrenar modelo de red neuronal - Parte 3"""
        X, y, _ = self.prepare_data(df, region)
        
        # prepare_data ya valida que hay datos, pero por seguridad
        if len(X) == 0:
            raise ValueError("No hay datos para entrenar el modelo")
        
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
        
        # model.evaluate() devuelve una lista/array numpy: [loss, metric1, metric2, ...]
        # En este caso: [loss, mae] porque compile tiene metrics=['mae']
        train_loss_result = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss_result = self.model.evaluate(X_test, y_test, verbose=0)
        test_predictions = self.model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        
        # Manejar diferentes tipos de retorno de evaluate()
        # Puede ser lista, tupla, array numpy, o escalar
        if isinstance(train_loss_result, (list, tuple)):
            train_loss = train_loss_result[0]
        elif isinstance(train_loss_result, np.ndarray):
            train_loss = train_loss_result.item() if train_loss_result.size == 1 else train_loss_result[0]
        else:
            train_loss = train_loss_result
            
        if isinstance(test_loss_result, (list, tuple)):
            test_loss = test_loss_result[0]
        elif isinstance(test_loss_result, np.ndarray):
            test_loss = test_loss_result.item() if test_loss_result.size == 1 else test_loss_result[0]
        else:
            test_loss = test_loss_result
        
        self.model_type = 'neural_network'
        self.is_trained = True
        
        return {
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "mse": float(mse),
            "r2_score": float(r2),
            "model_type": "Red Neuronal",
            "justification": "Las redes neuronales capturan relaciones no lineales complejas y patrones temporales en los datos de ventas, mejorando la precisión en predicciones."
        }
    
    def predict(self, start_date: str, days: int = 30, region: str = None, producto: str = None) -> List[Dict]:
        """Predecir ventas futuras - Por producto y región"""
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado")
        
        # Determinar qué combinaciones predecir
        if producto and region:
            # Predecir solo una combinación específica
            keys_to_predict = [(producto, region)]
        elif producto:
            # Predecir el producto en todas sus regiones
            keys_to_predict = [(producto, r) for r in self.regions if (producto, r) in self.models]
        elif region:
            # Predecir todos los productos de esa región
            keys_to_predict = [(p, region) for p in self.products if (p, region) in self.models]
        else:
            # Predecir todas las combinaciones
            keys_to_predict = list(self.models.keys())
        
        if not keys_to_predict:
            raise ValueError(f"No hay modelos entrenados para producto='{producto}', region='{region}'")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        dates = [start + timedelta(days=i) for i in range(days)]
        
        all_results = []
        
        for producto_key, region_key in keys_to_predict:
            if (producto_key, region_key) not in self.models:
                continue
            
            model = self.models[(producto_key, region_key)]
            scaler = self.scalers[(producto_key, region_key)]
        
        X = np.array([[
            d.day,
            d.month,
            d.year,
            d.weekday()
        ] for d in dates])
        
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            for i, date in enumerate(dates):
                all_results.append({
                    "fecha": date.strftime('%Y-%m-%d'),
                    "producto": producto_key,
                    "region": region_key,
                    "ventas_predichas": float(predictions[i])
                })
        
        return all_results
    
    def get_historical_data(self, df: pd.DataFrame, producto: str = None, region: str = None) -> List[Dict]:
        """Obtener datos históricos para gráficos"""
        df_filtered = df.copy()
        
        if producto and producto.strip():
            df_filtered = df_filtered[df_filtered['producto'] == producto]
        
        if region and region.strip():
            df_filtered = df_filtered[df_filtered['region'] == region]
        
        df_filtered['fecha'] = pd.to_datetime(df_filtered['fecha'])
        df_filtered = df_filtered.sort_values('fecha')
        
        results = []
        for _, row in df_filtered.iterrows():
            results.append({
                "fecha": row['fecha'].strftime('%Y-%m-%d'),
                "producto": row['producto'],
                "region": row['region'],
                "ventas": float(row['ventas']),
                "valor": float(row['valor']) if 'valor' in row else None
            })
        
        return results


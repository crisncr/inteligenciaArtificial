# mostrar_dataset.py
"""
Script para mostrar el dataset estructurado que se usa para entrenar el modelo.
"""
import sys
import os
import pandas as pd

# Configurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml_models.sentiment_nn import SentimentNeuralNetwork

def main():
    print("=" * 80)
    print("DATASET ESTRUCTURADO PARA ENTRENAMIENTO")
    print("=" * 80)
    print()
    
    # Crear modelo y generar dataset
    model = SentimentNeuralNetwork()
    dataset = model._create_training_dataset()
    
    # Convertir a DataFrame
    df = pd.DataFrame(dataset)
    
    # Estadísticas generales
    print(f"📊 Total de muestras: {len(df)}")
    print(f"📊 Distribución por sentimiento:")
    print(df['valor'].value_counts().to_string())
    print()
    
    # Comentarios con números
    with_numbers = df['comentario'].apply(lambda x: any(c.isdigit() for c in x)).sum()
    print(f"📊 Comentarios con números: {with_numbers} ({with_numbers/len(df)*100:.1f}%)")
    print()
    
    # Mostrar muestras por categoría
    print("=" * 80)
    print("MUESTRAS POSITIVAS (primeras 30)")
    print("=" * 80)
    positive_samples = df[df['valor'] == 'positivo'].head(30)
    for idx, row in positive_samples.iterrows():
        print(f"{row['valor']:10} | {row['comentario']}")
    print()
    
    print("=" * 80)
    print("MUESTRAS NEGATIVAS (primeras 30)")
    print("=" * 80)
    negative_samples = df[df['valor'] == 'negativo'].head(30)
    for idx, row in negative_samples.iterrows():
        print(f"{row['valor']:10} | {row['comentario']}")
    print()
    
    print("=" * 80)
    print("MUESTRAS NEUTRALES (primeras 30)")
    print("=" * 80)
    neutral_samples = df[df['valor'] == 'neutro'].head(30)
    for idx, row in neutral_samples.iterrows():
        print(f"{row['valor']:10} | {row['comentario']}")
    print()
    
    # Mostrar muestras con números
    print("=" * 80)
    print("MUESTRAS CON NÚMEROS (ejemplos)")
    print("=" * 80)
    samples_with_numbers = df[df['comentario'].apply(lambda x: any(c.isdigit() for c in x))].head(20)
    for idx, row in samples_with_numbers.iterrows():
        print(f"{row['valor']:10} | {row['comentario']}")
    print()
    
    # Guardar a CSV
    csv_path = 'dataset_entrenamiento.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"💾 Dataset guardado en: {csv_path}")
    print()
    
    # Mostrar tabla completa en formato más legible (primeras 50)
    print("=" * 80)
    print("TABLA COMPLETA DEL DATASET (primeras 50 filas)")
    print("=" * 80)
    print(df.head(50).to_string(index=True))
    print()
    print(f"... y {len(df) - 50} filas más")
    print()
    print(f"📋 Para ver todo el dataset, abre: {csv_path}")

if __name__ == "__main__":
    main()


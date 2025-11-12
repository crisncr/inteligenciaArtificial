# test_load_csv.py
"""
Script para probar la carga del CSV de opiniones
"""
import sys
import os
import pandas as pd
import io

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

def main():
    csv_path = r'c:\Users\crist\Downloads\opiniones_clientes.csv'
    
    print("=" * 80)
    print("PRUEBA DE CARGA DE CSV")
    print("=" * 80)
    print()
    print(f"📖 Archivo: {csv_path}")
    print()
    
    # Verificar que el archivo existe
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: El archivo no existe en: {csv_path}")
        return
    
    print(f"✅ Archivo encontrado")
    file_size = os.path.getsize(csv_path) / 1024  # KB
    print(f"📊 Tamaño: {file_size:.2f} KB")
    print()
    
    # Intentar leer el CSV con diferentes encodings
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    df = None
    encoding_used = None
    
    for encoding in encodings:
        try:
            print(f"🔄 Intentando leer con encoding: {encoding}...")
            df = pd.read_csv(csv_path, encoding=encoding)
            encoding_used = encoding
            print(f"✅ CSV leído correctamente con encoding: {encoding}")
            break
        except Exception as e:
            print(f"❌ Error con {encoding}: {str(e)[:100]}")
            continue
    
    if df is None:
        print("❌ No se pudo leer el CSV con ningún encoding")
        return
    
    print()
    print("=" * 80)
    print("INFORMACIÓN DEL CSV")
    print("=" * 80)
    print(f"📊 Total de filas: {len(df)}")
    print(f"📊 Total de columnas: {len(df.columns)}")
    print(f"📊 Columnas: {list(df.columns)}")
    print()
    
    # Mostrar tipos de datos
    print("📊 TIPOS DE DATOS:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        print(f"   - {col}: {dtype} (nulos: {null_count})")
    print()
    
    # Buscar columna de opinión
    text_column = None
    if 'opinion' in df.columns:
        text_column = 'opinion'
    elif 'texto' in df.columns:
        text_column = 'texto'
    elif 'comentario' in df.columns:
        text_column = 'comentario'
    else:
        for col in df.columns:
            if df[col].dtype == 'object':
                text_column = col
                break
    
    if not text_column:
        print("❌ No se encontró columna de texto")
        return
    
    print(f"✅ Columna de texto detectada: '{text_column}'")
    print()
    
    # Extraer textos
    texts = df[text_column].dropna().astype(str).tolist()
    texts = [t.strip() for t in texts if len(t.strip()) >= 2]
    
    print(f"📝 Total de textos válidos: {len(texts)}")
    print()
    
    # Mostrar distribución si hay columna de sentimiento
    if 'sentimiento' in df.columns:
        print("=" * 80)
        print("DISTRIBUCIÓN DE SENTIMIENTOS")
        print("=" * 80)
        distribution = df['sentimiento'].value_counts()
        print(distribution.to_string())
        print()
        
        # Comparar con textos extraídos
        print("📊 COMPARACIÓN:")
        print(f"   - Filas en CSV: {len(df)}")
        print(f"   - Textos válidos extraídos: {len(texts)}")
        print(f"   - Diferencia: {len(df) - len(texts)} textos descartados (vacíos o muy cortos)")
        print()
    
    # Mostrar primeros textos
    print("=" * 80)
    print("PRIMEROS 5 TEXTOS")
    print("=" * 80)
    for i, text in enumerate(texts[:5], 1):
        print(f"[{i}] {text[:80]}{'...' if len(text) > 80 else ''}")
    print()
    
    # Mostrar últimos textos
    print("=" * 80)
    print("ÚLTIMOS 5 TEXTOS")
    print("=" * 80)
    for i, text in enumerate(texts[-5:], len(texts)-4):
        print(f"[{i}] {text[:80]}{'...' if len(text) > 80 else ''}")
    print()
    
    # Estadísticas de longitud
    lengths = [len(t) for t in texts]
    print("=" * 80)
    print("ESTADÍSTICAS DE LONGITUD")
    print("=" * 80)
    print(f"   - Longitud mínima: {min(lengths)} caracteres")
    print(f"   - Longitud máxima: {max(lengths)} caracteres")
    print(f"   - Longitud promedio: {sum(lengths)/len(lengths):.1f} caracteres")
    print()
    
    print("=" * 80)
    print("✅ CARGA DE CSV COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    print(f"📊 Resumen: {len(texts)} textos listos para análisis")
    print()

if __name__ == "__main__":
    main()






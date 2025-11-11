# analizar_csv_sentimientos.py
"""
Script para analizar CSV de opiniones y mostrar resultados igual que en la web
"""
import sys
import os
import csv
import io
import pandas as pd

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.sentiment import analyze_sentiment

def main():
    csv_path = r'c:\Users\crist\Downloads\opiniones_clientes.csv'
    
    print("=" * 80)
    print("ANÁLISIS DE SENTIMIENTOS - CSV")
    print("=" * 80)
    print()
    
    # Leer CSV
    print(f"📖 Leyendo archivo: {csv_path}")
    print()
    
    try:
        # Intentar diferentes encodings
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        df = None
        encoding_used = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                encoding_used = encoding
                print(f"✅ CSV leído correctamente con encoding: {encoding}")
                break
            except Exception:
                continue
        
        if df is None:
            print("❌ No se pudo leer el CSV con ningún encoding")
            return
        
        print(f"📊 Total de filas: {len(df)}")
        print(f"📊 Columnas: {list(df.columns)}")
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
            # Usar la primera columna que parezca texto
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_column = col
                    break
        
        if not text_column:
            print("❌ No se encontró columna de texto")
            return
        
        print(f"✅ Columna de texto: '{text_column}'")
        print()
        
        # Extraer textos
        texts = df[text_column].dropna().astype(str).tolist()
        texts = [t.strip() for t in texts if len(t.strip()) >= 2]
        
        print(f"📝 Total de opiniones a analizar: {len(texts)}")
        print()
        print("=" * 80)
        print("ANALIZANDO OPINIONES...")
        print("=" * 80)
        print()
        
        # Analizar cada opinión
        resultados = []
        positivos = 0
        negativos = 0
        neutrales = 0
        
        for i, text in enumerate(texts, 1):
            try:
                # Usar la misma función que la web
                result = analyze_sentiment(text)
                
                sentiment = result.get('sentiment', 'neutral')
                score = result.get('score', 0.0)
                emoji = result.get('emoji', '😐')
                confidence = result.get('confidence', 0.0)
                
                # Contar
                if sentiment == 'positivo':
                    positivos += 1
                elif sentiment == 'negativo':
                    negativos += 1
                else:
                    neutrales += 1
                
                resultados.append({
                    'texto': text,
                    'sentiment': sentiment,
                    'score': score,
                    'emoji': emoji,
                    'confidence': confidence
                })
                
                # Mostrar resultado igual que en la web
                print(f"[{i:2d}] {emoji} {sentiment.upper():8s} | Score: {score:+.3f} | Confianza: {confidence*100:5.1f}%")
                print(f"     {text}")
                print()
                
            except Exception as e:
                print(f"❌ Error al analizar opinión {i}: {str(e)}")
                print(f"   Texto: {text[:80]}...")
                print()
        
        # Resumen final
        print("=" * 80)
        print("RESUMEN DE ANÁLISIS")
        print("=" * 80)
        print(f"📊 Total analizado: {len(resultados)}")
        if len(resultados) > 0:
            print(f"✅ Positivos: {positivos} ({positivos/len(resultados)*100:.1f}%)")
            print(f"❌ Negativos: {negativos} ({negativos/len(resultados)*100:.1f}%)")
            print(f"😐 Neutrales: {neutrales} ({neutrales/len(resultados)*100:.1f}%)")
        print()
        
        # Mostrar distribución
        print("=" * 80)
        print("DISTRIBUCIÓN DE SENTIMIENTOS")
        print("=" * 80)
        for r in resultados:
            print(f"{r['emoji']} {r['sentiment']:8s} | {r['texto'][:60]}...")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


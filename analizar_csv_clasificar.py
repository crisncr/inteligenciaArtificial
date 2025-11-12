# analizar_csv_clasificar.py
"""
Script para analizar CSV y mostrar textos clasificados como positivos o negativos
"""
import sys
import os
import pandas as pd
import io
import time

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    csv_path = r'c:\Users\crist\Downloads\opiniones_clientes.csv'
    
    print("=" * 80)
    print("ANÁLISIS DE SENTIMIENTOS - CLASIFICACIÓN")
    print("=" * 80)
    print()
    print(f"📖 Archivo: {csv_path}")
    print()
    
    # Leer CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        texts = df['opinion'].dropna().astype(str).tolist()
        texts = [t.strip() for t in texts if len(t.strip()) >= 2]
        print(f"✅ CSV leído: {len(texts)} textos")
    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
        return
    
    print()
    print("⏳ Cargando modelo de sentimientos...")
    print("   (Esto puede tardar unos segundos la primera vez)")
    print()
    
    # Importar y analizar
    try:
        from app.sentiment import analyze_sentiment
        
        # Analizar todos los textos
        print("🔄 Analizando textos...")
        print()
        
        start_time = time.time()
        resultados = []
        
        for i, text in enumerate(texts, 1):
            try:
                result = analyze_sentiment(text)
                sentiment = result.get('sentiment', 'neutral')
                confidence = result.get('confidence', 0.0)
                emoji = result.get('emoji', '😐')
                
                resultados.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'emoji': emoji
                })
                
                # Mostrar progreso cada 10 textos
                if i % 10 == 0:
                    print(f"   ✅ Procesados {i}/{len(texts)} textos...")
                    
            except Exception as e:
                print(f"   ❌ Error en texto {i}: {str(e)[:100]}")
                resultados.append({
                    'text': text,
                    'sentiment': 'error',
                    'confidence': 0.0,
                    'emoji': '❌'
                })
        
        total_time = time.time() - start_time
        print()
        print(f"✅ Análisis completado en {total_time:.2f} segundos")
        print()
        
        # Separar por sentimiento
        positivos = [r for r in resultados if r['sentiment'] == 'positivo']
        negativos = [r for r in resultados if r['sentiment'] == 'negativo']
        neutrales = [r for r in resultados if r['sentiment'] == 'neutral']
        errores = [r for r in resultados if r['sentiment'] == 'error']
        
        # Mostrar resultados
        print("=" * 80)
        print("RESULTADOS - TEXTOS POSITIVOS")
        print("=" * 80)
        print(f"📊 Total: {len(positivos)} textos positivos")
        print()
        for i, r in enumerate(positivos, 1):
            conf = r['confidence']
            emoji = r['emoji']
            text = r['text']
            print(f"[{i}] {emoji} (confianza: {conf:.3f}) - {text}")
        print()
        
        print("=" * 80)
        print("RESULTADOS - TEXTOS NEGATIVOS")
        print("=" * 80)
        print(f"📊 Total: {len(negativos)} textos negativos")
        print()
        for i, r in enumerate(negativos, 1):
            conf = r['confidence']
            emoji = r['emoji']
            text = r['text']
            print(f"[{i}] {emoji} (confianza: {conf:.3f}) - {text}")
        print()
        
        if neutrales:
            print("=" * 80)
            print("RESULTADOS - TEXTOS NEUTRALES")
            print("=" * 80)
            print(f"📊 Total: {len(neutrales)} textos neutrales")
            print()
            for i, r in enumerate(neutrales, 1):
                conf = r['confidence']
                emoji = r['emoji']
                text = r['text']
                print(f"[{i}] {emoji} (confianza: {conf:.3f}) - {text}")
            print()
        
        if errores:
            print("=" * 80)
            print("ERRORES")
            print("=" * 80)
            print(f"📊 Total: {len(errores)} textos con error")
            print()
            for i, r in enumerate(errores, 1):
                text = r['text']
                print(f"[{i}] ❌ - {text}")
            print()
        
        # Resumen final
        print("=" * 80)
        print("RESUMEN FINAL")
        print("=" * 80)
        print(f"📊 Total analizados: {len(resultados)}")
        print(f"✅ Positivos: {len(positivos)} ({len(positivos)/len(resultados)*100:.1f}%)")
        print(f"❌ Negativos: {len(negativos)} ({len(negativos)/len(resultados)*100:.1f}%)")
        if neutrales:
            print(f"😐 Neutrales: {len(neutrales)} ({len(neutrales)/len(resultados)*100:.1f}%)")
        if errores:
            print(f"⚠️  Errores: {len(errores)} ({len(errores)/len(resultados)*100:.1f}%)")
        print()
        
        # Comparar con etiquetas reales si están disponibles
        if 'sentimiento' in df.columns:
            print("=" * 80)
            print("COMPARACIÓN CON ETIQUETAS REALES")
            print("=" * 80)
            
            # Mapear sentimientos reales
            real_labels = df['sentimiento'].dropna().tolist()
            real_pos = sum(1 for l in real_labels if 'positiv' in l.lower())
            real_neg = sum(1 for l in real_labels if 'negativ' in l.lower())
            
            # Mapear predicciones
            pred_pos = len(positivos)
            pred_neg = len(negativos)
            
            print(f"📊 Etiquetas reales: {real_pos} positivas, {real_neg} negativas")
            print(f"📊 Predicciones: {pred_pos} positivas, {pred_neg} negativas")
            print()
            
            # Calcular precisión
            if len(real_labels) == len(resultados):
                correctos = 0
                for i, (real, pred) in enumerate(zip(real_labels, resultados)):
                    real_sent = 'positivo' if 'positiv' in real.lower() else 'negativo'
                    pred_sent = pred['sentiment']
                    if real_sent == pred_sent:
                        correctos += 1
                
                precision = (correctos / len(resultados)) * 100
                print(f"✅ Precisión: {correctos}/{len(resultados)} = {precision:.1f}%")
                print()
        
        print("=" * 80)
        print("✅ ANÁLISIS COMPLETADO")
        print("=" * 80)
        
    except ImportError as e:
        print(f"❌ Error al importar módulo: {e}")
        print("   Asegúrate de estar en el directorio raíz del proyecto")
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()





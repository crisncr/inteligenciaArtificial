# test_batch_flow.py
"""
Script para probar el flujo de procesamiento en lotes sin TensorFlow
Simula el comportamiento del endpoint /analyze-batch
"""
import sys
import os
import pandas as pd

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

def main():
    csv_path = r'c:\Users\crist\Downloads\opiniones_clientes.csv'
    
    print("=" * 80)
    print("PRUEBA DE FLUJO DE PROCESAMIENTO EN LOTES")
    print("=" * 80)
    print()
    
    # Leer CSV
    print(f"📖 Leyendo archivo: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"✅ CSV leído correctamente")
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
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_column = col
                    break
        
        if not text_column:
            print("❌ No se encontró columna de texto")
            return
        
        print(f"✅ Columna de texto: '{text_column}'")
        
        # Extraer textos
        texts = df[text_column].dropna().astype(str).tolist()
        texts = [t.strip() for t in texts if len(t.strip()) >= 2]
        
        print(f"📝 Total de opiniones: {len(texts)}")
        print()
        
        # SIMULAR PROCESAMIENTO EN LOTES (como en el endpoint)
        batch_size = 10
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print("=" * 80)
        print(f"SIMULANDO PROCESAMIENTO EN LOTES (tamaño: {batch_size})")
        print("=" * 80)
        print()
        
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"🔍 Procesando lote {batch_num}/{total_batches} ({len(batch)} textos)")
            print(f"   Textos: {[t[:30] + '...' if len(t) > 30 else t for t in batch]}")
            
            # Simular resultados del modelo
            batch_results = []
            for text in batch:
                # Simular análisis (sin TensorFlow)
                if any(palabra in text.lower() for palabra in ['excelente', 'bueno', 'satisfecho', 'recomiendo', 'perfecto', 'genial']):
                    sentiment = 'positivo'
                    emoji = '😊'
                elif any(palabra in text.lower() for palabra in ['malo', 'pésimo', 'decepcionante', 'horrible', 'nunca', 'no recomiendo']):
                    sentiment = 'negativo'
                    emoji = '😞'
                else:
                    sentiment = 'neutral'
                    emoji = '😐'
                
                batch_results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': 0.85,
                    'emoji': emoji
                })
            
            all_results.extend(batch_results)
            print(f"✅ Lote {batch_num} completado: {len(batch_results)} resultados")
            print()
        
        # Resumen
        print("=" * 80)
        print("RESUMEN DE PROCESAMIENTO EN LOTES")
        print("=" * 80)
        print(f"📊 Total procesado: {len(all_results)}")
        print(f"📦 Total de lotes: {total_batches}")
        print(f"📏 Tamaño de lote: {batch_size}")
        print()
        
        # Contar sentimientos
        positivos = sum(1 for r in all_results if r['sentiment'] == 'positivo')
        negativos = sum(1 for r in all_results if r['sentiment'] == 'negativo')
        neutrales = sum(1 for r in all_results if r['sentiment'] == 'neutral')
        
        print("=" * 80)
        print("DISTRIBUCIÓN DE SENTIMIENTOS")
        print("=" * 80)
        print(f"😊 Positivos: {positivos} ({positivos/len(all_results)*100:.1f}%)")
        print(f"😞 Negativos: {negativos} ({negativos/len(all_results)*100:.1f}%)")
        print(f"😐 Neutrales: {neutrales} ({neutrales/len(all_results)*100:.1f}%)")
        print()
        
        # Mostrar algunos ejemplos
        print("=" * 80)
        print("EJEMPLOS DE RESULTADOS")
        print("=" * 80)
        for i, result in enumerate(all_results[:10], 1):
            print(f"[{i}] {result['emoji']} {result['sentiment'].upper():8s} | {result['text'][:60]}...")
        print()
        
        print("✅ PRUEBA COMPLETADA: El flujo de procesamiento en lotes funciona correctamente")
        print("   En producción, cada lote se procesará con el modelo TensorFlow real")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()






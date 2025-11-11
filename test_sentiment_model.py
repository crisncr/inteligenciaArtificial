# test_sentiment_model.py
"""
Script para probar el modelo de sentimientos con un archivo CSV de pruebas.
Formato esperado: opinion,sentimiento (donde sentimiento es "positiva" o "negativa")
"""
import sys
import os
import csv
import io

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml_models.sentiment_nn import SentimentNeuralNetwork

def main():
    csv_path = r'c:\Users\crist\Downloads\opiniones_clientes.csv'
    
    print("=" * 80)
    print("PRUEBA DEL MODELO DE SENTIMIENTOS")
    print("=" * 80)
    print()
    
    # Cargar modelo
    print("📦 Cargando modelo entrenado...")
    model = SentimentNeuralNetwork()
    model.load_model()
    print("✅ Modelo cargado correctamente")
    print()
    
    # Leer CSV
    print(f"📖 Leyendo archivo: {csv_path}")
    test_cases = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                opinion = row.get('opinion', '').strip()
                sentimiento_esperado = row.get('sentimiento', '').strip().lower()
                
                if opinion and sentimiento_esperado:
                    # Convertir "positiva"/"negativa" a "positivo"/"negativo"
                    if sentimiento_esperado == 'positiva':
                        sentimiento_esperado = 'positivo'
                    elif sentimiento_esperado == 'negativa':
                        sentimiento_esperado = 'negativo'
                    
                    test_cases.append({
                        'opinion': opinion,
                        'esperado': sentimiento_esperado
                    })
        
        print(f"✅ {len(test_cases)} casos de prueba cargados")
        print()
    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")
        return
    
    # Hacer predicciones
    print("🔍 Analizando opiniones...")
    print()
    
    resultados = []
    correctos = 0
    incorrectos = 0
    
    for i, test_case in enumerate(test_cases, 1):
        opinion = test_case['opinion']
        esperado = test_case['esperado']
        
        try:
            # Predecir
            resultado = model.predict_single(opinion)
            predicho = resultado['sentiment']
            confianza = resultado['confidence']
            score = resultado['score']
            
            # Comparar
            es_correcto = (predicho == esperado)
            if es_correcto:
                correctos += 1
            else:
                incorrectos += 1
            
            resultados.append({
                'opinion': opinion,
                'esperado': esperado,
                'predicho': predicho,
                'confianza': confianza,
                'score': score,
                'correcto': es_correcto
            })
            
            # Mostrar resultado
            estado = "✅" if es_correcto else "❌"
            print(f"{estado} [{i:2d}] Esperado: {esperado:8s} | Predicho: {predicho:8s} | Confianza: {confianza*100:5.1f}% | Score: {score:+.3f}")
            print(f"    Texto: {opinion}")
            if not es_correcto:
                print(f"    ⚠️ ERROR: Se esperaba '{esperado}' pero se predijo '{predicho}'")
            print()
            
        except Exception as e:
            print(f"❌ Error al analizar: {opinion}")
            print(f"   Error: {e}")
            print()
            incorrectos += 1
    
    # Resumen
    total = len(resultados)
    precision = (correctos / total * 100) if total > 0 else 0
    
    print("=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)
    print(f"📊 Total de casos: {total}")
    print(f"✅ Correctos: {correctos} ({correctos/total*100:.1f}%)")
    print(f"❌ Incorrectos: {incorrectos} ({incorrectos/total*100:.1f}%)")
    print(f"🎯 Precisión: {precision:.1f}%")
    print()
    
    # Mostrar casos incorrectos
    if incorrectos > 0:
        print("=" * 80)
        print("CASOS INCORRECTOS")
        print("=" * 80)
        for r in resultados:
            if not r['correcto']:
                print(f"❌ Esperado: {r['esperado']:8s} | Predicho: {r['predicho']:8s} | Confianza: {r['confianza']*100:5.1f}%")
                print(f"   Texto: {r['opinion']}")
                print()
    
    # Distribución por sentimiento
    print("=" * 80)
    print("DISTRIBUCIÓN DE PREDICCIONES")
    print("=" * 80)
    positivos_pred = sum(1 for r in resultados if r['predicho'] == 'positivo')
    negativos_pred = sum(1 for r in resultados if r['predicho'] == 'negativo')
    neutrales_pred = sum(1 for r in resultados if r['predicho'] == 'neutral')
    
    positivos_esp = sum(1 for r in resultados if r['esperado'] == 'positivo')
    negativos_esp = sum(1 for r in resultados if r['esperado'] == 'negativo')
    neutrales_esp = sum(1 for r in resultados if r['esperado'] == 'neutral')
    
    print(f"Esperados:  Positivos: {positivos_esp:2d} | Negativos: {negativos_esp:2d} | Neutrales: {neutrales_esp:2d}")
    print(f"Predichos:  Positivos: {positivos_pred:2d} | Negativos: {negativos_pred:2d} | Neutrales: {neutrales_pred:2d}")
    print()

if __name__ == "__main__":
    main()


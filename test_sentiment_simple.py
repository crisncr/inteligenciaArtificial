# test_sentiment_simple.py
"""
Script simple para probar el modelo de análisis de sentimientos
"""
import sys
import os
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
    print("=" * 80)
    print("PRUEBA DEL MODELO DE ANÁLISIS DE SENTIMIENTOS")
    print("=" * 80)
    print()
    
    # Cargar modelo
    print("📦 Cargando modelo entrenado...")
    try:
        model = SentimentNeuralNetwork()
        model.load_model()
        print("✅ Modelo cargado correctamente")
        print()
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    # Casos de prueba
    test_cases = [
        # Casos positivos
        {"texto": "El servicio fue excelente, volveré pronto", "esperado": "positivo"},
        {"texto": "Me encantó la atención, muy profesional", "esperado": "positivo"},
        {"texto": "El empaque era bonito y seguro", "esperado": "positivo"},
        {"texto": "Estoy muy satisfecho con la compra", "esperado": "positivo"},
        {"texto": "Superó mis expectativas completamente", "esperado": "positivo"},
        {"texto": "Fácil proceso de compra, todo perfecto", "esperado": "positivo"},
        
        # Casos negativos
        {"texto": "Nunca volveré a comprar aquí", "esperado": "negativo"},
        {"texto": "Muy mala atención, no recomiendo", "esperado": "negativo"},
        {"texto": "El producto llegó dañado y tarde", "esperado": "negativo"},
        {"texto": "Pésimo servicio al cliente", "esperado": "negativo"},
        {"texto": "No cumplieron con lo prometido", "esperado": "negativo"},
        
        # Casos neutrales
        {"texto": "El producto llegó a tiempo", "esperado": "neutral"},
        {"texto": "Recibí el paquete el día indicado", "esperado": "neutral"},
        {"texto": "Información sobre el producto", "esperado": "neutral"},
    ]
    
    print("🔍 Analizando casos de prueba...")
    print()
    
    resultados = []
    correctos = 0
    incorrectos = 0
    
    for i, test_case in enumerate(test_cases, 1):
        texto = test_case['texto']
        esperado = test_case['esperado']
        
        try:
            # Predecir
            resultado = model.predict_single(texto)
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
                'texto': texto,
                'esperado': esperado,
                'predicho': predicho,
                'confianza': confianza,
                'score': score,
                'correcto': es_correcto
            })
            
            # Mostrar resultado
            estado = "✅" if es_correcto else "❌"
            print(f"{estado} [{i:2d}] Esperado: {esperado:8s} | Predicho: {predicho:8s} | Confianza: {confianza*100:5.1f}% | Score: {score:+.3f}")
            print(f"    Texto: {texto}")
            if not es_correcto:
                print(f"    ⚠️ ERROR: Se esperaba '{esperado}' pero se predijo '{predicho}'")
            print()
            
        except Exception as e:
            print(f"❌ Error al analizar: {texto}")
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
                print(f"   Texto: {r['texto']}")
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
    
    print("=" * 80)
    print("✅ PRUEBA COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    main()



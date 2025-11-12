# test_50_casos.py
"""
Script para probar los 50 casos del dataset (25 positivos, 25 negativos)
"""
import sys
import os
import io

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml_models.sentiment_nn import SentimentNeuralNetwork

def test_50_casos():
    print("=" * 80)
    print("PRUEBAS DE 50 CASOS (25 POSITIVOS, 25 NEGATIVOS)")
    print("=" * 80)
    print()
    
    # Cargar modelo
    print("[*] Cargando modelo...")
    model = SentimentNeuralNetwork()
    model.load_model()
    print("[OK] Modelo cargado\n")
    
    # Casos de prueba (50 casos: 25 positivos, 25 negativos)
    casos = [
        # POSITIVOS (1-25)
        {"texto": "El servicio fue excelente, volveré pronto", "esperado": "positivo"},
        {"texto": "La comida estaba deliciosa y el ambiente agradable", "esperado": "positivo"},
        {"texto": "La atención al cliente fue muy amable", "esperado": "positivo"},
        {"texto": "Excelente experiencia de compra", "esperado": "positivo"},
        {"texto": "El restaurante estaba limpio y acogedor", "esperado": "positivo"},
        {"texto": "Buena relación calidad-precio", "esperado": "positivo"},
        {"texto": "El empaque era bonito y seguro", "esperado": "positivo"},
        {"texto": "Estoy muy satisfecho con el servicio", "esperado": "positivo"},
        {"texto": "La app es fácil de usar y rápida", "esperado": "positivo"},
        {"texto": "Excelente calidad en todos los aspectos", "esperado": "positivo"},
        {"texto": "Atención rápida y eficiente", "esperado": "positivo"},
        {"texto": "Estoy muy feliz con mi compra", "esperado": "positivo"},
        {"texto": "Todo funcionó perfectamente", "esperado": "positivo"},
        {"texto": "Muy recomendable, todo perfecto", "esperado": "positivo"},
        {"texto": "Me encantó la atención personalizada", "esperado": "positivo"},
        {"texto": "Fácil proceso de compra y pago", "esperado": "positivo"},
        {"texto": "Excelente trato al cliente", "esperado": "positivo"},
        {"texto": "Me encantó el diseño del producto", "esperado": "positivo"},
        {"texto": "Muy buena experiencia general", "esperado": "positivo"},
        {"texto": "Recomiendo totalmente este servicio", "esperado": "positivo"},
        {"texto": "Gran calidad y atención", "esperado": "positivo"},
        {"texto": "Superó mis expectativas", "esperado": "positivo"},
        {"texto": "Muy satisfecho con el resultado", "esperado": "positivo"},
        {"texto": "Todo fue excelente", "esperado": "positivo"},
        {"texto": "Muy contento con mi compra", "esperado": "positivo"},
        
        # NEGATIVOS (26-50)
        {"texto": "Muy mala atención, no recomiendo este lugar", "esperado": "negativo"},
        {"texto": "El pedido llegó tarde y frío", "esperado": "negativo"},
        {"texto": "El producto llegó en mal estado", "esperado": "negativo"},
        {"texto": "Nunca volveré a comprar aquí", "esperado": "negativo"},
        {"texto": "El envío se demoró demasiado", "esperado": "negativo"},
        {"texto": "El personal fue grosero y poco atento", "esperado": "negativo"},
        {"texto": "El producto no cumplió con mis expectativas", "esperado": "negativo"},
        {"texto": "No funcionó como prometían", "esperado": "negativo"},
        {"texto": "La comida estaba fría y sin sabor", "esperado": "negativo"},
        {"texto": "No volveré a usar esta aplicación", "esperado": "negativo"},
        {"texto": "La experiencia fue decepcionante", "esperado": "negativo"},
        {"texto": "La entrega fue un desastre", "esperado": "negativo"},
        {"texto": "El producto tenía defectos visibles", "esperado": "negativo"},
        {"texto": "El servicio técnico nunca respondió", "esperado": "negativo"},
        {"texto": "El pedido llegó incompleto", "esperado": "negativo"},
        {"texto": "El sabor era horrible", "esperado": "negativo"},
        {"texto": "El precio no vale la calidad", "esperado": "negativo"},
        {"texto": "La página web estaba llena de errores", "esperado": "negativo"},
        {"texto": "La comida llegó con retraso", "esperado": "negativo"},
        {"texto": "No lo recomendaría a nadie", "esperado": "negativo"},
        {"texto": "El envío se perdió en el camino", "esperado": "negativo"},
        {"texto": "Una pésima experiencia", "esperado": "negativo"},
        {"texto": "No recibí lo que pedí", "esperado": "negativo"},
        {"texto": "Mala comunicación del soporte técnico", "esperado": "negativo"},
        {"texto": "El producto llegó roto", "esperado": "negativo"},
    ]
    
    # Verificar que tenemos 25 positivos y 25 negativos
    positivos = sum(1 for c in casos if c['esperado'] == 'positivo')
    negativos = sum(1 for c in casos if c['esperado'] == 'negativo')
    print(f"[*] Casos cargados: {len(casos)} totales")
    print(f"   - Positivos: {positivos}")
    print(f"   - Negativos: {negativos}")
    print()
    
    # Ejecutar pruebas
    resultados = []
    for i, caso in enumerate(casos, 1):
        try:
            resultado = model.predict_single(caso['texto'])
            sentimiento = resultado['sentiment']
            confianza = resultado['score']
            
            # Verificar si coincide con lo esperado
            coincide = sentimiento == caso['esperado']
            icono = "[OK]" if coincide else "[X]"
            
            if not coincide or i <= 5 or (i > 25 and i <= 30):  # Mostrar primeros 5 y errores
                print(f"{icono} [{i:2d}] {sentimiento:8s} (esperado: {caso['esperado']:8s}) - {confianza:.1%} - {caso['texto'][:60]}...")
            
            resultados.append({
                "numero": i,
                "texto": caso['texto'],
                "esperado": caso['esperado'],
                "obtenido": sentimiento,
                "confianza": confianza,
                "coincide": coincide
            })
        except Exception as e:
            print(f"[X] [{i:2d}] ERROR: {e}")
            resultados.append({
                "numero": i,
                "texto": caso['texto'],
                "esperado": caso['esperado'],
                "obtenido": "ERROR",
                "confianza": 0,
                "coincide": False
            })
    
    print()
    print("=" * 80)
    print("RESUMEN DETALLADO")
    print("=" * 80)
    
    # Estadísticas generales
    correctos = sum(1 for r in resultados if r['coincide'])
    total = len(resultados)
    porcentaje = (correctos / total) * 100 if total > 0 else 0
    
    # Estadísticas por categoría
    positivos_correctos = sum(1 for r in resultados if r['esperado'] == 'positivo' and r['coincide'])
    negativos_correctos = sum(1 for r in resultados if r['esperado'] == 'negativo' and r['coincide'])
    total_positivos = sum(1 for r in resultados if r['esperado'] == 'positivo')
    total_negativos = sum(1 for r in resultados if r['esperado'] == 'negativo')
    
    print(f"[OK] Correctos: {correctos}/{total} ({porcentaje:.1f}%)")
    print(f"   - Positivos: {positivos_correctos}/{total_positivos} ({positivos_correctos/total_positivos*100:.1f}%)")
    print(f"   - Negativos: {negativos_correctos}/{total_negativos} ({negativos_correctos/total_negativos*100:.1f}%)")
    print()
    
    # Mostrar errores
    errores = [r for r in resultados if not r['coincide']]
    if errores:
        print(f"[X] ERRORES ({len(errores)}):")
        for r in errores:
            print(f"   [{r['numero']:2d}] Esperado: {r['esperado']:8s}, Obtenido: {r['obtenido']:8s} ({r['confianza']:.1%})")
            print(f"       Texto: {r['texto']}")
        print()
    else:
        print("[OK] ¡Todos los casos fueron clasificados correctamente!")
        print()
    
    return correctos == total

if __name__ == "__main__":
    try:
        exito = test_50_casos()
        sys.exit(0 if exito else 1)
    except Exception as e:
        print(f"[X] Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


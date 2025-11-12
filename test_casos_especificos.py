# test_casos_especificos.py
"""
Script para probar los casos específicos reportados por el usuario
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml_models.sentiment_nn import SentimentNeuralNetwork

def test_casos():
    print("=" * 80)
    print("PRUEBAS DE CASOS ESPECÍFICOS")
    print("=" * 80)
    print()
    
    # Cargar modelo
    print("📦 Cargando modelo...")
    model = SentimentNeuralNetwork()
    model.load_model()
    print("✅ Modelo cargado\n")
    
    # Casos de prueba
    casos = [
        {
            "texto": "Fue una experiencia muy positiva. Me impresionó la rapidez con la que atendieron mi pedido, la amabilidad del personal y la calidad tan alta del servicio recibido.",
            "esperado": "positivo",
            "descripcion": "Texto explícitamente positivo"
        },
        {
            "texto": "El desempeño fue constante durante todo el proceso. Hubo buena comunicación en algunos puntos, pero también momentos de espera innecesarios. En general, fue una experiencia intermedia.",
            "esperado": "neutral",
            "descripcion": "Texto balanceado (positivo + negativo) = neutro"
        },
        {
            "texto": "Probé el servicio por primera vez y funcionó de manera adecuada. No tuve una impresión especialmente fuerte, pero considero que cumple con lo que se espera normalmente.",
            "esperado": "neutral",
            "descripcion": "Funcionó adecuadamente, cumple con lo esperado = neutro"
        },
        {
            "texto": "El producto cumple con lo que promete, aunque no ofrece nada fuera de lo común. Considero que es una opción adecuada para quien busca algo funcional y sencillo.",
            "esperado": "neutral",
            "descripcion": "Cumple con lo prometido pero no destacable = neutro"
        },
        {
            "texto": "El servicio se desarrolló de manera correcta. No tuve mayores inconvenientes, aunque tampoco hubo algo que destacara especialmente. Fue una experiencia promedio, sin sorpresas.",
            "esperado": "neutral",
            "descripcion": "Correcto pero sin destacar = neutro"
        },
        {
            "texto": "El resultado final fue correcto. No tengo grandes quejas ni elogios. Siento que cumplieron con lo acordado, aunque podrían agregar detalles que marquen una diferencia",
            "esperado": "neutral",
            "descripcion": "Correcto, sin quejas ni elogios = neutro"
        },
        {
            "texto": "Creo que podrían mejorar bastante. El proceso fue confuso, la información era poco clara y la atención al cliente no mostró la disposición necesaria para resolver los inconvenientes.",
            "esperado": "negativo",
            "descripcion": "Proceso confuso, información poco clara = negativo"
        },
        {
            "texto": "El servicio fue excelente, volveré pronto",
            "esperado": "positivo",
            "descripcion": "Excelente + volveré pronto = positivo"
        },
    ]
    
    # Ejecutar pruebas
    resultados = []
    for i, caso in enumerate(casos, 1):
        print(f"🧪 Prueba {i}/{len(casos)}: {caso['descripcion']}")
        print(f"   Texto: {caso['texto'][:80]}...")
        
        try:
            resultado = model.predict_single(caso['texto'])
            sentimiento = resultado['sentiment']
            confianza = resultado['score']
            
            # Verificar si coincide con lo esperado
            coincide = sentimiento == caso['esperado']
            icono = "✅" if coincide else "❌"
            
            print(f"   {icono} Resultado: {sentimiento} (esperado: {caso['esperado']}) - Confianza: {confianza:.2%}")
            
            resultados.append({
                "caso": caso['descripcion'],
                "esperado": caso['esperado'],
                "obtenido": sentimiento,
                "confianza": confianza,
                "coincide": coincide
            })
        except Exception as e:
            print(f"   ❌ Error: {e}")
            resultados.append({
                "caso": caso['descripcion'],
                "esperado": caso['esperado'],
                "obtenido": "ERROR",
                "confianza": 0,
                "coincide": False
            })
        
        print()
    
    # Resumen
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    correctos = sum(1 for r in resultados if r['coincide'])
    total = len(resultados)
    porcentaje = (correctos / total) * 100 if total > 0 else 0
    
    print(f"✅ Correctos: {correctos}/{total} ({porcentaje:.1f}%)")
    print()
    
    for r in resultados:
        icono = "✅" if r['coincide'] else "❌"
        print(f"{icono} {r['caso']}")
        if not r['coincide']:
            print(f"   Esperado: {r['esperado']}, Obtenido: {r['obtenido']} (confianza: {r['confianza']:.2%})")
    
    print()
    return correctos == total

if __name__ == "__main__":
    try:
        exito = test_casos()
        sys.exit(0 if exito else 1)
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


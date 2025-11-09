# train_model_local.py
"""
Script para entrenar el modelo localmente y guardar los archivos.
Ejecuta este script en tu computadora antes de subir a Render.
"""
import os
import sys

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml_models.sentiment_nn import SentimentNeuralNetwork

def main():
    print("=" * 60)
    print("🚀 ENTRENANDO MODELO LOCALMENTE")
    print("=" * 60)
    print()
    
    # Asegurar que el directorio existe
    model_dir = 'app/ml_models'
    os.makedirs(model_dir, exist_ok=True)
    
    print("📦 Creando modelo...")
    model = SentimentNeuralNetwork()
    
    print("🔄 Entrenando modelo (esto puede tomar 30-60 segundos)...")
    try:
        # Esto entrenará el modelo y lo guardará
        model.load_model()
        
        print()
        print("=" * 60)
        print("✅ MODELO ENTRENADO Y GUARDADO CORRECTAMENTE")
        print("=" * 60)
        print()
        print("📁 Archivos guardados en:")
        print(f"   - {model_dir}/sentiment_model.h5")
        print(f"   - {model_dir}/tokenizer.pkl")
        print(f"   - {model_dir}/label_encoder.pkl")
        print()
        print("📋 Próximos pasos:")
        print("   1. Verifica que los archivos se crearon correctamente")
        print("   2. Sube los archivos a GitHub Releases")
        print("   3. Las URLs se configurarán automáticamente en el código")
        print()
        
        # Verificar que los archivos existen
        files = [
            f'{model_dir}/sentiment_model.h5',
            f'{model_dir}/tokenizer.pkl',
            f'{model_dir}/label_encoder.pkl'
        ]
        
        all_exist = True
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024  # KB
                print(f"   ✅ {file} ({size:.1f} KB)")
            else:
                print(f"   ❌ {file} NO EXISTE")
                all_exist = False
        
        if all_exist:
            print()
            print("✅ Todos los archivos se crearon correctamente")
            print()
            print("📤 Ahora sube estos archivos a GitHub Releases:")
            print("   1. Ve a tu repositorio en GitHub")
            print("   2. Crea un nuevo Release (v1.0.0)")
            print("   3. Sube los 3 archivos")
            print("   4. Copia las URLs de descarga directa")
        else:
            print()
            print("⚠️ Algunos archivos no se crearon. Revisa los errores arriba.")
            
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERROR AL ENTRENAR MODELO")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


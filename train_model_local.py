# train_model_local.py
"""
Script para entrenar el modelo localmente y guardar los archivos.
Ejecuta este script en tu computadora antes de subir a Render.
"""
import os
import sys
import io

# Configurar encoding UTF-8 para Windows (antes de cualquier import o print)
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        # Si ya está configurado, no hacer nada
        pass

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml_models.sentiment_nn import SentimentNeuralNetwork

def main():
    
    print("=" * 60)
    print("ENTRENANDO MODELO LOCALMENTE")
    print("=" * 60)
    print()
    
    # Asegurar que el directorio existe
    model_dir = 'app/ml_models'
    os.makedirs(model_dir, exist_ok=True)
    
    print("📦 Creando modelo...")
    model = SentimentNeuralNetwork()
    
    print("🔄 Entrenando modelo (esto puede tomar 30-60 segundos)...")
    print("⚠️ NOTA: Este script entrena el modelo localmente y NO descarga desde GitHub")
    print()
    try:
        # Entrenar el modelo directamente usando el método interno
        # Esto NO intentará descargar, solo entrenará con los datos predefinidos
        model._create_pretrained_model()
        
        print()
        print("=" * 60)
        print("✅ MODELO ENTRENADO Y GUARDADO CORRECTAMENTE")
        print("=" * 60)
        print()
        print("📁 Archivos guardados en:")
        print(f"   - {model_dir}/sentiment_model.keras")
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
            f'{model_dir}/sentiment_model.keras',
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
            print()
            print("   📋 PASOS EN GITHUB:")
            print("   1. Ve a: https://github.com/crisncr/inteligenciaArtificial/releases")
            print("   2. Clic en 'Create a new release' (botón verde)")
            print("   3. Tag: v1.0.0 | Title: Modelo Pre-entrenado v1.0")
            print("   4. Arrastra los 3 archivos a 'Attach binaries'")
            print("   5. Clic en 'Publish release'")
            print("   6. Clic derecho en cada archivo → 'Copy link address'")
            print()
            print("   ✅ Las URLs ya están configuradas en el código por defecto")
            print("   📖 Ver GUIA_GITHUB_RELEASES.md para instrucciones detalladas")
            print("   📖 Ver PASOS_GITHUB_RELEASES_SIMPLE.md para pasos rápidos")
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


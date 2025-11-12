# test_batch_render.py
"""
Script para probar el análisis de sentimientos en Render (producción)
Usa el endpoint real con TensorFlow
"""
import sys
import os
import pandas as pd
import requests
import json

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

def login_and_get_token(server_url, email=None, password=None):
    """Hacer login y obtener token JWT"""
    if not email or not password:
        print("⚠️  No se proporcionaron credenciales. Usando token manual si está disponible.")
        token = os.getenv('API_TOKEN')
        if token:
            return token
        return None
    
    try:
        url = f"{server_url}/api/auth/login"
        response = requests.post(
            url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "username": email.lower().strip(),
                "password": password
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('access_token')
        else:
            print(f"❌ Error en login: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error al hacer login: {str(e)}")
        return None

def main():
    csv_path = r'c:\Users\crist\Downloads\opiniones_clientes.csv'
    
    # URL del servidor (cambiar por tu URL de Render)
    # Si está en local, usar: http://localhost:8000
    # Si está en Render, usar tu URL de Render
    SERVER_URL = os.getenv('API_URL', 'http://localhost:8000')
    
    print("=" * 80)
    print("PRUEBA DE ANÁLISIS DE SENTIMIENTOS CON TENSORFLOW (RENDER)")
    print("=" * 80)
    print(f"🌐 Servidor: {SERVER_URL}")
    print()
    
    # Intentar obtener token
    email = os.getenv('API_EMAIL')
    password = os.getenv('API_PASSWORD')
    token = os.getenv('API_TOKEN')
    
    if not token:
        if email and password:
            print("🔐 Obteniendo token de autenticación...")
            token = login_and_get_token(SERVER_URL, email, password)
            if token:
                print("✅ Token obtenido correctamente")
            else:
                print("❌ No se pudo obtener token. Usa variables de entorno:")
                print("   set API_EMAIL=tu_email@ejemplo.com")
                print("   set API_PASSWORD=tu_contraseña")
                print("   O usa: set API_TOKEN=tu_token_jwt")
                return
        else:
            print("⚠️  No hay token disponible. Usa variables de entorno:")
            print("   set API_EMAIL=tu_email@ejemplo.com")
            print("   set API_PASSWORD=tu_contraseña")
            print("   O usa: set API_TOKEN=tu_token_jwt")
            return
    else:
        print("✅ Usando token proporcionado")
    
    print()
    
    # Leer CSV
    print(f"📖 Leyendo archivo: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"✅ CSV leído correctamente")
        print(f"📊 Total de filas: {len(df)}")
        print(f"📊 Columnas: {list(df.columns)}")
        print()
        
        # Verificar distribución real
        if 'sentimiento' in df.columns:
            print("📊 DISTRIBUCIÓN REAL EN EL CSV:")
            print(df['sentimiento'].value_counts())
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
        
        # Preparar request para el endpoint
        print("=" * 80)
        print("ENVIANDO REQUEST AL ENDPOINT /api/datasets/analyze-batch")
        print("=" * 80)
        print()
        
        try:
            url = f"{SERVER_URL}/api/datasets/analyze-batch"
            payload = {"texts": texts}
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            # Primero "despertar" el servidor con una petición simple
            print("⏳ Despertando servidor (puede tardar 30-60 segundos si está dormido)...")
            try:
                wake_up = requests.get(f"{SERVER_URL}/api/auth/me", headers={"Authorization": f"Bearer {token}"}, timeout=60)
                print("✅ Servidor despierto")
            except:
                print("⚠️  Servidor puede estar despertándose...")
            
            print()
            print(f"🔄 Enviando {len(texts)} textos al servidor...")
            print(f"   URL: {url}")
            print("   ⏳ Esto puede tardar 1-2 minutos (procesamiento en lotes con TensorFlow)...")
            print()
            
            # Timeout más largo para permitir que el modelo se cargue y procese
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ RESPUESTA DEL SERVIDOR:")
                print("=" * 80)
                print(f"📊 Total analizado: {result.get('total', 0)}")
                print()
                
                summary = result.get('summary', {})
                print("📊 RESUMEN DE SENTIMIENTOS:")
                print(f"😊 Positivos: {summary.get('positive', 0)} ({summary.get('positive_percent', 0)}%)")
                print(f"😞 Negativos: {summary.get('negative', 0)} ({summary.get('negative_percent', 0)}%)")
                print(f"😐 Neutrales: {summary.get('neutral', 0)} ({summary.get('neutral_percent', 0)}%)")
                print()
                
                # Comparar con distribución real
                if 'sentimiento' in df.columns:
                    print("=" * 80)
                    print("COMPARACIÓN CON DISTRIBUCIÓN REAL:")
                    print("=" * 80)
                    real_pos = len(df[df['sentimiento'] == 'positiva'])
                    real_neg = len(df[df['sentimiento'] == 'negativa'])
                    pred_pos = summary.get('positive', 0)
                    pred_neg = summary.get('negative', 0)
                    
                    print(f"Real:  Positivos={real_pos}, Negativos={real_neg}")
                    print(f"Pred:  Positivos={pred_pos}, Negativos={pred_neg}")
                    print()
                    
                    # Calcular precisión aproximada
                    if len(texts) > 0:
                        # Asumir que los primeros son positivos y los últimos negativos
                        # (esto es una aproximación, el modelo real debería hacerlo mejor)
                        print("📈 PRECISIÓN APROXIMADA:")
                        print(f"   (Nota: Esta es una estimación, el modelo real usa ML)")
                
                # Mostrar algunos ejemplos
                results = result.get('results', [])
                if results:
                    print()
                    print("=" * 80)
                    print("EJEMPLOS DE RESULTADOS:")
                    print("=" * 80)
                    for i, r in enumerate(results[:10], 1):
                        sentiment = r.get('sentiment', 'unknown')
                        text = r.get('text', '')[:60]
                        confidence = r.get('confidence', 0)
                        emoji = r.get('emoji', '😐')
                        print(f"[{i:2d}] {emoji} {sentiment.upper():8s} | Conf: {confidence*100:5.1f}% | {text}...")
                
                print()
                print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
                
            elif response.status_code == 401:
                print("❌ ERROR: Se requiere autenticación")
                print("   Necesitas un token de acceso. Inicia sesión en la web y obtén el token.")
                print(f"   Response: {response.text}")
            elif response.status_code == 502:
                print("❌ ERROR 502: Timeout del servidor")
                print("   El procesamiento en lotes debería evitar esto, pero puede que el servidor esté sobrecargado.")
                print(f"   Response: {response.text[:200]}")
            else:
                print(f"❌ ERROR {response.status_code}: {response.text[:500]}")
                
        except requests.exceptions.ConnectionError:
            print("❌ ERROR: No se pudo conectar al servidor")
            print(f"   Verifica que el servidor esté corriendo en: {SERVER_URL}")
            print("   O cambia SERVER_URL a tu URL de Render")
        except requests.exceptions.Timeout:
            print("❌ ERROR: Timeout al esperar respuesta del servidor")
            print("   El servidor puede estar procesando. Intenta aumentar el timeout.")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print()
    print("💡 INSTRUCCIONES:")
    print("   1. Establece la URL del servidor:")
    print("      $env:API_URL='https://inteligenciaartificial-1-2ljl.onrender.com'")
    print("   2. Para autenticación, usa UNA de estas opciones:")
    print("      a) Email y contraseña:")
    print("         $env:API_EMAIL='tu_email@ejemplo.com'")
    print("         $env:API_PASSWORD='tu_contraseña'")
    print("      b) Token JWT directo (si ya lo tienes):")
    print("         $env:API_TOKEN='tu_token_jwt'")
    print("   3. Ejecuta el script:")
    print("      python test_batch_render.py")
    print()
    print("-" * 80)
    print()
    main()


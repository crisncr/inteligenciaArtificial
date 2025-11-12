# analizar_csv_render.py
"""
Script para analizar CSV usando la API de Render y mostrar textos clasificados
"""
import sys
import os
import pandas as pd
import requests
import json
import io
import time

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

def login_and_get_token(server_url, email, password):
    """Hacer login y obtener token JWT"""
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
    
    # Configuración
    SERVER_URL = os.getenv('API_URL', 'https://inteligenciaartificial-1-2ljl.onrender.com')
    EMAIL = os.getenv('API_EMAIL', 'cuevasn050@gmail.com')
    PASSWORD = os.getenv('API_PASSWORD', 'Axenoider2024.')
    
    print("=" * 80)
    print("ANÁLISIS DE SENTIMIENTOS - CLASIFICACIÓN (Render API)")
    print("=" * 80)
    print()
    print(f"📖 Archivo: {csv_path}")
    print(f"🌐 Servidor: {SERVER_URL}")
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
    print("🔐 Autenticando...")
    token = login_and_get_token(SERVER_URL, EMAIL, PASSWORD)
    if not token:
        print("❌ No se pudo obtener token de autenticación")
        return
    
    print("✅ Autenticación exitosa")
    print()
    
    # Despertar servidor
    print("⏳ Despertando servidor (puede tardar 30-60 segundos si está dormido)...")
    try:
        wake_up = requests.get(
            f"{SERVER_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        print("✅ Servidor despierto")
    except:
        print("⚠️  Servidor puede estar despertándose...")
    
    print()
    print("🔄 Enviando textos para análisis...")
    print(f"   Total: {len(texts)} textos")
    print("   ⏳ Procesando en lotes pequeños para evitar timeout...")
    print()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    url = f"{SERVER_URL}/api/datasets/analyze-batch"
    
    start_time = time.time()
    
    # Procesar en lotes pequeños para evitar timeout
    batch_size = 10
    all_results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"🔄 Procesando lote {batch_num}/{total_batches} ({len(batch)} textos)...")
            
            payload = {"texts": batch}
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=120)
                
                if response.status_code != 200:
                    print(f"   ❌ Error en lote {batch_num}: {response.status_code}")
                    print(f"   Respuesta: {response.text[:200]}")
                    # Agregar resultados de error para este lote
                    for text in batch:
                        all_results.append({
                            'text': text,
                            'sentiment': 'error',
                            'confidence': 0.0,
                            'emoji': '❌',
                            'error': f'Error {response.status_code}'
                        })
                    continue
                
                result = response.json()
                batch_results = result.get('results', [])
                all_results.extend(batch_results)
                
                print(f"   ✅ Lote {batch_num} completado: {len(batch_results)} resultados")
                
            except requests.exceptions.Timeout:
                print(f"   ❌ Timeout en lote {batch_num}")
                for text in batch:
                    all_results.append({
                        'text': text,
                        'sentiment': 'error',
                        'confidence': 0.0,
                        'emoji': '❌',
                        'error': 'Timeout'
                    })
            except Exception as e:
                print(f"   ❌ Error en lote {batch_num}: {str(e)[:100]}")
                for text in batch:
                    all_results.append({
                        'text': text,
                        'sentiment': 'error',
                        'confidence': 0.0,
                        'emoji': '❌',
                        'error': str(e)[:100]
                    })
        
        total_time = time.time() - start_time
        
        print()
        print(f"✅ Análisis completado en {total_time:.2f} segundos")
        print()
        
        # Usar resultados consolidados
        results = all_results
        
        # Separar por sentimiento
        positivos = [r for r in results if r.get('sentiment') == 'positivo']
        negativos = [r for r in results if r.get('sentiment') == 'negativo']
        neutrales = [r for r in results if r.get('sentiment') == 'neutral']
        errores = [r for r in results if r.get('sentiment') == 'error']
        
        # Mostrar resultados
        print("=" * 80)
        print("RESULTADOS - TEXTOS POSITIVOS")
        print("=" * 80)
        print(f"📊 Total: {len(positivos)} textos positivos")
        print()
        for i, r in enumerate(positivos, 1):
            conf = r.get('confidence', 0.0)
            emoji = r.get('emoji', '😐')
            text = r.get('text', '')
            print(f"[{i}] {emoji} (confianza: {conf:.3f}) - {text}")
        print()
        
        print("=" * 80)
        print("RESULTADOS - TEXTOS NEGATIVOS")
        print("=" * 80)
        print(f"📊 Total: {len(negativos)} textos negativos")
        print()
        for i, r in enumerate(negativos, 1):
            conf = r.get('confidence', 0.0)
            emoji = r.get('emoji', '😐')
            text = r.get('text', '')
            print(f"[{i}] {emoji} (confianza: {conf:.3f}) - {text}")
        print()
        
        if neutrales:
            print("=" * 80)
            print("RESULTADOS - TEXTOS NEUTRALES")
            print("=" * 80)
            print(f"📊 Total: {len(neutrales)} textos neutrales")
            print()
            for i, r in enumerate(neutrales, 1):
                conf = r.get('confidence', 0.0)
                emoji = r.get('emoji', '😐')
                text = r.get('text', '')
                print(f"[{i}] {emoji} (confianza: {conf:.3f}) - {text}")
            print()
        
        if errores:
            print("=" * 80)
            print("ERRORES")
            print("=" * 80)
            print(f"📊 Total: {len(errores)} textos con error")
            print()
            for i, r in enumerate(errores, 1):
                text = r.get('text', '')
                error_msg = r.get('error', 'Error desconocido')
                print(f"[{i}] ❌ - {text}")
                print(f"    Error: {error_msg[:100]}")
            print()
        
        # Resumen final
        print("=" * 80)
        print("RESUMEN FINAL")
        print("=" * 80)
        print(f"📊 Total analizados: {len(results)}")
        print(f"✅ Positivos: {len(positivos)} ({len(positivos)/len(results)*100:.1f}%)")
        print(f"❌ Negativos: {len(negativos)} ({len(negativos)/len(results)*100:.1f}%)")
        if neutrales:
            print(f"😐 Neutrales: {len(neutrales)} ({len(neutrales)/len(results)*100:.1f}%)")
        if errores:
            print(f"⚠️  Errores: {len(errores)} ({len(errores)/len(results)*100:.1f}%)")
        print()
        
        # Mostrar resumen de la API
        summary = result.get('summary', {})
        if summary:
            print("=" * 80)
            print("RESUMEN DE LA API")
            print("=" * 80)
            print(f"✅ Positivos: {summary.get('positive', 0)} ({summary.get('positive_percent', 0):.1f}%)")
            print(f"❌ Negativos: {summary.get('negative', 0)} ({summary.get('negative_percent', 0):.1f}%)")
            print(f"😐 Neutrales: {summary.get('neutral', 0)} ({summary.get('neutral_percent', 0):.1f}%)")
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
            if len(real_labels) == len(results):
                correctos = 0
                for i, (real, pred) in enumerate(zip(real_labels, results)):
                    real_sent = 'positivo' if 'positiv' in real.lower() else 'negativo'
                    pred_sent = pred.get('sentiment', '')
                    if real_sent == pred_sent:
                        correctos += 1
                
                precision = (correctos / len(results)) * 100
                print(f"✅ Precisión: {correctos}/{len(results)} = {precision:.1f}%")
                print()
        
        print("=" * 80)
        print("✅ ANÁLISIS COMPLETADO")
        print("=" * 80)
        
    except requests.exceptions.Timeout:
        print("❌ Error: Timeout - El servidor tardó demasiado en responder")
        print("   Intenta de nuevo o reduce el número de textos")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en la petición: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


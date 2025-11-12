# test_batch_small.py
"""
Script para probar con menos textos primero
"""
import sys
import os
import pandas as pd
import requests
import json

if sys.platform == 'win32':
    try:
        import io
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
            data={"username": email.lower().strip(), "password": password},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('access_token')
        return None
    except:
        return None

def main():
    csv_path = r'c:\Users\crist\Downloads\opiniones_clientes.csv'
    SERVER_URL = os.getenv('API_URL', 'https://inteligenciaartificial-1-2ljl.onrender.com')
    email = os.getenv('API_EMAIL', 'cuevasn050@gmail.com')
    password = os.getenv('API_PASSWORD', 'Axenoider2024.')
    
    print("=" * 80)
    print("PRUEBA CON MENOS TEXTOS (10 textos primero)")
    print("=" * 80)
    print()
    
    # Login
    print("🔐 Obteniendo token...")
    token = login_and_get_token(SERVER_URL, email, password)
    if not token:
        print("❌ Error en login")
        return
    print("✅ Token obtenido")
    print()
    
    # Leer CSV
    print(f"📖 Leyendo CSV...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    texts = df['opinion'].dropna().astype(str).tolist()[:10]  # Solo primeros 10
    
    print(f"📝 Probando con {len(texts)} textos")
    print()
    
    # Probar endpoint
    url = f"{SERVER_URL}/api/datasets/analyze-batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    print(f"🔄 Enviando request...")
    try:
        response = requests.post(
            url,
            json={"texts": texts},
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ÉXITO!")
            print(f"📊 Total: {result.get('total')}")
            summary = result.get('summary', {})
            print(f"😊 Positivos: {summary.get('positive')}")
            print(f"😞 Negativos: {summary.get('negative')}")
            print(f"😐 Neutrales: {summary.get('neutral')}")
        else:
            print(f"❌ Error {response.status_code}: {response.text[:200]}")
    except requests.exceptions.Timeout:
        print("❌ Timeout - El servidor tardó más de 60 segundos")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()







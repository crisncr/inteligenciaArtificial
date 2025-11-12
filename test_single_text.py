# test_single_text.py - Probar con UN solo texto
import requests
import os

SERVER_URL = "https://inteligenciaartificial-1-2ljl.onrender.com"
email = "cuevasn050@gmail.com"
password = "Axenoider2024."

# Login
print("🔐 Login...")
login_resp = requests.post(
    f"{SERVER_URL}/api/auth/login",
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    data={"username": email, "password": password},
    timeout=30
)
if login_resp.status_code != 200:
    print(f"❌ Login falló: {login_resp.status_code}")
    exit(1)

token = login_resp.json().get('access_token')
print("✅ Login OK")
print()

# Probar con 1 texto
print("🔄 Probando con 1 texto...")
url = f"{SERVER_URL}/api/datasets/analyze-batch"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

try:
    response = requests.post(
        url,
        json={"texts": ["El servicio fue excelente, volveré pronto"]},
        headers=headers,
        timeout=60
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("✅ ÉXITO!")
        print(f"Resultado: {result}")
    else:
        print(f"❌ Error: {response.text[:500]}")
except Exception as e:
    print(f"❌ Exception: {str(e)}")





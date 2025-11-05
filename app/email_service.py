"""
Servicio de envío de emails
Para producción, integrar con servicios como SendGrid, AWS SES, o SMTP
"""
import os
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# Template simple sin jinja2

# Configuración SMTP (para desarrollo)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

async def send_password_reset_email(email: str, reset_token: str):
    """
    Envía email para recuperación de contraseña
    """
    reset_url = f"{FRONTEND_URL}/reset-password?token={reset_token}"
    
    # Template del email
    html_template = """
    <html>
    <body>
        <h2>Recuperación de Contraseña</h2>
        <p>Has solicitado recuperar tu contraseña. Haz clic en el siguiente enlace:</p>
        <p><a href="{{ reset_url }}">Recuperar Contraseña</a></p>
        <p>O copia y pega este enlace en tu navegador:</p>
        <p>{{ reset_url }}</p>
        <p>Este enlace expirará en 1 hora.</p>
        <p>Si no solicitaste este cambio, ignora este email.</p>
    </body>
    </html>
    """
    
    text_template = """
    Recuperación de Contraseña
    
    Has solicitado recuperar tu contraseña. Visita el siguiente enlace:
    {{ reset_url }}
    
    Este enlace expirará en 1 hora.
    Si no solicitaste este cambio, ignora este email.
    """
    
    # Renderizar templates (simple string replacement)
    html_content = html_template.replace("{{ reset_url }}", reset_url)
    text_content = text_template.replace("{{ reset_url }}", reset_url)
    
    # Crear mensaje
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Recuperación de Contraseña - Sentimetría"
    msg["From"] = SMTP_USER
    msg["To"] = email
    
    # Agregar contenido
    part1 = MIMEText(text_content, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)
    
    # Enviar email
    if SMTP_USER and SMTP_PASSWORD:
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Error enviando email: {e}")
            # En desarrollo, solo imprimir el token
            print(f"Token de reset (solo desarrollo): {reset_token}")
            return False
    else:
        # En desarrollo sin SMTP configurado, solo imprimir
        print(f"Token de reset (solo desarrollo): {reset_token}")
        print(f"URL de reset: {reset_url}")
        return False



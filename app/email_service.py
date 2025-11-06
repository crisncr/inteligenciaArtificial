"""
Servicio de envío de emails
Soporta SMTP (Gmail) y SendGrid como alternativa
"""
import os
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx
# Template simple sin jinja2

# Configuración SMTP (para desarrollo)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Configuración SendGrid (alternativa que funciona en Render)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", SMTP_USER or "noreply@sentimetria.com")

async def send_email_via_sendgrid(to_email: str, subject: str, html_content: str, text_content: str) -> bool:
    """
    Envía email usando SendGrid API (funciona en Render)
    """
    if not SENDGRID_API_KEY:
        raise ValueError("SENDGRID_API_KEY no configurado")
    
    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "personalizations": [{
            "to": [{"email": to_email}]
        }],
        "from": {"email": SENDGRID_FROM_EMAIL},
        "subject": subject,
        "content": [
            {
                "type": "text/plain",
                "value": text_content
            },
            {
                "type": "text/html",
                "value": html_content
            }
        ]
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 202:
                print(f"✅ Email enviado con SendGrid a {to_email}")
                return True
            else:
                error_text = response.text
                print(f"❌ Error SendGrid ({response.status_code}): {error_text}")
                raise Exception(f"SendGrid error: {response.status_code} - {error_text}")
    except Exception as e:
        print(f"❌ Error enviando email con SendGrid: {e}")
        raise

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
    
    # Intentar enviar con SendGrid primero (funciona en Render)
    if SENDGRID_API_KEY:
        try:
            return await send_email_via_sendgrid(
                to_email=email,
                subject="Recuperación de Contraseña - Sentimetría",
                html_content=html_content,
                text_content=text_content
            )
        except Exception as e:
            print(f"⚠️ Error con SendGrid: {e}, intentando SMTP...")
    
    # Enviar email con SMTP (fallback)
    if SMTP_USER and SMTP_PASSWORD:
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            print(f"✅ Email de recuperación enviado a {email}")
            return True
        except Exception as e:
            print(f"❌ Error enviando email de recuperación (SMTP): {e}")
            print(f"💡 Tip: Render bloquea SMTP en plan gratuito. Usa SendGrid (gratis 100 emails/día)")
            print(f"Token de reset (solo desarrollo): {reset_token}")
            return False
    else:
        print(f"⚠️ Email de recuperación (SMTP no configurado): Token de reset: {reset_token}")
        return False

async def send_welcome_email(email: str, name: str):
    """
    Envía email de bienvenida cuando se registra un nuevo usuario
    """
    login_url = f"{FRONTEND_URL}"
    
    # Template del email
    html_template = """
    <html>
    <body>
        <h2>¡Bienvenido a Sentimetría, {{ name }}!</h2>
        <p>Gracias por registrarte en nuestra plataforma de análisis de sentimientos.</p>
        <p>Tu cuenta ha sido creada exitosamente con el plan gratuito.</p>
        <p>Con tu cuenta puedes:</p>
        <ul>
            <li>Analizar sentimientos en textos</li>
            <li>Ver historial de tus análisis</li>
            <li>Acceder a estadísticas detalladas</li>
            <li>Y mucho más...</li>
        </ul>
        <p><a href="{{ login_url }}">Inicia sesión ahora</a> para comenzar a usar la plataforma.</p>
        <p>Si tienes alguna pregunta, no dudes en contactarnos.</p>
        <p>¡Te deseamos mucho éxito!</p>
        <p>El equipo de Sentimetría</p>
    </body>
    </html>
    """
    
    text_template = """
    ¡Bienvenido a Sentimetría, {{ name }}!
    
    Gracias por registrarte en nuestra plataforma de análisis de sentimientos.
    Tu cuenta ha sido creada exitosamente con el plan gratuito.
    
    Con tu cuenta puedes:
    - Analizar sentimientos en textos
    - Ver historial de tus análisis
    - Acceder a estadísticas detalladas
    - Y mucho más...
    
    Visita {{ login_url }} para iniciar sesión y comenzar a usar la plataforma.
    
    Si tienes alguna pregunta, no dudes en contactarnos.
    ¡Te deseamos mucho éxito!
    
    El equipo de Sentimetría
    """
    
    # Renderizar templates
    html_content = html_template.replace("{{ name }}", name).replace("{{ login_url }}", login_url)
    text_content = text_template.replace("{{ name }}", name).replace("{{ login_url }}", login_url)
    
    # Crear mensaje
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "¡Bienvenido a Sentimetría!"
    msg["From"] = SMTP_USER
    msg["To"] = email
    
    # Agregar contenido
    part1 = MIMEText(text_content, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)
    
    # Intentar enviar con SendGrid primero (funciona en Render)
    if SENDGRID_API_KEY:
        try:
            return await send_email_via_sendgrid(
                to_email=email,
                subject="¡Bienvenido a Sentimetría!",
                html_content=html_content,
                text_content=text_content
            )
        except Exception as e:
            print(f"⚠️ Error con SendGrid: {e}, intentando SMTP...")
    
    # Enviar email con SMTP (fallback)
    if SMTP_USER and SMTP_PASSWORD:
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            print(f"✅ Email de bienvenida enviado a {email}")
            return True
        except Exception as e:
            print(f"❌ Error enviando email de bienvenida (SMTP): {e}")
            print(f"💡 Tip: Render bloquea SMTP en plan gratuito. Usa SendGrid (gratis 100 emails/día)")
            return False
    else:
        print(f"⚠️ Email de bienvenida (SMTP no configurado): {name} - {email}")
        return False

async def send_password_reset_success_email(email: str, name: str):
    """
    Envía email de confirmación cuando se restablece la contraseña exitosamente
    """
    login_url = f"{FRONTEND_URL}"
    
    # Template del email
    html_template = """
    <html>
    <body>
        <h2>Contraseña Restablecida Exitosamente</h2>
        <p>Hola {{ name }},</p>
        <p>Tu contraseña ha sido restablecida exitosamente.</p>
        <p>Si no fuiste tú quien realizó este cambio, por favor contacta con nosotros inmediatamente.</p>
        <p><strong>Recomendaciones de seguridad:</strong></p>
        <ul>
            <li>Usa una contraseña fuerte y única</li>
            <li>No compartas tu contraseña con nadie</li>
            <li>Si sospechas actividad sospechosa, cambia tu contraseña inmediatamente</li>
        </ul>
        <p><a href="{{ login_url }}">Inicia sesión ahora</a> con tu nueva contraseña.</p>
        <p>El equipo de Sentimetría</p>
    </body>
    </html>
    """
    
    text_template = """
    Contraseña Restablecida Exitosamente
    
    Hola {{ name }},
    
    Tu contraseña ha sido restablecida exitosamente.
    
    Si no fuiste tú quien realizó este cambio, por favor contacta con nosotros inmediatamente.
    
    Recomendaciones de seguridad:
    - Usa una contraseña fuerte y única
    - No compartas tu contraseña con nadie
    - Si sospechas actividad sospechosa, cambia tu contraseña inmediatamente
    
    Visita {{ login_url }} para iniciar sesión con tu nueva contraseña.
    
    El equipo de Sentimetría
    """
    
    # Renderizar templates
    html_content = html_template.replace("{{ name }}", name).replace("{{ login_url }}", login_url)
    text_content = text_template.replace("{{ name }}", name).replace("{{ login_url }}", login_url)
    
    # Crear mensaje
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Contraseña Restablecida - Sentimetría"
    msg["From"] = SMTP_USER
    msg["To"] = email
    
    # Agregar contenido
    part1 = MIMEText(text_content, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)
    
    # Intentar enviar con SendGrid primero (funciona en Render)
    if SENDGRID_API_KEY:
        try:
            return await send_email_via_sendgrid(
                to_email=email,
                subject="Contraseña Restablecida - Sentimetría",
                html_content=html_content,
                text_content=text_content
            )
        except Exception as e:
            print(f"⚠️ Error con SendGrid: {e}, intentando SMTP...")
    
    # Enviar email con SMTP (fallback)
    if SMTP_USER and SMTP_PASSWORD:
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            print(f"✅ Email de confirmación de restablecimiento enviado a {email}")
            return True
        except Exception as e:
            print(f"❌ Error enviando email de confirmación (SMTP): {e}")
            print(f"💡 Tip: Render bloquea SMTP en plan gratuito. Usa SendGrid (gratis 100 emails/día)")
            return False
    else:
        print(f"⚠️ Email de confirmación (SMTP no configurado): {name} - {email}")
        return False



"""
Servicio de envío de emails
Soporta SMTP (Gmail) y SendGrid como alternativa
"""
import os
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import quote
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

async def send_verification_email(email: str, name: str, verification_token: str):
    """
    Envía email de verificación cuando se registra un nuevo usuario
    """
    # Codificar el token para URL seguro
    encoded_token = quote(verification_token, safe='')
    verification_url = f"{FRONTEND_URL}?token={encoded_token}"
    login_url = f"{FRONTEND_URL}"
    
    # Template del email mejorado con diseño profesional
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #f4f4f4; }
            .container { max-width: 600px; margin: 0 auto; background-color: #ffffff; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; }
            .header h1 { color: #ffffff; margin: 0; font-size: 28px; font-weight: 600; }
            .content { padding: 40px 30px; }
            .greeting { font-size: 20px; color: #667eea; margin-bottom: 20px; font-weight: 600; }
            .message { color: #555; margin-bottom: 30px; font-size: 16px; }
            .button-container { text-align: center; margin: 30px 0; }
            .button { display: inline-block; padding: 14px 35px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
            .button:hover { box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); }
            .features { background-color: #f8f9fa; padding: 25px; border-radius: 8px; margin: 25px 0; }
            .features h3 { color: #667eea; margin-top: 0; font-size: 18px; }
            .features ul { margin: 15px 0; padding-left: 20px; }
            .features li { margin: 10px 0; color: #555; }
            .footer { padding: 30px; text-align: center; background-color: #f8f9fa; color: #777; font-size: 14px; }
            .link { color: #667eea; text-decoration: none; word-break: break-all; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎉 ¡Bienvenido a Sentimetría!</h1>
            </div>
            <div class="content">
                <div class="greeting">Hola {{ name }},</div>
                <div class="message">
                    Gracias por registrarte en nuestra plataforma de análisis de sentimientos. 
                    Estamos emocionados de tenerte con nosotros.
                </div>
                <div class="message">
                    Para completar tu registro y activar tu cuenta, por favor verifica tu dirección de correo electrónico haciendo clic en el botón de abajo:
                </div>
                <div class="button-container">
                    <a href="{{ verification_url }}" class="button" style="text-decoration: none;">Verificar mi Email</a>
                </div>
                <div class="features">
                    <h3>✨ Con tu cuenta verificada podrás:</h3>
                    <ul>
                        <li>Analizar sentimientos en textos de forma ilimitada</li>
                        <li>Ver historial completo de tus análisis</li>
                        <li>Acceder a estadísticas detalladas y gráficos</li>
                        <li>Exportar tus resultados y análisis</li>
                        <li>Y mucho más...</li>
                    </ul>
                </div>
                <div class="message" style="font-size: 14px; color: #999; margin-top: 30px;">
                    <strong>Nota:</strong> Este enlace de verificación expirará en 24 horas. 
                    Si no solicitaste este registro, puedes ignorar este correo.
                </div>
            </div>
            <div class="footer">
                <p><strong>Sentimetría</strong> - Análisis de Sentimientos Inteligente</p>
                <p>Si tienes alguna pregunta, no dudes en contactarnos.</p>
                <p style="margin-top: 20px;">
                    <a href="{{ login_url }}" style="color: #667eea; text-decoration: none;">Iniciar Sesión</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_template = """
    ¡Bienvenido a Sentimetría, {{ name }}!
    
    Gracias por registrarte en nuestra plataforma de análisis de sentimientos.
    
    Para completar tu registro y activar tu cuenta, por favor verifica tu dirección de correo electrónico visitando el siguiente enlace:
    
    {{ verification_url }}
    
    Con tu cuenta verificada podrás:
    - Analizar sentimientos en textos de forma ilimitada
    - Ver historial completo de tus análisis
    - Acceder a estadísticas detalladas y gráficos
    - Exportar tus resultados y análisis
    - Y mucho más...
    
    Nota: Este enlace de verificación expirará en 24 horas. Si no solicitaste este registro, puedes ignorar este correo.
    
    Si tienes alguna pregunta, no dudes en contactarnos.
    
    Sentimetría - Análisis de Sentimientos Inteligente
    """
    
    # Renderizar templates
    html_content = html_template.replace("{{ name }}", name).replace("{{ verification_url }}", verification_url).replace("{{ login_url }}", login_url)
    text_content = text_template.replace("{{ name }}", name).replace("{{ verification_url }}", verification_url).replace("{{ login_url }}", login_url)
    
    # Intentar enviar con SendGrid primero (funciona en Render)
    if SENDGRID_API_KEY:
        try:
            return await send_email_via_sendgrid(
                to_email=email,
                subject="Verifica tu cuenta - Sentimetría",
                html_content=html_content,
                text_content=text_content
            )
        except Exception as e:
            print(f"⚠️ Error con SendGrid: {e}, intentando SMTP...")
    
    # Crear mensaje para SMTP
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Verifica tu cuenta - Sentimetría"
    msg["From"] = SENDGRID_FROM_EMAIL or SMTP_USER
    msg["To"] = email
    
    # Agregar contenido
    part1 = MIMEText(text_content, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)
    
    # Enviar email con SMTP (fallback)
    if SMTP_USER and SMTP_PASSWORD:
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            print(f"✅ Email de verificación enviado a {email}")
            return True
        except Exception as e:
            print(f"❌ Error enviando email de verificación (SMTP): {e}")
            print(f"💡 Tip: Render bloquea SMTP en plan gratuito. Usa SendGrid (gratis 100 emails/día)")
            return False
    else:
        print(f"⚠️ Email de verificación (SMTP no configurado): {name} - {email}")
        return False

async def send_welcome_email(email: str, name: str):
    """
    Envía email de bienvenida después de verificar el email
    """
    login_url = f"{FRONTEND_URL}"
    
    # Template del email mejorado con diseño profesional
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #f4f4f4; }
            .container { max-width: 600px; margin: 0 auto; background-color: #ffffff; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; }
            .header h1 { color: #ffffff; margin: 0; font-size: 28px; font-weight: 600; }
            .content { padding: 40px 30px; }
            .greeting { font-size: 20px; color: #667eea; margin-bottom: 20px; font-weight: 600; }
            .message { color: #555; margin-bottom: 30px; font-size: 16px; }
            .button-container { text-align: center; margin: 30px 0; }
            .button { display: inline-block; padding: 14px 35px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
            .button:hover { box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); }
            .features { background-color: #f8f9fa; padding: 25px; border-radius: 8px; margin: 25px 0; }
            .features h3 { color: #667eea; margin-top: 0; font-size: 18px; }
            .features ul { margin: 15px 0; padding-left: 20px; }
            .features li { margin: 10px 0; color: #555; }
            .footer { padding: 30px; text-align: center; background-color: #f8f9fa; color: #777; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>✅ Cuenta Verificada</h1>
            </div>
            <div class="content">
                <div class="greeting">¡Hola {{ name }}!</div>
                <div class="message">
                    Tu cuenta ha sido verificada exitosamente. ¡Ya puedes comenzar a usar Sentimetría!
                </div>
                <div class="button-container">
                    <a href="{{ login_url }}" class="button">Iniciar Sesión</a>
                </div>
                <div class="features">
                    <h3>✨ Con tu cuenta puedes:</h3>
                    <ul>
                        <li>Analizar sentimientos en textos de forma ilimitada</li>
                        <li>Ver historial completo de tus análisis</li>
                        <li>Acceder a estadísticas detalladas y gráficos</li>
                        <li>Exportar tus resultados y análisis</li>
                        <li>Y mucho más...</li>
                    </ul>
                </div>
                <div class="message" style="color: #667eea; font-weight: 600;">
                    ¡Te deseamos mucho éxito en tu análisis de sentimientos!
                </div>
            </div>
            <div class="footer">
                <p><strong>Sentimetría</strong> - Análisis de Sentimientos Inteligente</p>
                <p>Si tienes alguna pregunta, no dudes en contactarnos.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_template = """
    ¡Hola {{ name }}!
    
    Tu cuenta ha sido verificada exitosamente. ¡Ya puedes comenzar a usar Sentimetría!
    
    Visita {{ login_url }} para iniciar sesión.
    
    Con tu cuenta puedes:
    - Analizar sentimientos en textos de forma ilimitada
    - Ver historial completo de tus análisis
    - Acceder a estadísticas detalladas y gráficos
    - Exportar tus resultados y análisis
    - Y mucho más...
    
    ¡Te deseamos mucho éxito en tu análisis de sentimientos!
    
    Sentimetría - Análisis de Sentimientos Inteligente
    """
    
    # Renderizar templates
    html_content = html_template.replace("{{ name }}", name).replace("{{ login_url }}", login_url)
    text_content = text_template.replace("{{ name }}", name).replace("{{ login_url }}", login_url)
    
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
    
    # Crear mensaje para SMTP
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "¡Bienvenido a Sentimetría!"
    msg["From"] = SENDGRID_FROM_EMAIL or SMTP_USER
    msg["To"] = email
    
    # Agregar contenido
    part1 = MIMEText(text_content, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)
    
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



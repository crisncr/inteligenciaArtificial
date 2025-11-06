import { useState } from 'react'
import { authAPI } from '../utils/api'

function VerifyEmail({ token, onClose }) {
  const [status, setStatus] = useState('pending') // pending, loading, success, error
  const [message, setMessage] = useState('')

  const handleVerify = async () => {
    if (!token) {
        setStatus('error')
        setMessage('No se proporcionó token de verificación')
        // Cerrar el componente antes de redirigir
        if (onClose) {
          setTimeout(() => {
            onClose()
          }, 100)
        }
        // Redirigir después de 2 segundos si hay error
        setTimeout(() => {
          window.location.replace('/')
        }, 2000)
        return
      }

    setStatus('loading')

    try {
      // Intentar primero con GET (para enlaces en email)
      const API_URL = import.meta.env.PROD ? '' : 'http://127.0.0.1:8000'
      const response = await fetch(`${API_URL}/api/auth/verify-email?token=${token}`)
      const result = await response.json()
      
      if (result.success) {
        setStatus('success')
        setMessage(result.message || 'Email verificado correctamente. ¡Bienvenido a Sentimetría!')
        // Cerrar el componente antes de redirigir
        if (onClose) {
          setTimeout(() => {
            onClose()
          }, 100)
        }
        // Redirigir automáticamente después de 2 segundos
        setTimeout(() => {
          window.location.replace('/')
        }, 2000)
      } else {
        setStatus('error')
        setMessage(result.message || 'Error al verificar el email')
        // Cerrar el componente antes de redirigir
        if (onClose) {
          setTimeout(() => {
            onClose()
          }, 100)
        }
        // Redirigir después de 3 segundos si hay error
        setTimeout(() => {
          window.location.replace('/')
        }, 3000)
      }
    } catch (err) {
        setStatus('error')
        setMessage(err.message || 'Error al verificar el email')
        // Cerrar el componente antes de redirigir
        if (onClose) {
          setTimeout(() => {
            onClose()
          }, 100)
        }
        // Redirigir después de 3 segundos si hay error
        setTimeout(() => {
          window.location.replace('/')
        }, 3000)
      }
  }

  return (
    <div className="verify-email-container" style={{ 
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px',
      zIndex: 3000
    }}>
      <div className="verify-email-content" style={{
        background: 'white',
        borderRadius: '12px',
        padding: '40px',
        maxWidth: '500px',
        width: '100%',
        boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
        textAlign: 'center'
      }}>
        {status === 'pending' && (
          <>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>📧</div>
            <h2 style={{ color: '#667eea', marginBottom: '20px' }}>Verificar tu Email</h2>
            <p style={{ marginBottom: '30px', color: '#555' }}>
              Haz clic en el botón para verificar tu dirección de correo electrónico y activar tu cuenta.
            </p>
            <button 
              className="btn" 
              onClick={handleVerify}
              style={{ padding: '14px 35px', fontSize: '16px', fontWeight: '600' }}
            >
              Verificar mi Email
            </button>
          </>
        )}

        {status === 'loading' && (
          <>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>⏳</div>
            <h2>Verificando tu email...</h2>
            <p>Por favor espera un momento.</p>
          </>
        )}
        
        {status === 'success' && (
          <>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>✅</div>
            <h2 style={{ color: '#667eea', marginBottom: '20px' }}>¡Email Verificado!</h2>
            <div style={{ 
              background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
              border: '2px solid #667eea',
              borderRadius: '12px',
              padding: '20px',
              marginBottom: '20px'
            }}>
              <p style={{ marginBottom: '10px', color: '#555', fontSize: '1.1rem', fontWeight: '500' }}>
                {message}
              </p>
              <p style={{ marginBottom: '0', color: '#667eea', fontSize: '0.95rem', fontWeight: '600' }}>
                Ya puedes iniciar sesión y comenzar a usar Sentimetría
              </p>
            </div>
            <p style={{ fontSize: '0.9rem', color: '#888', marginBottom: '0' }}>
              Redirigiendo a la página principal...
            </p>
          </>
        )}
        
        {status === 'error' && (
          <>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>❌</div>
            <h2 style={{ color: '#e74c3c', marginBottom: '20px' }}>Error de Verificación</h2>
            <p style={{ marginBottom: '20px', color: '#555' }}>{message}</p>
            <p style={{ fontSize: '0.9rem', color: '#888', marginBottom: '0' }}>
              Redirigiendo a la página principal...
            </p>
          </>
        )}
      </div>
    </div>
  )
}

export default VerifyEmail


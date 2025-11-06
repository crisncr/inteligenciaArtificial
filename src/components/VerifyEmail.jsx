import { useState } from 'react'
import { authAPI } from '../utils/api'

function VerifyEmail({ token, onClose }) {
  const [status, setStatus] = useState('pending') // pending, loading, success, error
  const [message, setMessage] = useState('')

  const handleVerify = async () => {
    if (!token) {
      setStatus('error')
      setMessage('No se proporcionó token de verificación')
      // Redirigir después de 2 segundos si hay error
      setTimeout(() => {
        window.location.href = '/'
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
        setMessage(result.message || 'Email verificado correctamente')
        // Redirigir automáticamente después de 2 segundos
        setTimeout(() => {
          window.location.href = '/'
        }, 2000)
      } else {
        setStatus('error')
        setMessage(result.message || 'Error al verificar el email')
        // Redirigir después de 3 segundos si hay error
        setTimeout(() => {
          window.location.href = '/'
        }, 3000)
      }
    } catch (err) {
      setStatus('error')
      setMessage(err.message || 'Error al verificar el email')
      // Redirigir después de 3 segundos si hay error
      setTimeout(() => {
        window.location.href = '/'
      }, 3000)
    }
  }

  return (
    <div className="verify-email-container" style={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
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
            <p style={{ marginBottom: '20px', color: '#555' }}>{message}</p>
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


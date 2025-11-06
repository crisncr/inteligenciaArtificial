import { useState, useEffect } from 'react'
import { authAPI } from '../utils/api'

function VerifyEmail({ token, onClose }) {
  const [status, setStatus] = useState('loading') // loading, success, error
  const [message, setMessage] = useState('')

  useEffect(() => {
    const verifyEmail = async () => {
      if (!token) {
        setStatus('error')
        setMessage('No se proporcionó token de verificación')
        return
      }

      try {
        // Intentar primero con GET (para enlaces en email)
        const API_URL = import.meta.env.PROD ? '' : 'http://127.0.0.1:8000'
        const response = await fetch(`${API_URL}/api/auth/verify-email?token=${token}`)
        const result = await response.json()
        
        if (result.success) {
          setStatus('success')
          setMessage(result.message || 'Email verificado correctamente')
        } else {
          setStatus('error')
          setMessage(result.message || 'Error al verificar el email')
        }
      } catch (err) {
        setStatus('error')
        setMessage(err.message || 'Error al verificar el email')
      }
    }

    verifyEmail()
  }, [token])

  return (
    <div style={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '40px',
        maxWidth: '500px',
        width: '100%',
        boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
        textAlign: 'center'
      }}>
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
            <p style={{ marginBottom: '30px', color: '#555' }}>{message}</p>
            <button 
              className="btn" 
              onClick={() => {
                if (onClose) onClose()
                window.location.href = '/'
              }}
              style={{ padding: '12px 30px', fontSize: '16px' }}
            >
              Ir a Iniciar Sesión
            </button>
          </>
        )}
        
        {status === 'error' && (
          <>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>❌</div>
            <h2 style={{ color: '#e74c3c', marginBottom: '20px' }}>Error de Verificación</h2>
            <p style={{ marginBottom: '30px', color: '#555' }}>{message}</p>
            <button 
              className="btn" 
              onClick={() => {
                if (onClose) onClose()
                window.location.href = '/'
              }}
              style={{ padding: '12px 30px', fontSize: '16px' }}
            >
              Volver al Inicio
            </button>
          </>
        )}
      </div>
    </div>
  )
}

export default VerifyEmail


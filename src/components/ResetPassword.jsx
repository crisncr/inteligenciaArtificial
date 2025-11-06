import { useState, useEffect } from 'react'
import { authAPI } from '../utils/api'

function ResetPassword({ token: tokenProp, onSuccess, onClose }) {
  const [token, setToken] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Obtener token de la URL o de props
    if (tokenProp) {
      setToken(tokenProp)
    } else {
      // Intentar obtener de la URL
      const urlParams = new URLSearchParams(window.location.search)
      const tokenParam = urlParams.get('token')
      if (tokenParam) {
        setToken(tokenParam)
      }
    }
  }, [tokenProp])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (newPassword !== confirmPassword) {
      setError('Las contraseñas no coinciden')
      return
    }

    // Validación de contraseña: primera letra mayúscula
    if (newPassword.length < 8) {
      setError('La contraseña debe tener al menos 8 caracteres')
      return
    }

    if (!newPassword[0] || !newPassword[0].match(/[A-Z]/)) {
      setError('La contraseña debe comenzar con una letra mayúscula')
      return
    }

    if (!newPassword.match(/[a-z]/)) {
      setError('La contraseña debe contener al menos una letra minúscula')
      return
    }

    if (!newPassword.match(/[0-9]/)) {
      setError('La contraseña debe contener al menos un número')
      return
    }

    if (!token) {
      setError('Token no válido')
      return
    }

    setLoading(true)

    try {
      await authAPI.resetPassword(token, newPassword)
      setSuccess(true)
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
    } catch (err) {
      setError(err.message || 'Error al restablecer contraseña')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  if (!token) {
    return (
      <div className="auth-modal">
        <div className="auth-content">
          <button className="auth-close" onClick={onClose}>×</button>
          <h2>Token Inválido</h2>
          <p className="auth-subtitle">El enlace de recuperación no es válido o ha expirado.</p>
          <button className="btn" onClick={onClose}>Cerrar</button>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-modal">
      <div className="auth-content">
        <button className="auth-close" onClick={onClose}>×</button>
        <h2>Restablecer Contraseña</h2>
        <p className="auth-subtitle">Ingresa tu nueva contraseña</p>

        {error && <div className="auth-error">{error}</div>}
        {success && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
            border: '2px solid #667eea',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '20px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '15px' }}>✅</div>
            <h3 style={{ color: '#667eea', marginBottom: '10px', marginTop: '0' }}>¡Contraseña Actualizada!</h3>
            <p style={{ marginBottom: '10px', color: '#555', fontSize: '1.1rem', fontWeight: '500' }}>
              Tu contraseña ha sido restablecida exitosamente.
            </p>
            <p style={{ marginBottom: '0', color: '#667eea', fontSize: '0.95rem', fontWeight: '600' }}>
              Ya puedes iniciar sesión con tu nueva contraseña
            </p>
            <p style={{ fontSize: '0.9rem', color: '#888', marginTop: '15px', marginBottom: '0' }}>
              Redirigiendo a la página principal...
            </p>
          </div>
        )}

        {!success && (
          <form onSubmit={handleSubmit}>
            <div className="auth-field">
              <label htmlFor="newPassword">Nueva Contraseña</label>
              <input
                id="newPassword"
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                placeholder="Primera letra mayúscula, mínimo 8 caracteres"
                minLength={8}
              />
            </div>

            <div className="auth-field">
              <label htmlFor="confirmPassword">Confirmar Contraseña</label>
              <input
                id="confirmPassword"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                placeholder="Repite tu contraseña"
              />
            </div>

            <button type="submit" className="btn" disabled={loading}>
              {loading ? 'Actualizando...' : 'Restablecer Contraseña'}
            </button>
          </form>
        )}
      </div>
    </div>
  )
}

export default ResetPassword


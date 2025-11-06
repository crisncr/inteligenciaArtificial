import { useState } from 'react'
import { authAPI } from '../utils/api'

function ForgotPassword({ onBack, onClose }) {
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setSuccess(false)
    setLoading(true)

    try {
      // Normalizar email a minúsculas
      const emailNormalized = email.toLowerCase().trim()
      await authAPI.forgotPassword(emailNormalized)
      setSuccess(true)
    } catch (err) {
      setError(err.message || 'Error al solicitar recuperación de contraseña')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-modal">
      <div className="auth-content">
        <button className="auth-close" onClick={onClose}>×</button>
        <h2>Recuperar Contraseña</h2>
        <p className="auth-subtitle">Ingresa tu email para recibir un enlace de recuperación</p>

        {error && <div className="auth-error">{error}</div>}
        {success && (
          <div className="auth-success">
            <p>Si el email existe, recibirás un enlace para recuperar tu contraseña en breve.</p>
            <p>Revisa tu bandeja de entrada y carpeta de spam.</p>
          </div>
        )}

        {!success ? (
          <form onSubmit={handleSubmit}>
            <div className="auth-field">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="tu@email.com"
              />
            </div>

            <button type="submit" className="btn" disabled={loading}>
              {loading ? 'Enviando...' : 'Enviar enlace de recuperación'}
            </button>
          </form>
        ) : (
          <button className="btn" onClick={onBack}>
            Volver a iniciar sesión
          </button>
        )}

        <div className="auth-switch">
          <p><button className="auth-link" onClick={onBack}>Volver a iniciar sesión</button></p>
        </div>
      </div>
    </div>
  )
}

export default ForgotPassword


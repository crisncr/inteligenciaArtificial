import { useState } from 'react'
import { authAPI } from '../utils/api'

function Login({ onLogin, onSwitchToRegister, onForgotPassword, onClose }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      // Login con API real
      await authAPI.login(email, password)
      
      // Obtener información del usuario
      const user = await authAPI.getCurrentUser()
      
      onLogin(user)
      if (onClose) onClose()
    } catch (err) {
      setError(err.message || 'Email o contraseña incorrectos')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-modal">
      <div className="auth-content">
        <button className="auth-close" onClick={onClose}>×</button>
        <h2>Iniciar Sesión</h2>
        <p className="auth-subtitle">Accede a tu cuenta para continuar</p>

        {error && <div className="auth-error">{error}</div>}

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

          <div className="auth-field">
            <label htmlFor="password">Contraseña</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="••••••••"
            />
          </div>

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Iniciando sesión...' : 'Iniciar Sesión'}
          </button>
        </form>

        <div className="auth-switch">
          <p>¿No tienes cuenta? <button className="auth-link" onClick={onSwitchToRegister}>Regístrate</button></p>
          <p><button type="button" className="auth-link" onClick={onForgotPassword}>¿Olvidaste tu contraseña?</button></p>
        </div>
      </div>
    </div>
  )
}

export default Login



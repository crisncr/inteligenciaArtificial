import { useState } from 'react'
import { authAPI } from '../utils/api'

function Register({ onRegister, onSwitchToLogin, onClose }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('Las contraseñas no coinciden')
      return
    }

    // Validación de contraseña: primera letra mayúscula
    if (password.length < 8) {
      setError('La contraseña debe tener al menos 8 caracteres')
      return
    }

    if (!password[0] || !password[0].match(/[A-Z]/)) {
      setError('La contraseña debe comenzar con una letra mayúscula')
      return
    }

    if (!password.match(/[a-z]/)) {
      setError('La contraseña debe contener al menos una letra minúscula')
      return
    }

    if (!password.match(/[0-9]/)) {
      setError('La contraseña debe contener al menos un número')
      return
    }

    setLoading(true)

    try {
      // Normalizar email a minúsculas
      const emailNormalized = email.toLowerCase().trim()
      
      // Registro con API real
      const newUser = await authAPI.register({ name, email: emailNormalized, password })
      console.log('Usuario registrado:', newUser)
      
      // Login automático después del registro (usar email normalizado)
      const loginResponse = await authAPI.login(emailNormalized, password)
      console.log('Login response:', loginResponse)
      
      // Verificar que el token se guardó
      if (!loginResponse?.access_token) {
        throw new Error('No se recibió el token de acceso')
      }
      
      // Verificar que el token se guardó en localStorage
      const savedToken = localStorage.getItem('access_token')
      console.log('Token guardado en localStorage:', savedToken ? 'Sí' : 'No')
      
      if (!savedToken) {
        throw new Error('El token no se guardó correctamente')
      }
      
      // Esperar un momento para asegurar que el token se guardó
      await new Promise(resolve => setTimeout(resolve, 200))
      
      // Obtener información completa del usuario
      try {
        const user = await authAPI.getCurrentUser()
        console.log('Usuario obtenido:', user)
        onRegister(user)
        if (onClose) onClose()
      } catch (userError) {
        console.error('Error obteniendo usuario:', userError)
        // Si falla obtener el usuario pero el login fue exitoso, usar los datos del registro
        onRegister(newUser)
        if (onClose) onClose()
      }
    } catch (err) {
      setError(err.message || 'Error al registrar usuario')
      console.error('Error en registro:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-modal">
      <div className="auth-content">
        <button className="auth-close" onClick={onClose}>×</button>
        <h2>Crear Cuenta</h2>
        <p className="auth-subtitle">Regístrate para acceder a todos los planes</p>

        {error && <div className="auth-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="auth-field">
            <label htmlFor="name">Nombre</label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              placeholder="Tu nombre"
            />
          </div>

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
            {loading ? 'Creando cuenta...' : 'Registrarse'}
          </button>
        </form>

        <div className="auth-switch">
          <p>¿Ya tienes cuenta? <button className="auth-link" onClick={onSwitchToLogin}>Inicia sesión</button></p>
        </div>
      </div>
    </div>
  )
}

export default Register



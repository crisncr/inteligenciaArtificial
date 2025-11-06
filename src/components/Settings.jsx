import { useState } from 'react'
import { authAPI } from '../utils/api'

function Settings({ user, onUserUpdate }) {
  const [name, setName] = useState(user?.name || '')
  const [email, setEmail] = useState(user?.email || '')
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState({ type: '', text: '' })

  const handleUpdateProfile = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMessage({ type: '', text: '' })

    try {
      const updatedUser = await authAPI.updateProfile({ name, email })
      setMessage({ type: 'success', text: 'Perfil actualizado correctamente' })
      if (onUserUpdate) {
        onUserUpdate(updatedUser)
      }
      // Limpiar campos de contraseña
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err) {
      setMessage({ 
        type: 'error', 
        text: err.response?.data?.detail || err.message || 'Error al actualizar perfil' 
      })
    } finally {
      setLoading(false)
    }
  }

  const handleChangePassword = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMessage({ type: '', text: '' })

    // Validar que las contraseñas coincidan
    if (newPassword !== confirmPassword) {
      setMessage({ type: 'error', text: 'Las contraseñas no coinciden' })
      setLoading(false)
      return
    }

    try {
      await authAPI.changePassword({ current_password: currentPassword, new_password: newPassword })
      setMessage({ type: 'success', text: 'Contraseña actualizada correctamente' })
      // Limpiar campos
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err) {
      setMessage({ 
        type: 'error', 
        text: err.response?.data?.detail || err.message || 'Error al cambiar contraseña' 
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="settings-panel">
      <h2>Ajustes de Cuenta</h2>
      <p className="subtitle">
        Gestiona tu información personal y seguridad
      </p>

      {message.text && (
        <div className={`message ${message.type}`}>
          {message.text}
        </div>
      )}

      <div className="settings-sections">
        {/* Sección de Perfil */}
        <div className="settings-section">
          <h3>Información Personal</h3>
          <form onSubmit={handleUpdateProfile} className="settings-form">
            <div className="form-field">
              <label htmlFor="name">Nombre</label>
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                placeholder="Tu nombre"
              />
            </div>
            <div className="form-field">
              <label htmlFor="email">Correo Electrónico</label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="tu@email.com"
              />
            </div>
            <button type="submit" className="btn" disabled={loading}>
              {loading ? 'Guardando...' : 'Actualizar Perfil'}
            </button>
          </form>
        </div>

        {/* Sección de Contraseña */}
        <div className="settings-section">
          <h3>Cambiar Contraseña</h3>
          <form onSubmit={handleChangePassword} className="settings-form">
            <div className="form-field">
              <label htmlFor="currentPassword">Contraseña Actual</label>
              <input
                type="password"
                id="currentPassword"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
                placeholder="Ingresa tu contraseña actual"
              />
            </div>
            <div className="form-field">
              <label htmlFor="newPassword">Nueva Contraseña</label>
              <input
                type="password"
                id="newPassword"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                placeholder="Ingresa tu nueva contraseña"
              />
            </div>
            <div className="form-field">
              <label htmlFor="confirmPassword">Confirmar Nueva Contraseña</label>
              <input
                type="password"
                id="confirmPassword"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                placeholder="Confirma tu nueva contraseña"
              />
            </div>
            <button type="submit" className="btn" disabled={loading}>
              {loading ? 'Cambiando...' : 'Cambiar Contraseña'}
            </button>
          </form>
        </div>
      </div>
    </section>
  )
}

export default Settings


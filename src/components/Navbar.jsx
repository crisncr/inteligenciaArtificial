import { useState, useEffect } from 'react'

function Navbar({ user, onLoginClick, onRegisterClick, onLogout, transparent = false }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Cerrar menú con ESC y cerrar cuando se hace clic fuera
  useEffect(() => {
    const handleEsc = (event) => {
      if (event.key === 'Escape' && mobileMenuOpen) {
        setMobileMenuOpen(false)
      }
    }
    
    const handleClickOutside = (event) => {
      // Si el menú está abierto y se hace clic fuera del navbar, cerrarlo
      if (mobileMenuOpen && !event.target.closest('.nav')) {
        setMobileMenuOpen(false)
      }
    }

    if (mobileMenuOpen) {
      document.addEventListener('keydown', handleEsc)
      // Usar setTimeout para evitar que el clic en el botón hamburguesa cierre el menú inmediatamente
      setTimeout(() => {
        document.addEventListener('click', handleClickOutside)
      }, 100)
    }

    return () => {
      document.removeEventListener('keydown', handleEsc)
      document.removeEventListener('click', handleClickOutside)
    }
  }, [mobileMenuOpen])

  const scrollToSection = (e, sectionId) => {
    e.preventDefault()
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
    // Cerrar menú móvil después de hacer clic
    setMobileMenuOpen(false)
  }

  const handleMenuClick = () => {
    console.log('🔍 [Navbar] Menú clickeado, estado actual:', mobileMenuOpen)
    console.log('🔍 [Navbar] Usuario:', user ? (user.name || user.email) : 'No hay usuario')
    setMobileMenuOpen(!mobileMenuOpen)
  }

  const handleCloseMenu = () => {
    console.log('🔍 [Navbar] Cerrando menú')
    setMobileMenuOpen(false)
  }

  return (
    <nav className={`nav ${transparent ? 'nav--transparent' : ''}`} style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 1001 }}>
      {/* Logo solo cuando NO hay usuario (página pública) */}
      {!user && (
        <div className="nav__brand">
          <img src="/favicon.svg" alt="logo" width="24" height="24" />
          <span>Sentimetría</span>
        </div>
      )}
      {/* Cuando hay usuario, no mostrar logo */}
      
      {/* Menú hamburguesa para móviles - Siempre visible en móviles */}
      <button 
        className="nav__mobile-toggle" 
        onClick={handleMenuClick}
        aria-label="Toggle menu"
        aria-expanded={mobileMenuOpen}
      >
        <span className={mobileMenuOpen ? 'open' : ''}></span>
        <span className={mobileMenuOpen ? 'open' : ''}></span>
        <span className={mobileMenuOpen ? 'open' : ''}></span>
      </button>

      {/* Menú de escritorio */}
      <div className="nav__links">
        {user ? (
          <>
            <span className="nav-user">👤 {user.name}</span>
            <button className="btn--ghost btn--small" onClick={onLogout}>Cerrar sesión</button>
          </>
        ) : (
          <>
            <button className="btn--ghost" onClick={onLoginClick}>Iniciar sesión</button>
            <button className="btn" onClick={onRegisterClick}>Registrarse</button>
          </>
        )}
      </div>

      {/* Menú móvil - Dropdown que se despliega hacia abajo */}
      <div className={`nav__mobile-menu ${mobileMenuOpen ? 'open' : ''}`}>
        <div className="nav__mobile-content">
          {user ? (
            <>
              <div className="nav-user">
                👤 {user.name || user.email || 'Usuario'}
              </div>
              <button 
                className="btn--ghost" 
                onClick={() => { 
                  console.log('🔍 [Navbar] Cerrar sesión clickeado')
                  onLogout(); 
                  handleCloseMenu(); 
                }}
              >
                Cerrar sesión
              </button>
            </>
          ) : (
            <>
              <button 
                className="btn--ghost" 
                onClick={() => { 
                  onLoginClick(); 
                  handleCloseMenu(); 
                }}
              >
                Iniciar sesión
              </button>
              <button 
                className="btn" 
                onClick={() => { 
                  onRegisterClick(); 
                  handleCloseMenu(); 
                }}
              >
                Registrarse
              </button>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}

export default Navbar


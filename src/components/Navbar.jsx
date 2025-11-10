import { useState, useEffect } from 'react'

function Navbar({ user, onLoginClick, onRegisterClick, onLogout, transparent = false }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Cerrar menú con ESC
  useEffect(() => {
    const handleEsc = (event) => {
      if (event.key === 'Escape' && mobileMenuOpen) {
        setMobileMenuOpen(false)
      }
    }

    if (mobileMenuOpen) {
      document.addEventListener('keydown', handleEsc)
      // Prevenir scroll del body cuando el menú está abierto
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }

    return () => {
      document.removeEventListener('keydown', handleEsc)
      document.body.style.overflow = ''
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
    setMobileMenuOpen(!mobileMenuOpen)
  }

  const handleCloseMenu = () => {
    setMobileMenuOpen(false)
  }

  return (
    <nav className={`nav ${transparent ? 'nav--transparent' : ''}`}>
      <div className="nav__brand">
        {!user && (
          <>
            <img src="/favicon.svg" alt="logo" width="24" height="24" />
            <span>Sentimetría</span>
          </>
        )}
      </div>
      
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

      {/* Menú móvil */}
      {mobileMenuOpen && (
        <div 
          className={`nav__mobile-menu ${mobileMenuOpen ? 'open' : ''}`}
          onClick={(e) => {
            // Cerrar menú si se hace clic en el overlay (fondo)
            if (e.target === e.currentTarget) {
              handleCloseMenu()
            }
          }}
        >
          <div className="nav__mobile-content" onClick={(e) => e.stopPropagation()}>
            {user ? (
              <>
                <div className="nav-user">👤 {user.name}</div>
                <button 
                  className="btn--ghost" 
                  onClick={() => { 
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
      )}
    </nav>
  )
}

export default Navbar


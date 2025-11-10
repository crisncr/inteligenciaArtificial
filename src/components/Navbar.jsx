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
    console.log('🔍 [Navbar] Menú clickeado, estado actual:', mobileMenuOpen)
    console.log('🔍 [Navbar] Usuario:', user ? (user.name || user.email) : 'No hay usuario')
    setMobileMenuOpen(!mobileMenuOpen)
  }

  const handleCloseMenu = () => {
    console.log('🔍 [Navbar] Cerrando menú')
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

      {/* Menú móvil - Siempre renderizado, controlado por CSS */}
      <div 
        className={`nav__mobile-menu ${mobileMenuOpen ? 'open' : ''}`}
        onClick={(e) => {
          // Cerrar menú si se hace clic en el overlay (fondo)
          if (e.target === e.currentTarget) {
            handleCloseMenu()
          }
        }}
      >
        <div 
          className="nav__mobile-content" 
          onClick={(e) => e.stopPropagation()}
          style={{
            display: mobileMenuOpen ? 'flex' : 'none',
            opacity: mobileMenuOpen ? 1 : 0,
            visibility: mobileMenuOpen ? 'visible' : 'hidden'
          }}
        >
          {user ? (
            <>
              <div 
                className="nav-user" 
                style={{ 
                  display: 'block', 
                  visibility: 'visible', 
                  opacity: 1,
                  color: 'var(--text)',
                  backgroundColor: 'rgba(110, 139, 255, 0.1)',
                  padding: '16px 20px',
                  borderRadius: '12px',
                  border: '1px solid rgba(110, 139, 255, 0.3)',
                  textAlign: 'center',
                  fontWeight: 600,
                  fontSize: '1.1rem',
                  marginBottom: '10px'
                }}
              >
                👤 {user.name || user.email || 'Usuario'}
              </div>
              <button 
                className="btn--ghost" 
                style={{ 
                  display: 'block', 
                  visibility: 'visible', 
                  opacity: 1,
                  width: '100%',
                  padding: '16px 20px',
                  fontSize: '1rem',
                  color: 'var(--text)',
                  backgroundColor: 'transparent',
                  border: '1px solid rgba(255,255,255,0.18)',
                  borderRadius: '12px',
                  cursor: 'pointer'
                }}
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
                style={{ 
                  display: 'block', 
                  visibility: 'visible', 
                  opacity: 1,
                  width: '100%',
                  padding: '16px 20px',
                  fontSize: '1rem',
                  color: 'var(--text)'
                }}
                onClick={() => { 
                  onLoginClick(); 
                  handleCloseMenu(); 
                }}
              >
                Iniciar sesión
              </button>
              <button 
                className="btn" 
                style={{ 
                  display: 'block', 
                  visibility: 'visible', 
                  opacity: 1,
                  width: '100%',
                  padding: '16px 20px',
                  fontSize: '1rem'
                }}
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


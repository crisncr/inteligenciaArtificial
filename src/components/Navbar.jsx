import { useState } from 'react'

function Navbar({ user, onLoginClick, onRegisterClick, onLogout }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

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
    <nav className="nav">
      <div className="nav__brand">
        <img src="/favicon.svg" alt="logo" width="24" height="24" />
        <span>Sentimetría</span>
      </div>
      
      {/* Menú hamburguesa para móviles */}
      <button 
        className="nav__mobile-toggle" 
        onClick={handleMenuClick}
        aria-label="Toggle menu"
      >
        <span className={mobileMenuOpen ? 'open' : ''}></span>
        <span className={mobileMenuOpen ? 'open' : ''}></span>
        <span className={mobileMenuOpen ? 'open' : ''}></span>
      </button>

      {/* Menú de escritorio */}
      <div className="nav__links">
        <a href="#features" onClick={(e) => scrollToSection(e, 'features')}>Características</a>
        <a href="#precio" onClick={(e) => scrollToSection(e, 'precio')}>Precios</a>
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
      <div 
        className={`nav__mobile-menu ${mobileMenuOpen ? 'open' : ''}`}
        onClick={(e) => {
          // Cerrar menú si se hace clic en el overlay (fondo)
          if (e.target === e.currentTarget) {
            handleCloseMenu()
          }
        }}
      >
        <div className="nav__mobile-content">
          <a href="#features" onClick={(e) => scrollToSection(e, 'features')}>Características</a>
          <a href="#precio" onClick={(e) => scrollToSection(e, 'precio')}>Precios</a>
          {user ? (
            <>
              <span className="nav-user">👤 {user.name}</span>
              <button className="btn--ghost btn--small" onClick={() => { onLogout(); handleCloseMenu(); }}>Cerrar sesión</button>
            </>
          ) : (
            <>
              <button className="btn--ghost" onClick={() => { onLoginClick(); handleCloseMenu(); }}>Iniciar sesión</button>
              <button className="btn" onClick={() => { onRegisterClick(); handleCloseMenu(); }}>Registrarse</button>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}

export default Navbar


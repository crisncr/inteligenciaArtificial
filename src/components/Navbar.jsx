function Navbar({ user, onLoginClick, onRegisterClick, onLogout }) {
  const scrollToSection = (e, sectionId) => {
    e.preventDefault()
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <nav className="nav">
      <div className="nav__brand">
        <img src="/favicon.svg" alt="logo" width="24" height="24" />
        <span>Sentimetría</span>
      </div>
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
    </nav>
  )
}

export default Navbar


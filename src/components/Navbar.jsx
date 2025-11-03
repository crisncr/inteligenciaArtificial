function Navbar() {
  return (
    <nav className="nav">
      <div className="nav__brand">
        <img src="/favicon.svg" alt="logo" width="24" height="24" />
        <span>Sentimetría</span>
      </div>
      <div className="nav__links">
        <a href="#features">Características</a>
        <a href="#precio">Precios</a>
        <a className="btn--ghost" href="#analizar">Demo</a>
        <a className="btn" href="#analizar">Probar gratis</a>
      </div>
    </nav>
  )
}

export default Navbar


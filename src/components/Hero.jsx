function Hero() {
  const scrollToSection = (e, sectionId) => {
    e.preventDefault()
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <section className="hero">
      <div className="hero__copy">
        <h1>Analiza sentimientos en segundos</h1>
        <p className="lead">
          Clasifica textos en <strong>positivo</strong>, <strong>negativo</strong> o <strong>neutral</strong> con un motor optimizado para español.
        </p>
        <div className="hero__ctas">
          <a className="btn" href="#analizar" onClick={(e) => scrollToSection(e, 'analizar')}>Probar ahora</a>
          <a className="btn--ghost" href="#features" onClick={(e) => scrollToSection(e, 'features')}>Ver características</a>
        </div>
      </div>
      <div className="hero__mock">
        <div className="device">
          <div className="device__top"></div>
          <div className="device__screen">
            <div className="pill pill--pos">🟢 Excelente atención</div>
            <div className="pill pill--neu">🟡 Aceptable, podría mejorar</div>
            <div className="pill pill--neg">🔴 Pésimo servicio</div>
            <div className="pill pill--pos">🟢 Muy rápido y amable</div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Hero


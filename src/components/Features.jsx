function Features() {
  return (
    <section id="features" className="features">
      <div className="feature">
        <div className="feat__icon">⚡</div>
        <h3>Respuesta inmediata</h3>
        <p>Procesamiento en milisegundos con reglas optimizadas para español.</p>
      </div>
      <div className="feature">
        <div className="feat__icon">🧠</div>
        <h3>Negaciones e intensificadores</h3>
        <p>Interpreta matices como "no muy bueno" o "extremadamente útil".</p>
      </div>
      <div className="feature">
        <div className="feat__icon">🎯</div>
        <h3>Fácil de integrar</h3>
        <p>API simple con `POST /analyze` para tus apps y automatizaciones.</p>
      </div>
    </section>
  )
}

export default Features


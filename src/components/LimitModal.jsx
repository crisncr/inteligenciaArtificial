function LimitModal({ onClose, onSelectPlan, isAuthenticated }) {
  const scrollToPricing = () => {
    onClose()
    setTimeout(() => {
      const element = document.getElementById('precio')
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }, 300)
  }

  return (
    <div className="limit-modal">
      <div className="limit-content">
        <div className="limit-icon">🔒</div>
        <h2>Límite de Análisis Gratuitos Alcanzado</h2>
        <p className="limit-text">
          Has utilizado tus 3 análisis gratuitos. Para continuar analizando textos, 
          necesitas seleccionar un plan.
        </p>

        {!isAuthenticated && (
          <div className="limit-auth-note">
            <p>💡 <strong>Nota:</strong> Debes iniciar sesión o registrarte para seleccionar un plan.</p>
          </div>
        )}

        <div className="limit-actions">
          {isAuthenticated ? (
            <button className="btn" onClick={scrollToPricing}>
              Ver Planes Disponibles
            </button>
          ) : (
            <button className="btn" onClick={onSelectPlan}>
              Iniciar Sesión / Registrarse
            </button>
          )}
          <button className="btn--ghost" onClick={onClose}>
            Cerrar
          </button>
        </div>
      </div>
    </div>
  )
}

export default LimitModal



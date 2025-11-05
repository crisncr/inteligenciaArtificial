function Pricing({ user, onSelectPlan, onLoginRequired }) {
  const plans = [
    {
      name: 'Básico',
      price: 'Gratis',
      features: [
        'Hasta 10 análisis por día',
        'Análisis básico de sentimientos',
        'Soporte por email',
        'API básica'
      ],
      popular: false
    },
    {
      name: 'Pro',
      price: '$9.99',
      period: '/mes',
      features: [
        'Análisis ilimitados',
        'Análisis avanzado de sentimientos',
        'Historial completo',
        'API completa',
        'Soporte prioritario',
        'Exportar resultados'
      ],
      popular: true
    },
    {
      name: 'Enterprise',
      price: 'Personalizado',
      features: [
        'Todo lo de Pro',
        'Análisis en tiempo real',
        'Integración personalizada',
        'Soporte 24/7',
        'Analytics avanzados',
        'SLA garantizado'
      ],
      popular: false
    }
  ]

  return (
    <section id="precio" className="pricing-panel">
      <h2>Planes y Precios</h2>
      <p className="subtitle">
        Elige el plan que mejor se adapte a tus necesidades
      </p>

      <div className="pricing-grid">
        {plans.map((plan, index) => (
          <div 
            key={index} 
            className={`pricing-card ${plan.popular ? 'popular' : ''}`}
          >
            {plan.popular && <div className="popular-badge">Más Popular</div>}
            <div className="pricing-header">
              <h3>{plan.name}</h3>
              <div className="pricing-price">
                <span className="price">{plan.price}</span>
                {plan.period && <span className="period">{plan.period}</span>}
              </div>
            </div>
            <ul className="pricing-features">
              {plan.features.map((feature, fIndex) => (
                <li key={fIndex}>
                  <span className="check-icon">✓</span>
                  {feature}
                </li>
              ))}
            </ul>
            <button 
              className={`btn ${plan.popular ? '' : 'btn--ghost'}`}
              onClick={() => {
                if (!user) {
                  if (onLoginRequired) {
                    onLoginRequired()
                  }
                } else {
                  if (onSelectPlan) {
                    onSelectPlan(plan.name.toLowerCase())
                  }
                }
              }}
            >
              {plan.price === 'Gratis' ? 'Comenzar gratis' : 
               plan.price === 'Personalizado' ? 'Contactar' : 
               !user ? 'Iniciar sesión' : 'Empezar ahora'}
            </button>
          </div>
        ))}
      </div>

      {!user && (
        <div className="pricing-auth-note">
          <p>🔐 <strong>Nota:</strong> Debes iniciar sesión o registrarte para seleccionar un plan.</p>
        </div>
      )}

      <div className="pricing-note">
        <p>Todos los planes incluyen análisis de sentimientos en español con precisión optimizada.</p>
      </div>
    </section>
  )
}

export default Pricing


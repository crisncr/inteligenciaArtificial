function Payments({ user }) {
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
      current: user?.plan === 'free'
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
      current: user?.plan === 'pro'
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
      current: user?.plan === 'enterprise'
    }
  ]

  return (
    <section className="payments-panel">
      <h2>Pagos y Suscripciones</h2>
      <p className="subtitle">
        Gestiona tu plan y suscripción
      </p>

      <div className="current-plan">
        <h3>Plan Actual</h3>
        <div className="plan-badge">
          <span className="plan-name">{user?.plan === 'free' ? 'Básico' : user?.plan === 'pro' ? 'Pro' : 'Enterprise'}</span>
          {user?.plan !== 'free' && <span className="plan-status">Activo</span>}
        </div>
      </div>

      <div className="pricing-grid">
        {plans.map((plan, index) => (
          <div 
            key={index} 
            className={`pricing-card ${plan.current ? 'current' : ''} ${plan.name === 'Pro' ? 'popular' : ''}`}
          >
            {plan.name === 'Pro' && <div className="popular-badge">Más Popular</div>}
            {plan.current && <div className="current-badge">Tu Plan</div>}
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
              className={`btn ${plan.current ? 'btn--ghost' : plan.name === 'Pro' ? '' : 'btn--ghost'}`}
              disabled={plan.current}
            >
              {plan.current ? 'Plan Actual' : 
               plan.price === 'Personalizado' ? 'Contactar' : 
               'Cambiar Plan'}
            </button>
          </div>
        ))}
      </div>
    </section>
  )
}

export default Payments


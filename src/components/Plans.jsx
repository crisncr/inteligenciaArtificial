function Plans({ user, onSelectPlan }) {
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
      current: user?.plan === 'pro',
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
      current: user?.plan === 'enterprise'
    }
  ]

  return (
    <section className="plans-panel">
      <h2>Planes y Precios</h2>
      <p className="subtitle">
        Elige el plan que mejor se adapte a tus necesidades
      </p>

      <div className="pricing-grid">
        {plans.map((plan, index) => (
          <div 
            key={index} 
            className={`pricing-card ${plan.current ? 'current' : ''} ${plan.popular ? 'popular' : ''}`}
          >
            {plan.popular && <div className="popular-badge">Más Popular</div>}
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
              className={`btn ${plan.current ? 'btn--ghost' : plan.popular ? '' : 'btn--ghost'}`}
              onClick={() => onSelectPlan && onSelectPlan(plan.name.toLowerCase())}
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

export default Plans


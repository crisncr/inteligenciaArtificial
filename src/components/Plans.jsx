import { getPlanFeatures } from '../utils/planFeatures'

function Plans({ user, onSelectPlan }) {
  const plans = [
    {
      name: 'Básico',
      planId: 'free',
      price: 'Gratis',
      features: getPlanFeatures('free'),
      current: user?.plan === 'free',
      purchasable: false
    },
    {
      name: 'Pro',
      planId: 'pro',
      price: '$9.99',
      period: '/mes',
      features: getPlanFeatures('pro'),
      current: user?.plan === 'pro',
      popular: true,
      purchasable: true
    },
    {
      name: 'Enterprise',
      planId: 'enterprise',
      price: '$29.99',
      period: '/mes',
      features: getPlanFeatures('enterprise'),
      current: user?.plan === 'enterprise',
      purchasable: true
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
              onClick={() => onSelectPlan && onSelectPlan(plan.planId)}
              disabled={plan.current}
            >
              {plan.current ? 'Plan Actual' : 
               plan.purchasable ? 'Comprar Plan' : 
               'Seleccionar Plan'}
            </button>
          </div>
        ))}
      </div>
    </section>
  )
}

export default Plans


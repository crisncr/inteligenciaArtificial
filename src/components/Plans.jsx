import { useState } from 'react'
import { getPlanFeatures } from '../utils/planFeatures'

function Plans({ user, onSelectPlan }) {
  const [expandedPlans, setExpandedPlans] = useState({})
  
  const togglePlan = (index) => {
    setExpandedPlans(prev => ({
      ...prev,
      [index]: !prev[index]
    }))
  }
  
  const getVisibleFeatures = (planFeatures, isExpanded) => {
    if (isExpanded) return planFeatures
    // Mostrar solo las primeras 6 características (excluyendo líneas vacías)
    const nonEmptyFeatures = planFeatures.filter(f => f.trim() !== '')
    const visibleCount = Math.min(6, nonEmptyFeatures.length)
    return planFeatures.filter(f => {
      if (f.trim() === '') return true // Mantener líneas vacías para formato
      const nonEmptyIndex = nonEmptyFeatures.indexOf(f)
      return nonEmptyIndex < visibleCount
    })
  }
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
              {getVisibleFeatures(plan.features, expandedPlans[index]).map((feature, fIndex) => {
                // No mostrar líneas vacías
                if (feature.trim() === '') return null
                return (
                  <li key={fIndex}>
                    <span className="check-icon">✓</span>
                    {feature}
                  </li>
                )
              })}
            </ul>
            {plan.features.filter(f => f.trim() !== '').length > 6 && (
              <button
                className="btn--link"
                onClick={() => togglePlan(index)}
                style={{ 
                  marginTop: '12px', 
                  background: 'transparent', 
                  border: 'none', 
                  color: 'var(--primary)', 
                  cursor: 'pointer',
                  textDecoration: 'underline',
                  fontSize: '0.9rem'
                }}
              >
                {expandedPlans[index] ? 'Ver menos' : 'Ver más'}
              </button>
            )}
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


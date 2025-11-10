import { useState } from 'react'

function Pricing({ user, onSelectPlan, onLoginRequired }) {
  const [expandedPlans, setExpandedPlans] = useState({})
  
  const plans = [
    {
      name: 'Básico',
      price: 'Gratis',
      features: [
        'Análisis de sentimientos con Red Neuronal',
        'Hasta 10 análisis por día',
        'Carga de datasets (CSV/JSON)',
        'Clasificación automática de texto',
        'Soporte por email',
        'API básica para integraciones'
      ],
      popular: false
    },
    {
      name: 'Pro',
      price: '$9.99',
      period: '/mes',
      features: [
        'Análisis ilimitados con Red Neuronal',
        'Optimización de rutas de distribución',
        'Historial completo de análisis',
        'Estadísticas y métricas avanzadas',
        'API completa con documentación',
        'Integración con APIs externas',
        'Soporte prioritario',
        'Exportar resultados en CSV/JSON',
        'Diagnósticos automáticos',
        'Búsqueda avanzada en comentarios'
      ],
      popular: true
    },
    {
      name: 'Enterprise',
      price: 'Personalizado',
      features: [
        'Todo lo incluido en Pro',
        'Predicción de ventas por región',
        'Análisis en tiempo real',
        'Análisis avanzado multi-idioma',
        'Integración personalizada',
        'Soporte 24/7 dedicado',
        'Analytics avanzados y reportes',
        'Exportación de datos ilimitada',
        'Integraciones con Slack, Zapier',
        'SLA garantizado',
        'Capacitación personalizada',
        'Cuenta manager dedicado'
      ],
      popular: false
    }
  ]
  
  const togglePlan = (index) => {
    setExpandedPlans(prev => ({
      ...prev,
      [index]: !prev[index]
    }))
  }
  
  const getVisibleFeatures = (planFeatures, isExpanded) => {
    if (isExpanded) return planFeatures
    return planFeatures.slice(0, 4) // Mostrar solo las primeras 4 características
  }

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
              {getVisibleFeatures(plan.features, expandedPlans[index]).map((feature, fIndex) => (
                <li key={fIndex}>
                  <span className="check-icon">✓</span>
                  {feature}
                </li>
              ))}
            </ul>
            {plan.features.length > 4 && (
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


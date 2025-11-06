import { getPlanFeatures } from '../utils/planFeatures'

function DashboardHome({ user, onSelectPlan }) {
  const plan = user?.plan || 'free'
  const planName = plan === 'free' ? 'Básico' : plan === 'pro' ? 'Pro' : 'Enterprise'
  const features = getPlanFeatures(plan)

  return (
    <section className="dashboard-home">
      <div className="overview-header">
        <div className="overview-title">
          <h1>Overview</h1>
          <p className="overview-subtitle">Resumen de tu cuenta y servicios</p>
        </div>
      </div>

      <div className="overview-sections">
        {/* Sección de Plan Actual */}
        <div className="overview-section">
          <h2>Plan Actual</h2>
          <div className="current-plan-card">
            <div className="plan-info">
              <div className="plan-name-badge">
                <span className="plan-name">{planName}</span>
                {plan !== 'free' && <span className="plan-status">Activo</span>}
              </div>
              <p className="plan-description">
                {plan === 'free' && 'Plan gratuito con funcionalidades básicas'}
                {plan === 'pro' && 'Plan profesional con análisis ilimitados y funciones avanzadas'}
                {plan === 'enterprise' && 'Plan empresarial con todas las funciones premium'}
              </p>
            </div>
            <div className="plan-actions">
              <button 
                className="btn btn--ghost"
                onClick={() => onSelectPlan && onSelectPlan('planes')}
              >
                Cambiar Plan
              </button>
            </div>
          </div>
        </div>

        {/* Sección de Servicios */}
        <div className="overview-section">
          <h2>Servicios</h2>
          <div className="services-grid">
            {/* Análisis de Sentimientos - Disponible para todos */}
            <div className="service-card">
              <div className="service-icon">📊</div>
              <div className="service-info">
                <h3>Análisis de Sentimientos</h3>
                <p>Análisis de sentimientos desde APIs externas</p>
                <span className="service-status active">✓ Activo</span>
              </div>
            </div>
            
            {/* API Externa - Solo Pro y Enterprise */}
            {(plan === 'pro' || plan === 'enterprise') && (
              <div className="service-card">
                <div className="service-icon">🔌</div>
                <div className="service-info">
                  <h3>API Externa</h3>
                  <p>Integración con APIs externas para análisis automático</p>
                  <span className="service-status active">✓ Disponible</span>
                </div>
              </div>
            )}
            
            {/* Estadísticas - Solo Pro y Enterprise */}
            {(plan === 'pro' || plan === 'enterprise') && (
              <div className="service-card">
                <div className="service-icon">📈</div>
                <div className="service-info">
                  <h3>Estadísticas</h3>
                  <p>Visualización de métricas y estadísticas detalladas</p>
                  <span className="service-status active">✓ Disponible</span>
                </div>
              </div>
            )}
            
            {/* Análisis Avanzado - Solo Enterprise */}
            {plan === 'enterprise' && (
              <div className="service-card">
                <div className="service-icon">🎯</div>
                <div className="service-info">
                  <h3>Análisis Avanzado</h3>
                  <p>ML avanzado, multi-idioma y análisis de emociones</p>
                  <span className="service-status active">✓ Disponible</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Sección de Features del Plan */}
        <div className="overview-section">
          <h2>Funcionalidades de tu Plan</h2>
          <div className="features-list">
            {features.map((feature, index) => (
              <div key={index} className="feature-item">
                <span className="feature-check">✓</span>
                <span className="feature-text">{feature}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

    </section>
  )
}

export default DashboardHome


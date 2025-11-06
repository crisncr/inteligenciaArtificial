// Items del sidebar según el plan
export const getSidebarItems = (plan) => {
  const baseItems = [
    { id: 'inicio', label: 'Inicio', icon: '🏠', plan: 'free' },
    { id: 'pagos', label: 'Pagos', icon: '💳', plan: 'free' },
    { id: 'planes', label: 'Planes', icon: '📦', plan: 'free' },
    { id: 'soporte', label: 'Ayuda', icon: '💬', plan: 'free' },
    { id: 'ajustes', label: 'Ajustes', icon: '⚙️', plan: 'free' },
  ]

  const proItems = [
    { id: 'historial', label: 'Historial', icon: '📋', plan: 'pro' },
    { id: 'estadisticas', label: 'Estadísticas', icon: '📊', plan: 'pro' },
    { id: 'api-externa', label: 'API Externa', icon: '🔌', plan: 'pro' },
    { id: 'diagnosticos', label: 'Diagnósticos', icon: '🔍', plan: 'pro' },
  ]

  const enterpriseItems = [
    { id: 'analisis-avanzado', label: 'Análisis Avanzado', icon: '🎯', plan: 'enterprise' },
    { id: 'exportar-datos', label: 'Exportar Datos', icon: '📤', plan: 'enterprise' },
    { id: 'integraciones', label: 'Integraciones', icon: '🔗', plan: 'enterprise' },
    { id: 'reportes', label: 'Reportes', icon: '📈', plan: 'enterprise' },
  ]

  let items = [...baseItems]

  if (plan === 'pro' || plan === 'enterprise') {
    items = [...items, ...proItems]
  }

  if (plan === 'enterprise') {
    items = [...items, ...enterpriseItems]
  }

  return items
}

// Features por plan
export const getPlanFeatures = (plan) => {
  const features = {
    free: [
      'Hasta 10 análisis por día',
      'Análisis básico de sentimientos',
      'Soporte por email',
      'API básica',
      'Historial limitado (últimos 30 días)',
    ],
    pro: [
      'Análisis ilimitados',
      'Análisis avanzado de sentimientos',
      'Historial completo',
      'API completa',
      'Soporte prioritario',
      'Exportar resultados',
      'Estadísticas detalladas',
      'API Externa',
      'Diagnósticos',
    ],
    enterprise: [
      'Todo lo de Pro',
      'Análisis en tiempo real',
      'Integración personalizada',
      'Soporte 24/7',
      'Analytics avanzados',
      'SLA garantizado',
      'Análisis Avanzado',
      'Exportar Datos',
      'Integraciones',
      'Reportes personalizados',
      'API dedicada',
      'Webhooks',
    ],
  }

  return features[plan] || features.free
}


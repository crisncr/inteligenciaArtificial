// Items del sidebar según el plan
export const getSidebarItems = (plan) => {
  const baseItems = [
    { id: 'inicio', label: 'Inicio', icon: '🏠', plan: 'free' },
    { id: 'analisis-sentimientos', label: 'Análisis de Sentimientos', icon: '📊', plan: 'free' },
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

// Features por plan - deben coincidir con los items disponibles en el sidebar
export const getPlanFeatures = (plan) => {
  const features = {
    free: [
      'Inicio - Overview de tu cuenta',
      'Análisis de Sentimientos - Analizar texto directamente',
      'Pagos - Historial de pagos',
      'Planes - Cambiar de plan',
      'Ayuda - Soporte y documentación',
      'Ajustes - Configuración de cuenta',
    ],
    pro: [
      'Todo lo de Básico',
      'Historial - Análisis de API externa',
      'Estadísticas - Métricas detalladas',
      'API Externa - Integración con APIs',
      'Diagnósticos - Análisis de comentarios',
    ],
    enterprise: [
      'Todo lo de Pro',
      'Análisis Avanzado - ML y multi-idioma',
      'Exportar Datos - CSV y JSON',
      'Integraciones - Slack, Zapier, Webhooks',
      'Reportes - Personalizados y programados',
    ],
  }

  return features[plan] || features.free
}


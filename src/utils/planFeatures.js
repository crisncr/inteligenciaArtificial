// Items del sidebar según el plan
export const getSidebarItems = (plan) => {
  // Items base que todos los planes tienen
  const baseItems = [
    { id: 'inicio', label: 'Inicio', icon: '🏠', plan: 'free' },
    { id: 'pagos', label: 'Pagos', icon: '💳', plan: 'free' },
    { id: 'planes', label: 'Planes', icon: '📦', plan: 'free' },
    { id: 'soporte', label: 'Ayuda', icon: '💬', plan: 'free' },
  ]

  // Plan Free
  const parte1Items = [
    { id: 'analisis-sentimientos', label: 'Análisis de Sentimientos', icon: '📊', plan: 'free' },
  ]

  // Plan Pro
  const parte2Items = [
    { id: 'historial', label: 'Historial', icon: '📋', plan: 'pro' },
    { id: 'estadisticas', label: 'Estadísticas', icon: '📊', plan: 'pro' },
    { id: 'api-externa', label: 'API Externa', icon: '🔌', plan: 'pro' },
    { id: 'diagnosticos', label: 'Diagnósticos', icon: '🔍', plan: 'pro' },
    { id: 'optimizacion-rutas', label: 'Optimización de Rutas', icon: '🗺️', plan: 'pro' },
  ]

  // Plan Enterprise
  const parte3Items = [
    { id: 'analisis-avanzado', label: 'Análisis Avanzado', icon: '🎯', plan: 'enterprise' },
    { id: 'prediccion-ventas', label: 'Predicción de Ventas', icon: '💰', plan: 'enterprise' },
    { id: 'exportar-datos', label: 'Exportar Datos', icon: '📤', plan: 'enterprise' },
    { id: 'integraciones', label: 'Integraciones', icon: '🔗', plan: 'enterprise' },
    { id: 'reportes', label: 'Reportes', icon: '📈', plan: 'enterprise' },
  ]

  // Ajustes siempre al final
  const settingsItem = [
    { id: 'ajustes', label: 'Ajustes', icon: '⚙️', plan: 'all' },
  ]

  // Construir items según el plan (ACUMULATIVO)
  let items = [...baseItems]

  if (plan === 'free') {
    // Plan Free
    items = [...items, ...parte1Items]
  } else if (plan === 'pro') {
    // Plan Pro
    items = [...items, ...parte1Items, ...parte2Items]
  } else if (plan === 'enterprise') {
    // Plan Enterprise
    items = [...items, ...parte1Items, ...parte2Items, ...parte3Items]
  }

  // Ajustes siempre al final
  items = [...items, ...settingsItem]

  return items
}

// Features por plan - deben coincidir con los items disponibles en el sidebar
export const getPlanFeatures = (plan) => {
  // Plan Free
  const parte1Features = [
    'Inicio - Overview de tu cuenta',
    'Pagos - Historial de pagos',
    'Planes - Cambiar de plan',
    'Ayuda - Soporte y documentación',
    'Ajustes - Configuración de cuenta',
    '',
    '📊 Análisis de Sentimientos',
    '✓ Análisis de Sentimientos - Red Neuronal (10 análisis/día)',
    '✓ Carga de Datasets - Hasta 100 comentarios (CSV/JSON)',
    '✓ Limpieza de Texto - Técnicas de NLP',
    '✓ Búsqueda de Texto - Buscar en comentarios',
    '✓ Clasificación Automática - Positivo/Negativo',
    '✓ Método de Aprendizaje: Supervisado',
    '✓ Algoritmo: Red Neuronal (LSTM)',
  ]

  // Plan Pro
  const parte2Features = [
    '',
    '📊 Análisis de Sentimientos Mejorado',
    '✓ Análisis ilimitado con Red Neuronal',
    '✓ Datasets ilimitados',
    '',
    '🗺️ Optimización de Rutas',
    '✓ Optimización de Rutas - Hasta 50 puntos',
    '✓ Algoritmos de Búsqueda - A*, Dijkstra, TSP',
    '✓ Visualización de Rutas Óptimas',
    '✓ Explicación de Selección de Nodos',
    '✓ Algoritmo: A* (con heurística)',
    '',
    'Funcionalidades Adicionales:',
    '✓ Historial - Análisis de API externa',
    '✓ Estadísticas - Métricas de comentarios de API externa',
    '✓ API Externa - Integración con APIs (obtener comentarios)',
    '✓ Diagnósticos - Análisis de comentarios',
  ]

  // Plan Enterprise
  const parte3Features = [
    '',
    '💰 Predicción de Ventas',
    '✓ Predicción de Ventas - Por región',
    '✓ Modelos de IA - Regresión Lineal / Red Neuronal',
    '✓ Análisis Predictivo - Tendencia de ventas',
    '✓ Visualización de Predicciones',
    '✓ Tipo de Aprendizaje: Supervisado (Regresión)',
    '✓ Algoritmo: Regresión Lineal / Red Neuronal',
    '',
    'Mejoras Adicionales:',
    '✓ Rutas ilimitadas - Sin límite de puntos',
    '',
    'Funcionalidades Adicionales:',
    '✓ Análisis Avanzado - ML y multi-idioma',
    '✓ Exportar Datos - CSV y JSON',
    '✓ Integraciones - Slack, Zapier, Webhooks',
    '✓ Reportes - Personalizados y programados',
  ]

  // Construir features según el plan (ACUMULATIVO)
  if (plan === 'free') {
    return parte1Features
  } else if (plan === 'pro') {
    // Plan Pro
    return [...parte1Features, ...parte2Features]
  } else if (plan === 'enterprise') {
    // Plan Enterprise
    return [...parte1Features, ...parte2Features, ...parte3Features]
  }

  return parte1Features
}


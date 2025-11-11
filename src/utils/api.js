// Utilidades para llamadas a la API
const API_URL = import.meta.env.PROD ? '' : 'http://127.0.0.1:8000'

// Obtener token del localStorage
export const getToken = () => {
  return localStorage.getItem('access_token')
}

// Guardar token en localStorage
export const setToken = (token) => {
  localStorage.setItem('access_token', token)
}

// Eliminar token
export const removeToken = () => {
  localStorage.removeItem('access_token')
}

// Función para hacer requests a la API
export const apiRequest = async (endpoint, options = {}) => {
  const token = getToken()
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    // Si es 401, limpiar el token ya que está inválido
    if (response.status === 401) {
      removeToken()
    }
    let errorData
    try {
      errorData = await response.json()
    } catch {
      errorData = { detail: `Error ${response.status}`, error: `Error ${response.status}` }
    }
    
    // Si es un error 503 (Service Unavailable), el modelo está cargando
    if (response.status === 503) {
      const errorMsg = errorData.error || errorData.detail || 'El modelo se está cargando. Por favor, espera 15-30 segundos e intenta de nuevo.'
      const error = new Error(errorMsg)
      error.response = { data: errorData, status: response.status }
      throw error
    }
    
    const error = new Error(errorData.detail || errorData.message || errorData.error || `Error ${response.status}`)
    error.response = { data: errorData, status: response.status }
    throw error
  }

  return response.json()
}

// Endpoints de autenticación
export const authAPI = {
  register: async (userData) => {
    return apiRequest('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    })
  },

  login: async (email, password) => {
    // Normalizar email a minúsculas
    const emailNormalized = email.toLowerCase().trim()

    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        username: emailNormalized,
        password: password,
      }),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Error desconocido' }))
      throw new Error(error.detail || `Error ${response.status}`)
    }

    const data = await response.json()
    
    // Verificar que se recibió el token
    if (!data.access_token) {
      throw new Error('No se recibió el token de acceso')
    }
    
    setToken(data.access_token)
    return data
  },

  logout: async () => {
    try {
      await apiRequest('/api/auth/logout', { method: 'POST' })
    } catch (error) {
      console.error('Error al cerrar sesión:', error)
    } finally {
      removeToken()
    }
  },

  getCurrentUser: async () => {
    const token = getToken()
    if (!token) {
      throw new Error('No hay token de acceso')
    }
    return apiRequest('/api/auth/me')
  },

  forgotPassword: async (email) => {
    // Normalizar email a minúsculas
    const emailNormalized = email.toLowerCase().trim()
    return apiRequest('/api/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email: emailNormalized }),
    })
  },

  resetPassword: async (token, newPassword) => {
    return apiRequest('/api/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify({ token, new_password: newPassword }),
    })
  },

  verifyEmail: async (token) => {
    return apiRequest('/api/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ token }),
    })
  },

  updateProfile: async (userData) => {
    return apiRequest('/api/auth/me', {
      method: 'PUT',
      body: JSON.stringify(userData),
    })
  },

  changePassword: async (passwordData) => {
    return apiRequest('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(passwordData),
    })
  },
}

// Endpoints de análisis
export const analysesAPI = {
  create: async (text) => {
    return apiRequest('/api/analyses', {
      method: 'POST',
      body: JSON.stringify({ text }),
    })
  },

  getAll: async (skip = 0, limit = 50) => {
    return apiRequest(`/api/analyses?skip=${skip}&limit=${limit}`)
  },

  getById: async (id) => {
    return apiRequest(`/api/analyses/${id}`)
  },

  delete: async (id) => {
    return apiRequest(`/api/analyses/${id}`, { method: 'DELETE' })
  },

  getStats: async () => {
    return apiRequest('/api/analyses/stats/summary')
  },

  batch: async (texts) => {
    return apiRequest('/api/analyses/batch', {
      method: 'POST',
      body: JSON.stringify({ texts }),
    })
  },
}

// Endpoints de API Externa
export const externalAPI = {
  create: async (apiData) => {
    return apiRequest('/api/external-apis', {
      method: 'POST',
      body: JSON.stringify(apiData),
    })
  },

  getAll: async () => {
    return apiRequest('/api/external-apis')
  },

  getById: async (id) => {
    return apiRequest(`/api/external-apis/${id}`)
  },

  update: async (id, apiData) => {
    return apiRequest(`/api/external-apis/${id}`, {
      method: 'PUT',
      body: JSON.stringify(apiData),
    })
  },

  delete: async (id) => {
    return apiRequest(`/api/external-apis/${id}`, { method: 'DELETE' })
  },

  test: async (id) => {
    return apiRequest(`/api/external-apis/${id}/test`, { method: 'POST' })
  },

  analyze: async (id) => {
    return apiRequest(`/api/external-apis/${id}/analyze`, { method: 'POST' })
  },
}

// Endpoints de Pagos
export const paymentsAPI = {
  getAll: async () => {
    return apiRequest('/api/payments')
  },

  getById: async (id) => {
    return apiRequest(`/api/payments/${id}`)
  },
}

// Endpoints de Datasets (Parte 1)
export const datasetsAPI = {
  upload: async (file) => {
    const token = getToken()
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_URL}/api/datasets/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Error ${response.status}` }))
      throw new Error(errorData.detail || errorData.message || `Error ${response.status}`)
    }

    return response.json()
  },

  analyzeBatch: async (texts) => {
    return apiRequest('/api/datasets/analyze-batch', {
      method: 'POST',
      body: JSON.stringify({ texts }),
    })
  },

  search: async (query, texts) => {
    return apiRequest('/api/datasets/search', {
      method: 'POST',
      body: JSON.stringify({ query, texts }),
    })
  },
}

// Endpoints de Optimización de Rutas (Parte 2)
export const routeOptimizationAPI = {
  optimize: async (points, algorithm = 'astar', startPoint = 0, saveRoute = false, routeName = null) => {
    return apiRequest('/api/route-optimization/optimize', {
      method: 'POST',
      body: JSON.stringify({
        points,
        algorithm,
        start_point: startPoint,
        save_route: saveRoute,
        route_name: routeName,
      }),
    })
  },

  applyAddress: async (address) => {
    return apiRequest('/api/route-optimization/apply-address', {
      method: 'POST',
      body: JSON.stringify({ address }),
    })
  },

  getRoutes: async () => {
    return apiRequest('/api/route-optimization')
  },

  getRoute: async (routeId) => {
    return apiRequest(`/api/route-optimization/${routeId}`)
  },

  deleteRoute: async (routeId) => {
    return apiRequest(`/api/route-optimization/${routeId}`, {
      method: 'DELETE',
    })
  },

  autocomplete: async (query) => {
    if (!query || query.length < 2) {
      return []
    }
    return apiRequest(`/api/route-optimization/autocomplete/search?query=${encodeURIComponent(query)}`)
  },
}

// Endpoints de Predicción de Ventas (Parte 3)
export const salesPredictionAPI = {
  upload: async (file) => {
    const token = getToken()
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_URL}/api/sales-prediction/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Error ${response.status}` }))
      throw new Error(errorData.detail || errorData.message || `Error ${response.status}`)
    }

    return response.json()
  },

  train: async (file, region = '', modelType = 'linear_regression') => {
    const token = getToken()
    const formData = new FormData()
    formData.append('file', file)
    
    let url = `${API_URL}/api/sales-prediction/train?model_type=${modelType}`
    if (region) {
      url += `&region=${encodeURIComponent(region)}`
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Error ${response.status}` }))
      throw new Error(errorData.detail || errorData.message || `Error ${response.status}`)
    }

    return response.json()
  },

  predict: async (region = null, producto = null, modelType = 'linear_regression', startDate, days = 30) => {
    return apiRequest('/api/sales-prediction/predict', {
      method: 'POST',
      body: JSON.stringify({
        region,
        producto,
        model_type: modelType,
        start_date: startDate,
        days,
      }),
    })
  },

  getHistoricalData: async (producto = null, region = null) => {
    let url = `${API_URL}/api/sales-prediction/historical-data`
    const params = []
    if (producto) params.push(`producto=${encodeURIComponent(producto)}`)
    if (region) params.push(`region=${encodeURIComponent(region)}`)
    if (params.length > 0) url += `?${params.join('&')}`

    return apiRequest(url.replace(API_URL, ''), {
      method: 'GET',
    })
  },
}


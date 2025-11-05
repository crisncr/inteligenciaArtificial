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
    const error = await response.json().catch(() => ({ detail: 'Error desconocido' }))
    throw new Error(error.detail || error.message || `Error ${response.status}`)
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
    const formData = new FormData()
    formData.append('username', email)
    formData.append('password', password)

    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        username: email,
        password: password,
      }),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Error desconocido' }))
      throw new Error(error.detail || `Error ${response.status}`)
    }

    const data = await response.json()
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
    return apiRequest('/api/auth/me')
  },

  forgotPassword: async (email) => {
    return apiRequest('/api/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email }),
    })
  },

  resetPassword: async (token, newPassword) => {
    return apiRequest('/api/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify({ token, new_password: newPassword }),
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


import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import AnalyzePanel from './components/AnalyzePanel'
import Features from './components/Features'
import Pricing from './components/Pricing'
import Footer from './components/Footer'
import CookieBar from './components/CookieBar'
import Login from './components/Login'
import Register from './components/Register'
import ForgotPassword from './components/ForgotPassword'
import VerifyEmail from './components/VerifyEmail'
import ResetPassword from './components/ResetPassword'
import LimitModal from './components/LimitModal'
import Dashboard from './components/Dashboard'
import ImageCarousel from './components/ImageCarousel'
import { authAPI, analysesAPI, getToken, removeToken } from './utils/api'

function App() {
  const [cookieAccepted, setCookieAccepted] = useState(false)
  const [history, setHistory] = useState([])
  const [reanalyzeText, setReanalyzeText] = useState(null)
  const [user, setUser] = useState(null)
  const [showLogin, setShowLogin] = useState(false)
  const [showRegister, setShowRegister] = useState(false)
  const [showForgotPassword, setShowForgotPassword] = useState(false)
  const [showLimitModal, setShowLimitModal] = useState(false)
  const [showVerifyEmail, setShowVerifyEmail] = useState(false)
  const [showResetPassword, setShowResetPassword] = useState(false)
  const [verificationToken, setVerificationToken] = useState(null)
  const [resetToken, setResetToken] = useState(null)
  const [freeAnalysesUsed, setFreeAnalysesUsed] = useState(0)
  const [loading, setLoading] = useState(true)

  // Verificar si hay token de verificación o reset en la URL
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const token = urlParams.get('token')
    const resetTokenParam = urlParams.get('reset-token')
    
    if (token) {
      setVerificationToken(token)
      setShowVerifyEmail(true)
      // Limpiar URL
      window.history.replaceState({}, document.title, window.location.pathname)
    } else if (resetTokenParam) {
      setResetToken(resetTokenParam)
      setShowResetPassword(true)
      // Limpiar URL
      window.history.replaceState({}, document.title, window.location.pathname)
    }
  }, [])

  // Cargar usuario desde API si hay token JWT
  useEffect(() => {
    const loadUser = async () => {
      const token = getToken()
      if (token) {
        try {
          const userData = await authAPI.getCurrentUser()
          setUser(userData)
          
          // Cargar solo análisis de API externa del usuario
          try {
            const analyses = await analysesAPI.getAll()
            // Filtrar solo análisis de API externa
            const apiAnalyses = analyses
              .filter(a => a.source === 'api_external')
              .map(a => ({
                text: a.text,
                sentiment: a.sentiment,
                score: a.score,
                emoji: a.emoji,
                source: a.source || 'api_external',
                external_api_id: a.external_api_id || null,
                timestamp: a.created_at
              }))
            setHistory(apiAnalyses)
          } catch (err) {
            console.error('Error al cargar análisis de API:', err)
          }
        } catch (err) {
          console.error('Error al cargar usuario:', err)
          // Token inválido, limpiar
          removeToken()
        }
      } else {
        // Sin token, cargar desde sessionStorage (usuarios no autenticados)
        // Limpiar contador antiguo de localStorage si existe (migración)
        if (localStorage.getItem('freeAnalysesUsed')) {
          localStorage.removeItem('freeAnalysesUsed')
        }
        
        const savedAnalyses = sessionStorage.getItem('freeAnalysesUsed')
        if (savedAnalyses) {
          setFreeAnalysesUsed(parseInt(savedAnalyses, 10))
        }

        const savedHistory = localStorage.getItem('sentimentHistory')
        if (savedHistory) {
          try {
            setHistory(JSON.parse(savedHistory))
          } catch (err) {
            console.error('Error al cargar historial:', err)
          }
        }
      }
      setLoading(false)
    }
    
    loadUser()
  }, [])

  // Guardar historial en localStorage cuando cambie
  useEffect(() => {
    if (history.length > 0) {
      localStorage.setItem('sentimentHistory', JSON.stringify(history))
    }
  }, [history])

  // Calcular análisis gratuitos restantes
  // Si no hay usuario o tiene plan free, aplicar límite de 3
  const freeAnalysesLeft = !user || user.plan === 'free' ? Math.max(0, 3 - freeAnalysesUsed) : Infinity

  const handleLogin = async (loggedUser) => {
    setUser(loggedUser)
    setShowLogin(false)
    setShowRegister(false)
    
    // Limpiar historial local cuando el usuario inicia sesión
    // Solo cargar análisis de API externa desde la base de datos
    setHistory([])
    localStorage.removeItem('sentimentHistory')
    sessionStorage.removeItem('freeAnalysesUsed')
    setFreeAnalysesUsed(0)
    
    // Cargar solo análisis de API externa del usuario
    try {
      const analyses = await analysesAPI.getAll()
      // Filtrar solo análisis de API externa
      const apiAnalyses = analyses
        .filter(a => a.source === 'api_external')
        .map(a => ({
          text: a.text,
          sentiment: a.sentiment,
          score: a.score,
          emoji: a.emoji,
          source: a.source || 'api_external',
          external_api_id: a.external_api_id || null,
          timestamp: a.created_at
        }))
      setHistory(apiAnalyses)
    } catch (err) {
      console.error('Error al cargar análisis de API:', err)
    }
  }

  const handleRegister = (newUser) => {
    setUser(newUser)
    setShowLogin(false)
    setShowRegister(false)
  }

  const handleLogout = async () => {
    try {
      await authAPI.logout()
    } catch (err) {
      console.error('Error al cerrar sesión:', err)
    } finally {
      setUser(null)
      removeToken()
      setFreeAnalysesUsed(0)
      sessionStorage.removeItem('freeAnalysesUsed')
      setHistory([])
      localStorage.removeItem('sentimentHistory')
    }
  }

  const handleAnalyze = async (result) => {
    // Si el usuario está autenticado, solo recargar análisis de API externa
    if (user) {
      // Recargar solo análisis de API externa desde API
      try {
        const analyses = await analysesAPI.getAll()
        // Filtrar solo análisis de API externa
        const apiAnalyses = analyses
          .filter(a => a.source === 'api_external')
          .map(a => ({
            text: a.text,
            sentiment: a.sentiment,
            score: a.score,
            emoji: a.emoji,
            source: a.source || 'api_external',
            external_api_id: a.external_api_id || null,
            timestamp: a.created_at
          }))
        setHistory(apiAnalyses)
      } catch (err) {
        console.error('Error al cargar análisis de API:', err)
      }
    } else {
      // Usuario no autenticado: guardar en localStorage (solo para demo)
      setHistory(prev => [{ ...result, timestamp: new Date().toISOString() }, ...prev].slice(0, 50))
      
      // Incrementar contador de análisis gratuitos
      const newCount = freeAnalysesUsed + 1
      setFreeAnalysesUsed(newCount)
      sessionStorage.setItem('freeAnalysesUsed', newCount.toString())
      
      // Si alcanzó el límite, mostrar modal
      if (newCount >= 3) {
        setShowLimitModal(true)
      }
    }
  }

  const handleReanalyze = (text) => {
    // Scroll al panel de análisis
    const element = document.getElementById('analizar')
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
      // Actualizar el texto para re-analizar
      setTimeout(() => {
        setReanalyzeText(text)
        setTimeout(() => setReanalyzeText(null), 100)
      }, 500)
    }
  }

  const handleClearHistory = () => {
    if (window.confirm('¿Estás seguro de que quieres limpiar todo el historial?')) {
      setHistory([])
      localStorage.removeItem('sentimentHistory')
    }
  }

  const handleLimitReached = () => {
    setShowLimitModal(true)
  }

  const handleSelectPlan = async (planId) => {
    if (!user) {
      setShowLogin(true)
      return
    }

    try {
      // Aquí deberías llamar a la API para actualizar el plan del usuario
      // Por ahora, actualizamos localmente
      const updatedUser = { ...user, plan: planId }
      setUser(updatedUser)
      
      // Actualizar en localStorage
      const users = JSON.parse(localStorage.getItem('users') || '[]')
      const userIndex = users.findIndex(u => u.id === user.id)
      if (userIndex !== -1) {
        users[userIndex] = updatedUser
        localStorage.setItem('users', JSON.stringify(users))
        localStorage.setItem('currentUser', JSON.stringify(updatedUser))
      }
      
      // El estado del usuario ya se actualizó con setUser

      setShowLimitModal(false)
      const planNames = { free: 'Básico', pro: 'Pro', enterprise: 'Enterprise' }
      alert(`Plan ${planNames[planId] || planId} seleccionado correctamente`)
    } catch (err) {
      console.error('Error al actualizar plan:', err)
      alert('Error al actualizar el plan. Por favor, intenta de nuevo.')
    }
  }

  const handleLoginClick = () => {
    setShowLogin(true)
    setShowRegister(false)
    setShowForgotPassword(false)
  }

  const handleRegisterClick = () => {
    setShowRegister(true)
    setShowLogin(false)
    setShowForgotPassword(false)
  }

  const handleForgotPasswordClick = () => {
    setShowForgotPassword(true)
    setShowLogin(false)
    setShowRegister(false)
  }

  const handleUserUpdate = (updatedUser) => {
    setUser(updatedUser)
  }

  return (
    <>
      <Navbar 
        user={user} 
        onLoginClick={handleLoginClick}
        onRegisterClick={handleRegisterClick}
        onLogout={handleLogout}
      />
      {user ? (
        // Dashboard para usuarios autenticados
        <Dashboard
          user={user}
          history={history}
          onReanalyze={handleReanalyze}
          onClearHistory={handleClearHistory}
          onSelectPlan={handleSelectPlan}
          onUserUpdate={handleUserUpdate}
          onAnalyze={handleAnalyze}
          reanalyzeText={reanalyzeText}
          freeAnalysesLeft={freeAnalysesLeft}
          onLimitReached={handleLimitReached}
        />
      ) : (
        // Página pública para usuarios no autenticados
        <>
          {/* Hero Section con Carrusel */}
          <section className="hero-section">
            <div className="hero-carousel">
              <ImageCarousel 
                images={[
                  '/images/1711396661636.png',
                  '/images/diferencia-entre-emocion-y-sentimiento.jpg',
                  '/images/istockphoto-1409988922-612x612.jpg',
                ]}
                autoPlay={true}
                interval={4000}
              />
              <div className="hero-overlay"></div>
            </div>
          </section>
          
          <div className="container">
            <AnalyzePanel 
              onAnalyze={handleAnalyze} 
              reanalyzeText={reanalyzeText}
              user={user}
              freeAnalysesLeft={freeAnalysesLeft}
              onLimitReached={handleLimitReached}
            />
            <Features />
            <Pricing 
              user={user}
              onSelectPlan={handleSelectPlan}
              onLoginRequired={handleLoginClick}
            />
            <Footer />
          </div>
        </>
      )}
      {!cookieAccepted && <CookieBar onAccept={() => setCookieAccepted(true)} />}
      
      {showLogin && (
        <Login 
          onLogin={handleLogin}
          onSwitchToRegister={handleRegisterClick}
          onForgotPassword={handleForgotPasswordClick}
          onClose={() => setShowLogin(false)}
        />
      )}
      
      {showRegister && (
        <Register 
          onRegister={handleRegister}
          onSwitchToLogin={handleLoginClick}
          onClose={() => setShowRegister(false)}
        />
      )}

      {showForgotPassword && (
        <ForgotPassword 
          onBack={handleLoginClick}
          onClose={() => setShowForgotPassword(false)}
        />
      )}

      {showLimitModal && (
        <LimitModal 
          onClose={() => setShowLimitModal(false)}
          onSelectPlan={() => {
            setShowLimitModal(false)
            const element = document.getElementById('precio')
            if (element) {
              element.scrollIntoView({ behavior: 'smooth', block: 'start' })
            }
          }}
          isAuthenticated={!!user}
        />
      )}

      {showVerifyEmail && (
        <VerifyEmail 
          token={verificationToken}
          onClose={() => {
            setShowVerifyEmail(false)
            setVerificationToken(null)
            // Limpiar también la URL para evitar que vuelva a aparecer
            window.history.replaceState({}, document.title, window.location.pathname)
          }}
        />
      )}

      {showResetPassword && (
        <ResetPassword 
          token={resetToken}
          onSuccess={() => {
            setShowResetPassword(false)
            setResetToken(null)
          }}
          onClose={() => {
            setShowResetPassword(false)
            setResetToken(null)
            window.history.replaceState({}, document.title, window.location.pathname)
          }}
        />
      )}
    </>
  )
}

export default App


import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import AnalyzePanel from './components/AnalyzePanel'
import Features from './components/Features'
import History from './components/History'
import Stats from './components/Stats'
import Pricing from './components/Pricing'
import Footer from './components/Footer'
import CookieBar from './components/CookieBar'
import Login from './components/Login'
import Register from './components/Register'
import ForgotPassword from './components/ForgotPassword'
import VerifyEmail from './components/VerifyEmail'
import LimitModal from './components/LimitModal'
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
  const [verificationToken, setVerificationToken] = useState(null)
  const [freeAnalysesUsed, setFreeAnalysesUsed] = useState(0)
  const [loading, setLoading] = useState(true)

  // Verificar si hay token de verificación en la URL
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const token = urlParams.get('token')
    if (token) {
      setVerificationToken(token)
      setShowVerifyEmail(true)
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
          
          // Cargar análisis del usuario desde API
          try {
            const analyses = await analysesAPI.getAll()
            setHistory(analyses.map(a => ({
              text: a.text,
              sentiment: a.sentiment,
              score: a.score,
              emoji: a.emoji,
              timestamp: a.created_at
            })))
          } catch (err) {
            console.error('Error al cargar análisis:', err)
          }
        } catch (err) {
          console.error('Error al cargar usuario:', err)
          // Token inválido, limpiar
          removeToken()
        }
      } else {
        // Sin token, cargar desde localStorage (usuarios no autenticados)
        const savedAnalyses = localStorage.getItem('freeAnalysesUsed')
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

  const handleLogin = (loggedUser) => {
    setUser(loggedUser)
    setShowLogin(false)
    setShowRegister(false)
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
      localStorage.removeItem('freeAnalysesUsed')
      setHistory([])
      localStorage.removeItem('sentimentHistory')
    }
  }

  const handleAnalyze = async (result) => {
    // Si el usuario está autenticado, el análisis ya se guardó en BD por AnalyzePanel
    // Solo actualizar el historial local para mostrar
    if (user) {
      // Recargar análisis desde API
      try {
        const analyses = await analysesAPI.getAll()
        setHistory(analyses.map(a => ({
          text: a.text,
          sentiment: a.sentiment,
          score: a.score,
          emoji: a.emoji,
          timestamp: a.created_at
        })))
      } catch (err) {
        console.error('Error al cargar análisis:', err)
        // Si falla, agregar al historial local
        setHistory(prev => [{ ...result, timestamp: new Date().toISOString() }, ...prev].slice(0, 50))
      }
    } else {
      // Usuario no autenticado: guardar en localStorage
      setHistory(prev => [{ ...result, timestamp: new Date().toISOString() }, ...prev].slice(0, 50))
      
      // Incrementar contador de análisis gratuitos
      const newCount = freeAnalysesUsed + 1
      setFreeAnalysesUsed(newCount)
      localStorage.setItem('freeAnalysesUsed', newCount.toString())
      
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

  const handleSelectPlan = (planName) => {
    if (!user) {
      setShowLogin(true)
      return
    }

    // Actualizar plan del usuario
    const updatedUser = { ...user, plan: planName }
    setUser(updatedUser)
    
    // Actualizar en localStorage
    const users = JSON.parse(localStorage.getItem('users') || '[]')
    const userIndex = users.findIndex(u => u.id === user.id)
    if (userIndex !== -1) {
      users[userIndex] = updatedUser
      localStorage.setItem('users', JSON.stringify(users))
      localStorage.setItem('currentUser', JSON.stringify(updatedUser))
    }

    setShowLimitModal(false)
    alert(`Plan ${planName} seleccionado correctamente`)
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

  return (
    <>
      <Navbar 
        user={user} 
        onLoginClick={handleLoginClick}
        onRegisterClick={handleRegisterClick}
        onLogout={handleLogout}
      />
      <div className="container">
        <Hero />
        <AnalyzePanel 
          onAnalyze={handleAnalyze} 
          reanalyzeText={reanalyzeText}
          user={user}
          freeAnalysesLeft={freeAnalysesLeft}
          onLimitReached={handleLimitReached}
        />
        {history.length > 0 && (
          <>
            <History 
              history={history} 
              onReanalyze={handleReanalyze}
              onClearHistory={handleClearHistory}
            />
            <Stats history={history} />
          </>
        )}
        <Features />
        <Pricing 
          user={user}
          onSelectPlan={handleSelectPlan}
          onLoginRequired={handleLoginClick}
        />
        <Footer />
      </div>
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
    </>
  )
}

export default App


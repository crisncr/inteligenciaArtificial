import { useState } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import AnalyzePanel from './components/AnalyzePanel'
import Features from './components/Features'
import Footer from './components/Footer'
import CookieBar from './components/CookieBar'

function App() {
  const [cookieAccepted, setCookieAccepted] = useState(false)

  return (
    <>
      <Navbar />
      <div className="container">
        <Hero />
        <AnalyzePanel />
        <Features />
        <Footer />
      </div>
      {!cookieAccepted && <CookieBar onAccept={() => setCookieAccepted(true)} />}
    </>
  )
}

export default App


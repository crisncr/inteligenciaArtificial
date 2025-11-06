import { useState } from 'react'
import AnalyzePanel from './AnalyzePanel'
import Pricing from './Pricing'
import Features from './Features'
import History from './History'
import Stats from './Stats'
import Payments from './Payments'
import Settings from './Settings'
import ExternalAPI from './ExternalAPI'
import Diagnostics from './Diagnostics'

function DashboardContent({ 
  activeSection, 
  user, 
  history, 
  onReanalyze, 
  onClearHistory,
  onSelectPlan,
  onUserUpdate,
  onAnalyze,
  reanalyzeText,
  freeAnalysesLeft,
  onLimitReached
}) {
  const [historyFilter, setHistoryFilter] = useState('all')

  const filteredHistory = historyFilter === 'all' 
    ? history 
    : history.filter(item => {
        if (historyFilter === 'positivo') return item.sentiment === 'positivo'
        if (historyFilter === 'negativo') return item.sentiment === 'negativo'
        if (historyFilter === 'neutral') return item.sentiment === 'neutral' || item.sentiment === 'moderado/neutral'
        return true
      })

  const renderContent = () => {
    switch (activeSection) {
      case 'inicio':
        return (
          <>
            <AnalyzePanel 
              onAnalyze={onAnalyze} 
              reanalyzeText={reanalyzeText}
              user={user}
              freeAnalysesLeft={freeAnalysesLeft}
              onLimitReached={onLimitReached}
            />
            <Features />
            <Pricing 
              user={user}
              onSelectPlan={onSelectPlan}
            />
          </>
        )
      case 'pagos':
        return <Payments user={user} />
      case 'historial':
        return (
          <History 
            history={filteredHistory} 
            onReanalyze={onReanalyze}
            onClearHistory={onClearHistory}
            filter={historyFilter}
            onFilterChange={setHistoryFilter}
          />
        )
      case 'estadisticas':
        return <Stats history={history} />
      case 'api-externa':
        return <ExternalAPI user={user} onAnalyze={() => {
          // Recargar historial después de analizar
          if (onAnalyze) {
            onAnalyze()
          }
        }} />
      case 'diagnosticos':
        return <Diagnostics user={user} history={history} onReanalyze={onReanalyze} />
      case 'ajustes':
        return <Settings user={user} onUserUpdate={onUserUpdate} />
      default:
        return (
          <>
            <Features />
            <Pricing 
              user={user}
              onSelectPlan={onSelectPlan}
            />
          </>
        )
    }
  }

  return (
    <div className="dashboard-content">
      {renderContent()}
    </div>
  )
}

export default DashboardContent


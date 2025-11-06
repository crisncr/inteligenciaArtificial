import { useState } from 'react'
import DashboardHome from './DashboardHome'
import History from './History'
import Stats from './Stats'
import PaymentsHistory from './PaymentsHistory'
import Plans from './Plans'
import Settings from './Settings'
import ExternalAPI from './ExternalAPI'
import Diagnostics from './Diagnostics'
import Support from './Support'

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
        return <DashboardHome user={user} />
      case 'pagos':
        return <PaymentsHistory user={user} />
      case 'planes':
        return <Plans user={user} onSelectPlan={onSelectPlan} />
      case 'historial':
        // Filtrar solo análisis de API externa
        const apiHistory = filteredHistory.filter(item => item.source === 'api_external')
        return (
          <History 
            history={apiHistory} 
            onReanalyze={onReanalyze}
            onClearHistory={onClearHistory}
            filter={historyFilter}
            onFilterChange={setHistoryFilter}
          />
        )
      case 'estadisticas':
        // Filtrar solo análisis de API externa
        const apiStatsHistory = history.filter(item => item.source === 'api_external')
        return <Stats history={apiStatsHistory} />
      case 'api-externa':
        return <ExternalAPI user={user} onAnalyze={() => {
          // Recargar historial después de analizar
          if (onAnalyze) {
            onAnalyze()
          }
        }} />
      case 'diagnosticos':
        // Filtrar solo análisis de API externa
        const apiDiagnosticsHistory = history.filter(item => item.source === 'api_external')
        return <Diagnostics user={user} history={apiDiagnosticsHistory} onReanalyze={onReanalyze} />
      case 'ajustes':
        return <Settings user={user} onUserUpdate={onUserUpdate} />
      case 'soporte':
        return <Support user={user} />
      default:
        return <DashboardHome user={user} />
    }
  }

  return (
    <div className="dashboard-content">
      {renderContent()}
    </div>
  )
}

export default DashboardContent


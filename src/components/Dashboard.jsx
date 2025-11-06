import { useState } from 'react'
import DashboardSidebar from './DashboardSidebar'
import DashboardContent from './DashboardContent'

function Dashboard({ 
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
  const [activeSection, setActiveSection] = useState('inicio')

  return (
    <div className="dashboard">
      <DashboardSidebar 
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />
      <div className="dashboard-main">
        <DashboardContent
          activeSection={activeSection}
          user={user}
          history={history}
          onReanalyze={onReanalyze}
          onClearHistory={onClearHistory}
          onSelectPlan={onSelectPlan}
          onUserUpdate={onUserUpdate}
          onAnalyze={onAnalyze}
          reanalyzeText={reanalyzeText}
          freeAnalysesLeft={freeAnalysesLeft}
          onLimitReached={onLimitReached}
        />
      </div>
    </div>
  )
}

export default Dashboard


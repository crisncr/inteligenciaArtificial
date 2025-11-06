import { useState } from 'react'

function DashboardSidebar({ activeSection, onSectionChange }) {
  const menuItems = [
    { id: 'inicio', label: 'Inicio', icon: '🏠' },
    { id: 'pagos', label: 'Pagos', icon: '💳' },
    { id: 'planes', label: 'Planes', icon: '📦' },
    { id: 'historial', label: 'Historial', icon: '📋' },
    { id: 'estadisticas', label: 'Estadísticas', icon: '📊' },
    { id: 'api-externa', label: 'API Externa', icon: '🔌' },
    { id: 'diagnosticos', label: 'Diagnósticos', icon: '🔍' },
    { id: 'soporte', label: 'Ayuda', icon: '💬' },
    { id: 'ajustes', label: 'Ajustes', icon: '⚙️' }
  ]

  return (
    <aside className="dashboard-sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <img src="/favicon.svg" alt="logo" width="24" height="24" />
          <span>Sentimetría</span>
        </div>
      </div>
      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <button
            key={item.id}
            className={`sidebar-item ${activeSection === item.id ? 'active' : ''}`}
            onClick={() => onSectionChange(item.id)}
          >
            <span className="sidebar-icon">{item.icon}</span>
            <span className="sidebar-label">{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  )
}

export default DashboardSidebar


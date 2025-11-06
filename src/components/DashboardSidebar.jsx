import { useState } from 'react'
import { getSidebarItems } from '../utils/planFeatures'

function DashboardSidebar({ activeSection, onSectionChange, user }) {
  const plan = user?.plan || 'free'
  const menuItems = getSidebarItems(plan)

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


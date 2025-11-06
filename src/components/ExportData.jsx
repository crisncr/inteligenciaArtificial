import { useState } from 'react'
import { analysesAPI } from '../utils/api'

function ExportData({ user, history }) {
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState({ type: '', text: '' })

  const handleExport = async (format) => {
    setLoading(true)
    setMessage({ type: '', text: '' })

    try {
      // Obtener todos los análisis
      const analyses = await analysesAPI.getAll()
      const apiAnalyses = analyses.filter(a => a.source === 'api_external')

      let content = ''
      let filename = ''
      let mimeType = ''

      if (format === 'csv') {
        // Exportar a CSV
        const headers = ['Fecha', 'Texto', 'Sentimiento', 'Score', 'Emoji']
        const rows = apiAnalyses.map(a => [
          new Date(a.created_at).toLocaleString('es-ES'),
          `"${a.text.replace(/"/g, '""')}"`,
          a.sentiment,
          a.score,
          a.emoji
        ])
        content = [headers, ...rows].map(row => row.join(',')).join('\n')
        filename = `analisis-sentimientos-${new Date().toISOString().split('T')[0]}.csv`
        mimeType = 'text/csv'
      } else if (format === 'json') {
        // Exportar a JSON
        content = JSON.stringify(apiAnalyses, null, 2)
        filename = `analisis-sentimientos-${new Date().toISOString().split('T')[0]}.json`
        mimeType = 'application/json'
      }

      // Crear blob y descargar
      const blob = new Blob([content], { type: mimeType })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)

      setMessage({ type: 'success', text: `Datos exportados correctamente como ${format.toUpperCase()}` })
    } catch (err) {
      setMessage({ type: 'error', text: 'Error al exportar datos: ' + err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="export-data-panel">
      <h2>Exportar Datos</h2>
      <p className="subtitle">
        Exporta tus análisis de sentimientos en diferentes formatos
      </p>

      {message.text && (
        <div className={`message ${message.type}`}>
          {message.text}
        </div>
      )}

      <div className="export-options">
        <div className="export-card">
          <div className="export-icon">📄</div>
          <h3>Exportar a CSV</h3>
          <p>Ideal para análisis en Excel o Google Sheets</p>
          <button 
            className="btn" 
            onClick={() => handleExport('csv')}
            disabled={loading}
          >
            {loading ? 'Exportando...' : 'Exportar CSV'}
          </button>
        </div>

        <div className="export-card">
          <div className="export-icon">📋</div>
          <h3>Exportar a JSON</h3>
          <p>Formato estructurado para desarrolladores</p>
          <button 
            className="btn" 
            onClick={() => handleExport('json')}
            disabled={loading}
          >
            {loading ? 'Exportando...' : 'Exportar JSON'}
          </button>
        </div>
      </div>

      {history && history.length > 0 && (
        <div className="export-stats">
          <p>Total de análisis disponibles para exportar: <strong>{history.length}</strong></p>
        </div>
      )}
    </section>
  )
}

export default ExportData


import { useState } from 'react'
import { salesPredictionAPI } from '../utils/api'

function SalesPrediction({ user }) {
  const [salesFile, setSalesFile] = useState(null)
  const [salesData, setSalesData] = useState(null)
  const [region, setRegion] = useState('')
  const [modelType, setModelType] = useState('linear_regression')
  const [trainResult, setTrainResult] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [startDate, setStartDate] = useState('')
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setSalesFile(file)
    setLoading(true)

    try {
      const result = await salesPredictionAPI.upload(file)
      setSalesData(result)
      if (result.regions && result.regions.length > 0) {
        setRegion(result.regions[0])
      }
    } catch (err) {
      console.error('Error al cargar datos:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleTrain = async () => {
    if (!salesFile) {
      alert('Primero carga un archivo con datos históricos')
      return
    }

    setTraining(true)
    setTrainResult(null)

    try {
      const result = await salesPredictionAPI.train(salesFile, region, modelType)
      setTrainResult(result)
    } catch (err) {
      console.error('Error al entrenar modelo:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setTraining(false)
    }
  }

  const handlePredict = async () => {
    if (!startDate) {
      alert('Selecciona una fecha de inicio')
      return
    }

    if (!trainResult) {
      alert('Primero entrena el modelo')
      return
    }

    setLoading(true)
    setPredictions(null)

    try {
      const result = await salesPredictionAPI.predict(region, modelType, startDate, days)
      setPredictions(result)
    } catch (err) {
      console.error('Error al predecir:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Obtener fecha de mañana por defecto
  const getTomorrowDate = () => {
    const tomorrow = new Date()
    tomorrow.setDate(tomorrow.getDate() + 1)
    return tomorrow.toISOString().split('T')[0]
  }

  return (
    <section className="dashboard-section">
      <div className="section-header">
        <h1>Predicción de Ventas</h1>
        <p className="section-subtitle">
          Parte 3: Predicción de ventas por región usando IA (Regresión Lineal / Red Neuronal)
        </p>
      </div>

      {/* Carga de Datos */}
      <div className="api-form" style={{ marginBottom: '20px' }}>
        <h3>Cargar Datos Históricos</h3>
        <div className="form-field">
          <label htmlFor="sales-file">Archivo CSV con datos de ventas</label>
          <input
            type="file"
            id="sales-file"
            accept=".csv"
            onChange={handleFileUpload}
            className="form-input"
            disabled={loading}
          />
          <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
            El archivo debe contener las columnas: fecha, region, ventas
          </small>
        </div>

        {salesData && (
          <div className="stats-grid" style={{ marginTop: '20px' }}>
            <div className="stat-card">
              <div className="stat-value">{salesData.total}</div>
              <div className="stat-label">Registros Cargados</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{salesData.regions?.length || 0}</div>
              <div className="stat-label">Regiones</div>
            </div>
          </div>
        )}
      </div>

      {/* Selector de Región y Modelo */}
      {salesData && (
        <div className="api-form" style={{ marginBottom: '20px' }}>
          <h3>Configuración del Modelo</h3>
          <div className="form-row">
            <div className="form-field">
              <label htmlFor="region">Región</label>
              <select
                id="region"
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                className="form-input"
              >
                {salesData.regions?.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="model-type">Tipo de Modelo</label>
              <select
                id="model-type"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="form-input"
              >
                <option value="linear_regression">Regresión Lineal</option>
                <option value="neural_network">Red Neuronal</option>
              </select>
            </div>
          </div>
          <button 
            className="btn" 
            onClick={handleTrain} 
            disabled={training || !region}
          >
            {training ? 'Entrenando modelo...' : 'Entrenar Modelo'}
          </button>
        </div>
      )}

      {/* Resultados del Entrenamiento */}
      {trainResult && (
        <div className="stats-panel" style={{ marginBottom: '20px' }}>
          <h3>Resultados del Entrenamiento</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{trainResult.model_type}</div>
              <div className="stat-label">Modelo</div>
            </div>
            {trainResult.r2_score && (
              <div className="stat-card">
                <div className="stat-value">{trainResult.r2_score.toFixed(3)}</div>
                <div className="stat-label">R² Score</div>
              </div>
            )}
            {trainResult.mse && (
              <div className="stat-card">
                <div className="stat-value">{trainResult.mse.toFixed(2)}</div>
                <div className="stat-label">MSE</div>
              </div>
            )}
          </div>
          <div className="message" style={{ marginTop: '15px', background: 'rgba(110, 139, 255, 0.1)', padding: '15px', borderRadius: '8px' }}>
            <p><strong>Justificación:</strong> {trainResult.justification}</p>
          </div>
        </div>
      )}

      {/* Predicción */}
      {trainResult && (
        <div className="api-form" style={{ marginBottom: '20px' }}>
          <h3>Predecir Ventas Futuras</h3>
          <div className="form-row">
            <div className="form-field">
              <label htmlFor="start-date">Fecha de Inicio</label>
              <input
                type="date"
                id="start-date"
                value={startDate || getTomorrowDate()}
                onChange={(e) => setStartDate(e.target.value)}
                className="form-input"
              />
            </div>
            <div className="form-field">
              <label htmlFor="days">Días a Predecir</label>
              <input
                type="number"
                id="days"
                value={days}
                onChange={(e) => setDays(parseInt(e.target.value) || 30)}
                className="form-input"
                min="1"
                max="365"
              />
            </div>
          </div>
          <button 
            className="btn" 
            onClick={handlePredict} 
            disabled={loading || !startDate}
          >
            {loading ? 'Prediciendo...' : 'Predecir Ventas'}
          </button>
        </div>
      )}

      {/* Resultados de la Predicción */}
      {predictions && (
        <div className="stats-panel">
          <h3>Predicciones de Ventas</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{predictions.summary?.total_predicted?.toFixed(2) || 0}</div>
              <div className="stat-label">Total Predicho</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{predictions.summary?.average_daily?.toFixed(2) || 0}</div>
              <div className="stat-label">Promedio Diario</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{predictions.summary?.days || 0}</div>
              <div className="stat-label">Días</div>
            </div>
          </div>

          {predictions.predictions && predictions.predictions.length > 0 && (
            <div className="history-list" style={{ marginTop: '20px' }}>
              <h3>Predicciones Diarias</h3>
              {predictions.predictions.slice(0, 10).map((pred, index) => (
                <div key={index} className="history-item">
                  <div className="history-item-header">
                    <span><strong>{pred.fecha}:</strong> {pred.ventas_predichas.toFixed(2)} ventas</span>
                  </div>
                </div>
              ))}
              {predictions.predictions.length > 10 && (
                <p style={{ marginTop: '10px', color: 'var(--text-secondary)' }}>
                  ... y {predictions.predictions.length - 10} días más
                </p>
              )}
            </div>
          )}
        </div>
      )}

    </section>
  )
}

export default SalesPrediction


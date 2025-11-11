import { useState, useEffect, useRef } from 'react'
import { salesPredictionAPI } from '../utils/api'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { Line } from 'react-chartjs-2'

// Registrar componentes de Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

function SalesPrediction({ user }) {
  const [salesFile, setSalesFile] = useState(null)
  const [salesData, setSalesData] = useState(null)
  const [region, setRegion] = useState('')
  const [selectedProducts, setSelectedProducts] = useState([])
  const [trainResult, setTrainResult] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [historicalData, setHistoricalData] = useState(null)
  const [startDate, setStartDate] = useState('')
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)
  const [chart1Products, setChart1Products] = useState([])
  const [chart2Product, setChart2Product] = useState('')

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
      if (result.products && result.products.length > 0) {
        setSelectedProducts([result.products[0]])
        setChart1Products([result.products[0]])
        setChart2Product(result.products[0])
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
    setPredictions(null)
    setHistoricalData(null)

    try {
      const result = await salesPredictionAPI.train(salesFile, region, 'linear_regression')
      setTrainResult(result)
      
      // Cargar datos históricos después del entrenamiento
      await loadHistoricalData()
    } catch (err) {
      console.error('Error al entrenar modelo:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setTraining(false)
    }
  }

  const loadHistoricalData = async () => {
    if (!salesData) return
    
    try {
      const data = await salesPredictionAPI.getHistoricalData()
      setHistoricalData(data)
    } catch (err) {
      console.error('Error al cargar datos históricos:', err)
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
      const result = await salesPredictionAPI.predict(
        region || null,
        chart2Product || null,
        'linear_regression',
        startDate,
        days
      )
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

  // Calcular regresión lineal para gráfico
  const calculateLinearRegression = (x, y) => {
    const n = x.length
    const sumX = x.reduce((a, b) => a + b, 0)
    const sumY = y.reduce((a, b) => a + b, 0)
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0)
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0)
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
    const intercept = (sumY - slope * sumX) / n
    
    return { slope, intercept }
  }

  // Preparar datos para Gráfico 1: Ventas por producto (filtrado por región seleccionada)
  const prepareChart1Data = () => {
    if (!historicalData || !chart1Products.length || !region) return null

    // Filtrar datos por la región seleccionada
    const filteredData = historicalData.historical_data.filter(d => d.region === region)
    
    const datasets = []
    const productAverages = []

    chart1Products.forEach((producto, idx) => {
      const colors = [
        'rgba(54, 162, 235, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)'
      ]
      const color = colors[idx % colors.length]

      // Obtener ventas de este producto en la región seleccionada
      const ventas = filteredData
        .filter(d => d.producto === producto)
        .map(d => d.ventas)

      // Calcular promedio de ventas para este producto
      const promedio = ventas.length > 0 
        ? ventas.reduce((a, b) => a + b, 0) / ventas.length 
        : 0

      productAverages.push(promedio)
    })

    // Calcular regresión lineal sobre los productos
    const x = chart1Products.map((_, i) => i)
    const y = productAverages
    const regression = calculateLinearRegression(x, y)
    const regressionLine = x.map(xi => regression.slope * xi + regression.intercept)

    datasets.push({
      label: `Ventas (Datos) - ${region}`,
      data: productAverages,
      borderColor: 'rgba(54, 162, 235, 1)',
      backgroundColor: 'rgba(54, 162, 235, 0.2)',
      fill: false,
      tension: 0.1
    })

    datasets.push({
      label: `Ventas (Regresión Lineal) - ${region}`,
      data: regressionLine,
      borderColor: 'rgba(255, 99, 132, 1)',
      backgroundColor: 'transparent',
      borderDash: [5, 5],
      fill: false,
      tension: 0.1,
      pointRadius: 0
    })

    return {
      labels: chart1Products,
      datasets
    }
  }

  // Preparar datos para Gráfico 2: Predicción diaria (usando región seleccionada)
  const prepareChart2Data = () => {
    if (!predictions || !chart2Product || !region) return null

    // Filtrar por producto y la región seleccionada al entrenar
    const filtered = predictions.predictions.filter(
      p => p.producto === chart2Product && p.region === region
    )

    if (filtered.length === 0) return null

    // Ordenar por fecha
    const data = filtered
      .map(p => ({
        fecha: p.fecha,
        ventas: p.ventas_predichas
      }))
      .sort((a, b) => new Date(a.fecha) - new Date(b.fecha))

    const labels = data.map(d => d.fecha)
    const ventas = data.map(d => d.ventas)

    // Calcular regresión lineal
    const x = labels.map((_, i) => i)
    const y = ventas
    const regression = calculateLinearRegression(x, y)
    const regressionLine = x.map(xi => regression.slope * xi + regression.intercept)

    const datasets = [
      {
        label: `${chart2Product} - ${region} (Predicción)`,
        data: ventas,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        fill: false,
        tension: 0.1
      },
      {
        label: `${chart2Product} - ${region} (Regresión Lineal)`,
        data: regressionLine,
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'transparent',
        borderDash: [5, 5],
        fill: false,
        tension: 0.1,
        pointRadius: 0
      }
    ]

    return {
      labels,
      datasets
    }
  }

  const chart1Data = prepareChart1Data()
  const chart2Data = prepareChart2Data()

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Gráfico de Regresión Lineal'
      }
    },
    scales: {
      y: {
        beginAtZero: false
      }
    }
  }

  useEffect(() => {
    if (salesData && !historicalData) {
      loadHistoricalData()
    }
  }, [salesData])

  return (
    <section className="dashboard-section">
      <div className="section-header">
        <h1>Predicción de Ventas</h1>
        <p className="section-subtitle">
          Predicción de ventas por producto y región usando IA (Regresión Lineal)
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
            El archivo debe contener las columnas: fecha, region, producto, valor, ventas
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
            <div className="stat-card">
              <div className="stat-value">{salesData.products?.length || 0}</div>
              <div className="stat-label">Productos</div>
            </div>
          </div>
        )}
      </div>

      {/* Configuración del Modelo */}
      {salesData && (
        <div className="api-form" style={{ marginBottom: '20px' }}>
          <h3>Configuración del Modelo</h3>
          <div className="form-field">
            <label htmlFor="region">Región (opcional, para filtrar entrenamiento)</label>
            <select
              id="region"
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="form-input"
            >
              <option value="">Todas las regiones</option>
              {salesData.regions?.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>
          <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px', marginBottom: '15px' }}>
            Modelo: Regresión Lineal (entrenará un modelo por cada combinación producto-región)
          </small>
          <button 
            className="btn" 
            onClick={handleTrain} 
            disabled={training}
          >
            {training ? 'Entrenando modelos...' : 'Entrenar Modelos'}
          </button>
        </div>
      )}

      {/* Resultados del Entrenamiento */}
      {trainResult && (
        <div className="stats-panel" style={{ marginBottom: '20px' }}>
          <h3>Resultados del Entrenamiento</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{trainResult.products_trained || 0}</div>
              <div className="stat-label">Modelos Entrenados</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{trainResult.average_r2_score?.toFixed(3) || 'N/A'}</div>
              <div className="stat-label">R² Promedio</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{trainResult.average_mse?.toFixed(2) || 'N/A'}</div>
              <div className="stat-label">MSE Promedio</div>
            </div>
          </div>
          <div className="message" style={{ marginTop: '15px', background: 'rgba(110, 139, 255, 0.1)', padding: '15px', borderRadius: '8px' }}>
            <p><strong>Justificación:</strong> {trainResult.justification}</p>
            {trainResult.products && (
              <p style={{ marginTop: '10px' }}>
                <strong>Productos entrenados:</strong> {trainResult.products.slice(0, 10).join(', ')}
                {trainResult.products.length > 10 && ` y ${trainResult.products.length - 10} más`}
              </p>
            )}
          </div>
        </div>
      )}

      {/* GRÁFICO 1: Ventas por Producto (Región seleccionada) */}
      {trainResult && historicalData && region && (
        <div className="api-form" style={{ marginBottom: '20px' }}>
          <h3>Gráfico 1: Ventas por Producto - Región: {region} (con Regresión Lineal)</h3>
          <div className="form-field" style={{ marginBottom: '15px' }}>
            <label htmlFor="chart1-products">Seleccionar Productos (múltiple)</label>
            <select
              id="chart1-products"
              multiple
              value={chart1Products}
              onChange={(e) => {
                const selected = Array.from(e.target.selectedOptions, option => option.value)
                setChart1Products(selected)
              }}
              className="form-input"
              style={{ minHeight: '100px' }}
            >
              {salesData.products?.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
              Mantén presionado Ctrl (o Cmd en Mac) para seleccionar múltiples productos. Muestra datos de la región: <strong>{region}</strong>
            </small>
          </div>
          
          {chart1Data && (
            <div style={{ height: '400px', marginTop: '20px' }}>
              <Line data={chart1Data} options={chartOptions} />
            </div>
          )}
        </div>
      )}

      {/* GRÁFICO 2: Predicción Diaria */}
      {trainResult && region && (
        <div className="api-form" style={{ marginBottom: '20px' }}>
          <h3>Gráfico 2: Predicción Diaria por Producto - Región: {region}</h3>
          
          <div className="form-field" style={{ marginBottom: '15px' }}>
            <label htmlFor="chart2-product">Producto</label>
            <select
              id="chart2-product"
              value={chart2Product}
              onChange={(e) => setChart2Product(e.target.value)}
              className="form-input"
            >
              <option value="">Seleccionar producto</option>
              {salesData.products?.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
              Región: <strong>{region}</strong> (seleccionada al entrenar el modelo)
            </small>
          </div>

          <div className="form-row" style={{ marginTop: '15px' }}>
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

          <div className="form-row" style={{ marginTop: '15px' }}>
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
            disabled={loading || !startDate || !chart2Product}
            style={{ marginTop: '15px' }}
          >
            {loading ? 'Prediciendo...' : 'Generar Predicción y Gráfico'}
          </button>

          {chart2Data && (
            <div style={{ height: '400px', marginTop: '20px' }}>
              <Line data={chart2Data} options={chartOptions} />
            </div>
          )}
        </div>
      )}

      {/* Resultados de la Predicción */}
      {predictions && (
        <div className="stats-panel">
          <h3>Resumen de Predicciones</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{predictions.overall_summary?.total_predicted?.toFixed(2) || 0}</div>
              <div className="stat-label">Total Predicho</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{predictions.overall_summary?.average_daily?.toFixed(2) || 0}</div>
              <div className="stat-label">Promedio Diario</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{predictions.overall_summary?.combinations || 0}</div>
              <div className="stat-label">Combinaciones</div>
            </div>
          </div>

          {predictions.summary && Object.keys(predictions.summary).length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <h4>Resumen por Combinación</h4>
              {Object.entries(predictions.summary).map(([key, value]) => (
                <div key={key} className="history-item" style={{ marginTop: '10px' }}>
                  <div className="history-item-header">
                    <span><strong>{key}:</strong> Total: {value.total_predicted}, Promedio diario: {value.average_daily.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

    </section>
  )
}

export default SalesPrediction

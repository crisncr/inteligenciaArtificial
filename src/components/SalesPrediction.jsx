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
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)
  const [chart1Products, setChart1Products] = useState([])
  const [chart2Product, setChart2Product] = useState('')
  const [availableProducts, setAvailableProducts] = useState([])

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
    setAvailableProducts([])

    try {
      const result = await salesPredictionAPI.train(salesFile, region, 'linear_regression')
      setTrainResult(result)
      
      // Cargar datos históricos después del entrenamiento
      // loadHistoricalData ya filtra los productos disponibles
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
      
      // Filtrar productos disponibles de la región seleccionada
      if (region && data.historical_data) {
        const regionProducts = [...new Set(
          data.historical_data
            .filter(d => d.region === region)
            .map(d => d.producto)
        )].sort()
        setAvailableProducts(regionProducts)
        
        // Actualizar selecciones si no hay productos seleccionados
        if (regionProducts.length > 0 && chart1Products.length === 0) {
          setChart1Products([regionProducts[0]])
          setChart2Product(regionProducts[0])
        }
      }
    } catch (err) {
      console.error('Error al cargar datos históricos:', err)
    }
  }

  const handlePredict = async () => {
    if (!trainResult) {
      alert('Primero entrena el modelo')
      return
    }

    if (!chart2Product) {
      alert('Selecciona un producto')
      return
    }

    if (!historicalData || !historicalData.historical_data || historicalData.historical_data.length === 0) {
      alert('No hay datos históricos disponibles')
      return
    }

    // Obtener la última fecha histórica para este producto y región
    const lastDate = historicalData.historical_data
      .filter(d => d.region === region && d.producto === chart2Product)
      .map(d => new Date(d.fecha))
      .sort((a, b) => b - a)[0]

    if (!lastDate || isNaN(lastDate.getTime())) {
      alert('No hay datos históricos para este producto en esta región')
      return
    }

    // Calcular fecha de inicio (día siguiente al último histórico)
    const startDate = new Date(lastDate)
    startDate.setDate(startDate.getDate() + 1)
    const startDateStr = startDate.toISOString().split('T')[0]

    setLoading(true)
    setPredictions(null)

    try {
      const result = await salesPredictionAPI.predict(
        region || null,
        chart2Product || null,
        'linear_regression',
        startDateStr,
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

  // Preparar datos para Gráfico 1: Ventas por días (evolución temporal)
  const prepareChart1Data = () => {
    if (!historicalData || !chart1Products.length || !region) return null

    // Filtrar datos por la región seleccionada y productos seleccionados
    const filteredData = historicalData.historical_data.filter(
      d => d.region === region && chart1Products.includes(d.producto)
    )
    
    if (filteredData.length === 0) return null

    // Obtener todas las fechas únicas y ordenarlas (como fechas, no strings)
    const allDates = [...new Set(filteredData.map(d => d.fecha))]
      .sort((a, b) => new Date(a) - new Date(b))
    
    const datasets = []
    const colors = [
      'rgba(54, 162, 235, 1)',
      'rgba(255, 99, 132, 1)',
      'rgba(75, 192, 192, 1)',
      'rgba(255, 206, 86, 1)',
      'rgba(153, 102, 255, 1)',
      'rgba(255, 159, 64, 1)'
    ]

    chart1Products.forEach((producto, idx) => {
      const color = colors[idx % colors.length]

      // Agrupar datos por fecha para este producto
      const dataByDate = {}
      filteredData
        .filter(d => d.producto === producto)
        .forEach(d => {
          if (!dataByDate[d.fecha]) {
            dataByDate[d.fecha] = []
          }
          dataByDate[d.fecha].push(d.ventas)
        })

      // Crear array de ventas por fecha (promedio si hay múltiples registros el mismo día)
      const ventas = allDates.map(fecha => {
        const ventasDelDia = dataByDate[fecha] || []
        return ventasDelDia.length > 0 
          ? ventasDelDia.reduce((a, b) => a + b, 0) / ventasDelDia.length 
          : null
      })

      // Calcular regresión lineal - MEJORADO: requiere mínimo 2 puntos
      const validData = ventas
        .map((v, i) => v !== null ? { x: i, y: v } : null)
        .filter(d => d !== null)
      
      // Datos reales (siempre mostrar si hay al menos 1 punto)
      if (validData.length > 0) {
        datasets.push({
          label: `${producto} (Datos)`,
          data: ventas,
          borderColor: color,
          backgroundColor: color.replace('1)', '0.2)'),
          fill: false,
          tension: 0.1,
          pointRadius: 3
        })

        // Línea de regresión (solo si hay mínimo 2 puntos)
        if (validData.length >= 2) {
          const x = validData.map(d => d.x)
          const y = validData.map(d => d.y)
          const regression = calculateLinearRegression(x, y)
          
          const regressionLine = allDates.map((_, i) => {
            const validPoint = validData.find(d => d.x === i)
            if (validPoint) {
              return regression.slope * i + regression.intercept
            }
            return null
          })

          datasets.push({
            label: `${producto} (Regresión)`,
            data: regressionLine,
            borderColor: color,
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            fill: false,
            tension: 0.1,
            pointRadius: 0
          })
        }
      }
    })

    return {
      labels: allDates,
      datasets
    }
  }

  // Preparar datos para Gráfico 2: Combinar histórico + predicciones
  const prepareChart2Data = () => {
    if (!chart2Product || !region) return null
    
    // Obtener datos históricos del producto
    const historical = historicalData?.historical_data
      ?.filter(d => d.producto === chart2Product && d.region === region)
      .map(d => ({
        fecha: d.fecha,
        ventas: d.ventas,
        tipo: 'histórico'
      }))
      .sort((a, b) => new Date(a.fecha) - new Date(b.fecha)) || []

    // Obtener predicciones
    const predicted = predictions?.predictions
      ?.filter(p => p.producto === chart2Product && p.region === region)
      .map(p => ({
        fecha: p.fecha,
        ventas: p.ventas_predichas,
        tipo: 'predicción'
      }))
      .sort((a, b) => new Date(a.fecha) - new Date(b.fecha)) || []

    if (historical.length === 0 && predicted.length === 0) return null

    // Combinar y ordenar por fecha
    const allData = [...historical, ...predicted]
      .sort((a, b) => new Date(a.fecha) - new Date(b.fecha))

    const labels = allData.map(d => d.fecha)
    const ventasHistoricas = allData.map(d => d.tipo === 'histórico' ? d.ventas : null)
    const ventasPredichas = allData.map(d => d.tipo === 'predicción' ? d.ventas : null)

    // Calcular regresión lineal solo sobre datos históricos
    const historicalVentas = historical.map(d => d.ventas)
    let regressionLine = null
    if (historicalVentas.length >= 2) {
      const x = historicalVentas.map((_, i) => i)
      const y = historicalVentas
      const regression = calculateLinearRegression(x, y)
      regressionLine = allData.map((_, i) => {
        if (i < historical.length) {
          return regression.slope * i + regression.intercept
        }
        // Extender regresión a predicciones
        return regression.slope * i + regression.intercept
      })
    }

    const datasets = []
    
    // Datos históricos
    if (historical.length > 0) {
      datasets.push({
        label: `${chart2Product} - ${region} (Histórico)`,
        data: ventasHistoricas,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        fill: false,
        tension: 0.1,
        pointRadius: 3
      })
    }

    // Predicciones
    if (predicted.length > 0) {
      datasets.push({
        label: `${chart2Product} - ${region} (Predicción)`,
        data: ventasPredichas,
        borderColor: 'rgba(255, 206, 86, 1)',
        backgroundColor: 'rgba(255, 206, 86, 0.2)',
        fill: false,
        tension: 0.1,
        pointRadius: 3,
        borderDash: [3, 3]
      })
    }

    // Regresión lineal
    if (regressionLine) {
      datasets.push({
        label: `${chart2Product} - ${region} (Regresión Lineal)`,
        data: regressionLine,
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'transparent',
        borderDash: [5, 5],
        fill: false,
        tension: 0.1,
        pointRadius: 0
      })
    }

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

      {/* GRÁFICO 1: Ventas por Días (Evolución Temporal) */}
      {trainResult && historicalData && region && (
        <div className="api-form" style={{ marginBottom: '20px' }}>
          <h3>Gráfico 1: Ventas por Días - Región: {region} (con Regresión Lineal)</h3>
          <div className="form-field" style={{ marginBottom: '15px' }}>
            <label htmlFor="chart1-product-select">Seleccionar Producto</label>
            <select
              id="chart1-product-select"
              value=""
              onChange={(e) => {
                const selectedProduct = e.target.value
                if (selectedProduct && !chart1Products.includes(selectedProduct)) {
                  setChart1Products([...chart1Products, selectedProduct])
                }
                e.target.value = "" // Resetear el selector
              }}
              className="form-input"
            >
              <option value="">Seleccionar producto</option>
              {availableProducts.length > 0 ? availableProducts
                .filter(p => !chart1Products.includes(p))
                .map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                )) : salesData.products?.filter(p => !chart1Products.includes(p)).map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
            </select>
            {chart1Products.length > 0 && (
              <div style={{ marginTop: '12px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {chart1Products.map((producto) => (
                  <div
                    key={producto}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '6px 12px',
                      background: 'rgba(110, 139, 255, 0.2)',
                      border: '1px solid rgba(110, 139, 255, 0.4)',
                      borderRadius: '8px',
                      fontSize: '0.9em',
                      color: 'var(--primary)'
                    }}
                  >
                    <span>{producto}</span>
                    <button
                      type="button"
                      onClick={() => setChart1Products(chart1Products.filter(p => p !== producto))}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--primary)',
                        cursor: 'pointer',
                        fontSize: '16px',
                        padding: '0',
                        width: '20px',
                        height: '20px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        borderRadius: '50%',
                        transition: 'all 0.2s'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(110, 139, 255, 0.3)'
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'transparent'
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
            <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
              Selecciona productos uno por uno para agregarlos al gráfico. Muestra evolución temporal de ventas por día en la región: <strong>{region}</strong>
              {availableProducts.length > 0 && ` (${availableProducts.length} productos disponibles)`}
            </small>
          </div>
          
          {chart1Data && (
            <div style={{ height: '400px', marginTop: '20px' }}>
              <Line data={chart1Data} options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: `Evolución de Ventas por Día - ${region}`
                  }
                },
                scales: {
                  ...chartOptions.scales,
                  x: {
                    title: {
                      display: true,
                      text: 'Fecha'
                    }
                  },
                  y: {
                    title: {
                      display: true,
                      text: 'Ventas'
                    },
                    beginAtZero: false
                  }
                }
              }} />
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
              {availableProducts.length > 0 ? availableProducts.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              )) : salesData.products?.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
              Región: <strong>{region}</strong> (seleccionada al entrenar el modelo)
              {availableProducts.length > 0 && ` - ${availableProducts.length} productos disponibles`}
            </small>
          </div>

          <div className="form-field" style={{ marginTop: '15px' }}>
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
            <small style={{ color: 'var(--text-secondary)', display: 'block', marginTop: '5px' }}>
              Las predicciones comenzarán desde el día siguiente al último dato histórico disponible
            </small>
          </div>

          <button 
            className="btn" 
            onClick={handlePredict} 
            disabled={loading || !chart2Product}
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

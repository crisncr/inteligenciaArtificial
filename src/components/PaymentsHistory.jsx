import { useState, useEffect } from 'react'
import { paymentsAPI } from '../utils/api'

function PaymentsHistory({ user }) {
  const [payments, setPayments] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadPayments()
  }, [user])

  const loadPayments = async () => {
    if (!user) return
    
    setLoading(true)
    try {
      const data = await paymentsAPI.getAll()
      setPayments(data || [])
    } catch (err) {
      console.error('Error al cargar pagos:', err)
      setPayments([])
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (timestamp) => {
    if (!timestamp) return 'Fecha no disponible'
    const date = new Date(timestamp)
    return new Intl.DateTimeFormat('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date)
  }

  const getStatusClass = (status) => {
    if (status === 'completed') return 'status-completed'
    if (status === 'pending') return 'status-pending'
    if (status === 'failed') return 'status-failed'
    return ''
  }

  const getStatusLabel = (status) => {
    if (status === 'completed') return 'Completado'
    if (status === 'pending') return 'Pendiente'
    if (status === 'failed') return 'Fallido'
    return status
  }

  if (loading) {
    return (
      <section className="payments-history-panel">
        <h2>Historial de Pagos</h2>
        <div className="loading">Cargando pagos...</div>
      </section>
    )
  }

  if (payments.length === 0) {
    return (
      <section className="payments-history-panel">
        <h2>Historial de Pagos</h2>
        <p className="subtitle">
          Gestiona tus pagos y suscripciones
        </p>
        <div className="empty-state">
          <p>No hay pagos registrados aún.</p>
        </div>
      </section>
    )
  }

  return (
    <section className="payments-history-panel">
      <h2>Historial de Pagos</h2>
      <p className="subtitle">
        Gestiona tus pagos y suscripciones
      </p>

      <div className="payments-list">
        {payments.map((payment) => (
          <div key={payment.id} className="payment-item">
            <div className="payment-header">
              <div className="payment-info">
                <h3>Pago #{payment.id}</h3>
                <span className={`payment-status ${getStatusClass(payment.status)}`}>
                  {getStatusLabel(payment.status)}
                </span>
              </div>
              <div className="payment-amount">
                ${payment.amount.toFixed(2)}
              </div>
            </div>
            <div className="payment-details">
              <p><strong>Fecha:</strong> {formatDate(payment.created_at)}</p>
              {payment.payment_method && (
                <p><strong>Método:</strong> {payment.payment_method}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default PaymentsHistory


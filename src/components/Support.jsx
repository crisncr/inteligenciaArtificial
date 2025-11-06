function Support({ user }) {
  return (
    <section className="support-panel">
      <h2>Ayuda y Soporte</h2>
      <p className="subtitle">
        ¿Necesitas ayuda? Estamos aquí para asistirte
      </p>

      <div className="support-sections">
        <div className="support-section">
          <h3>📧 Contacto</h3>
          <p>Si tienes alguna pregunta o necesitas asistencia, no dudes en contactarnos:</p>
          <div className="contact-info">
            <p><strong>Email:</strong> soporte@sentimetria.com</p>
            <p><strong>Horario:</strong> Lunes a Viernes, 9:00 AM - 6:00 PM</p>
          </div>
        </div>

        <div className="support-section">
          <h3>❓ Preguntas Frecuentes</h3>
          <div className="faq-list">
            <div className="faq-item">
              <h4>¿Cómo funciona el análisis de sentimientos?</h4>
              <p>Nuestro motor utiliza procesamiento de lenguaje natural optimizado para español para clasificar textos en positivo, negativo o neutral.</p>
            </div>
            <div className="faq-item">
              <h4>¿Puedo usar la API externa?</h4>
              <p>Sí, puedes configurar una API externa en la sección "API Externa" para obtener comentarios automáticamente y analizarlos.</p>
            </div>
            <div className="faq-item">
              <h4>¿Cómo cambio mi plan?</h4>
              <p>Ve a la sección "Planes" y selecciona el plan que deseas. Los cambios se aplicarán inmediatamente.</p>
            </div>
            <div className="faq-item">
              <h4>¿Qué pasa si alcanzo el límite de análisis?</h4>
              <p>Si alcanzas el límite de tu plan, puedes actualizar a un plan superior para continuar analizando sin límites.</p>
            </div>
          </div>
        </div>

        <div className="support-section">
          <h3>📚 Documentación</h3>
          <p>Consulta nuestra documentación para obtener más información sobre:</p>
          <ul className="docs-list">
            <li>Guía de uso de la API</li>
            <li>Integración con APIs externas</li>
            <li>Límites y planes</li>
            <li>Mejores prácticas</li>
          </ul>
        </div>
      </div>
    </section>
  )
}

export default Support


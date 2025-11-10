function CookieBar({ onAccept }) {
  return (
    <div className="cookies">
      <span>Al utilizar este sitio aceptas nuestra política de cookies.</span>
      <button className="btn" onClick={onAccept} style={{ whiteSpace: 'nowrap', minWidth: 'auto', padding: '10px 20px' }}>
        Aceptar
      </button>
    </div>
  )
}

export default CookieBar


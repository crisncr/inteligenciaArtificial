function CookieBar({ onAccept }) {
  return (
    <div className="cookies">
      <span>Al utilizar este sitio aceptas nuestra política de cookies.</span>
      <button className="btn btn--small" onClick={onAccept}>
        Aceptar
      </button>
    </div>
  )
}

export default CookieBar


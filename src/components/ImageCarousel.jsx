import { useState, useEffect } from 'react'

function ImageCarousel({ images, autoPlay = true, interval = 3000 }) {
  const [currentIndex, setCurrentIndex] = useState(0)

  useEffect(() => {
    if (!autoPlay || images.length <= 1) return

    const timer = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length)
    }, interval)

    return () => clearInterval(timer)
  }, [autoPlay, interval, images.length])

  const goToSlide = (index) => {
    setCurrentIndex(index)
  }

  const goToPrevious = () => {
    setCurrentIndex((prevIndex) => 
      prevIndex === 0 ? images.length - 1 : prevIndex - 1
    )
  }

  const goToNext = () => {
    setCurrentIndex((prevIndex) => 
      prevIndex === images.length - 1 ? 0 : prevIndex + 1
    )
  }

  if (!images || images.length === 0) {
    // Si no hay imágenes, mostrar un placeholder
    return (
      <div className="image-carousel">
        <div className="carousel-container">
          <div className="carousel-slide">
            <div className="carousel-placeholder">
              <span style={{ fontSize: '4rem' }}>✨</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="image-carousel">
      <div className="carousel-container">
        {images.length > 1 && (
          <button 
            className="carousel-button carousel-button-prev" 
            onClick={goToPrevious}
            aria-label="Imagen anterior"
          >
            ‹
          </button>
        )}
        
        <div className="carousel-slide">
          <img 
            src={images[currentIndex]} 
            alt={`Slide ${currentIndex + 1}`}
            className="carousel-image"
          />
        </div>

        {images.length > 1 && (
          <button 
            className="carousel-button carousel-button-next" 
            onClick={goToNext}
            aria-label="Siguiente imagen"
          >
            ›
          </button>
        )}
      </div>

      {images.length > 1 && (
        <div className="carousel-indicators">
          {images.map((_, index) => (
            <button
              key={index}
              className={`carousel-indicator ${index === currentIndex ? 'active' : ''}`}
              onClick={() => goToSlide(index)}
              aria-label={`Ir a slide ${index + 1}`}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default ImageCarousel


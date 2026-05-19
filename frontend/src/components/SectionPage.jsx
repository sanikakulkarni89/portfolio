import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

export default function SectionPage({ color, icon, title, children }) {
  const panelRef = useRef(null)
  const navigate = useNavigate()

  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      panelRef.current?.classList.add('active')
    })
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <div style={{ '--pc': color }}>
      <div className="room-bg">
        <img src="/room.jpg" alt="" className="room-bg-img" />
      </div>

      <div className="content-panel" id="panel" ref={panelRef}>
        <div className="panel-inner">
          <div className="panel-top">
            <div className="panel-title-row">
              <span className="panel-icon">{icon}</span>
              <h1 className="panel-title">{title}</h1>
            </div>
            <button className="back-btn" onClick={() => navigate('/')}>← BACK</button>
          </div>

          <div className="panel-scroll">
            {children}
          </div>
        </div>
      </div>
    </div>
  )
}

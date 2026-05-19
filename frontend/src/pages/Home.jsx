import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

const SPARKS = ['✦', '★', '✧', '◆', '·', '✶', '✵']

function spawnSparkles(hotspot, count = 7) {
  const color = getComputedStyle(hotspot).getPropertyValue('--c').trim()
  for (let i = 0; i < count; i++) {
    const el = document.createElement('i')
    el.className = 'sparkle'
    el.textContent = SPARKS[Math.floor(Math.random() * SPARKS.length)]
    const x = 10 + Math.random() * 80
    const y = 10 + Math.random() * 80
    const dx1 = (Math.random() - 0.5) * 30 + 'px'
    const dy1 = -(10 + Math.random() * 20) + 'px'
    const dx2 = (Math.random() - 0.5) * 60 + 'px'
    const dy2 = -(25 + Math.random() * 40) + 'px'
    el.style.cssText = `left:${x}%;top:${y}%;--dx1:${dx1};--dy1:${dy1};--dx2:${dx2};--dy2:${dy2};--c:${color};animation-delay:${Math.random() * 0.3}s;`
    hotspot.appendChild(el)
    el.addEventListener('animationend', () => el.remove(), { once: true })
  }
}

const HOTSPOTS = [
  {
    id: 'hs-education',
    color: '#f5a7c7',
    style: { left: '3%', top: '9%', width: '27%', height: '70%' },
    clip: 'polygon(44% 36%, 97% 26%, 130% 19%, 130% 73%, 81% 83%, 44% 90%)',
    imgStyle: { width: '370.37%', height: '142.86%', left: '-11.11%', top: '-12.86%' },
    icon: '📚',
    label: 'EDUCATION',
    path: '/education',
  },
  {
    id: 'hs-work',
    color: '#c4a8e8',
    style: { left: '38%', top: '20%', width: '20%', height: '26%' },
    clip: 'polygon(42% 57%, 52% 45%, 85% 44%, 94% 54%, 96% 77%, 95% 95%, 96% 107%, 78% 107%, 57% 107%, 40% 106%, 40% 96%, 39% 82%)',
    imgStyle: { width: '500%', height: '384.62%', left: '-190%', top: '-76.92%' },
    icon: '💻',
    label: 'WORK',
    path: '/work',
  },
{
    id: 'hs-cat',
    color: '#e8e0a4',
    style: { left: '63%', top: '50%', width: '19%', height: '21%' },
    clip: 'polygon(45% 71%, 53% 54%, 63% 45%, 75% 20%, 82% 34%, 94% 32%, 94% 49%, 85% 52%, 96% 67%, 95% 81%, 79% 72%, 67% 81%, 66% 67%)',
    imgStyle: { width: '526.32%', height: '476.19%', left: '-331.58%', top: '-238.1%' },
    icon: '🐱',
    label: 'MY CAT',
    path: '/cat',
  },
  {
    id: 'hs-hobbies',
    color: '#a8e8c4',
    style: { left: '4%', top: '6%', width: '26%', height: '18%' },
    clip: 'polygon(69% 87%, 109% 53%, 130% 81%, 90% 114%, 56% 130%, 19% 118%)',
    imgStyle: { width: '384.62%', height: '555.56%', left: '-15.38%', top: '-33.33%' },
    icon: '🌿',
    label: 'HOBBIES',
    path: '/hobbies',
  },
  {
    id: 'hs-projects',
    color: '#a4bde8',
    style: { left: '46%', top: '42%', width: '22%', height: '22%' },
    clip: 'polygon(58% -3%, 83% 13%, 130% 37%, 88% 64%, 69% 55%, 38% 38%, 49% 18%)',
    imgStyle: { width: '454.55%', height: '454.55%', left: '-209.09%', top: '-190.91%' },
    icon: '🗂️',
    label: 'PROJECTS',
    path: '/projects',
  },
]

export default function Home() {
  const navigate = useNavigate()
  const containerRef = useRef(null)
  const imgRef = useRef(null)

  useEffect(() => {
    const intervals = new Map()

    const hotspots = containerRef.current?.querySelectorAll('.hotspot') || []

    hotspots.forEach(hs => {
      hs.addEventListener('mouseenter', () => {
        spawnSparkles(hs)
        intervals.set(hs, setInterval(() => spawnSparkles(hs, 3), 700))
      })
      hs.addEventListener('mouseleave', () => {
        clearInterval(intervals.get(hs))
        intervals.delete(hs)
      })
    })

    function tick() {
      const all = Array.from(hotspots)
      if (!all.length) return
      spawnSparkles(all[Math.floor(Math.random() * all.length)], 2)
      idleTimer = setTimeout(tick, 2000 + Math.random() * 3000)
    }
    let idleTimer = setTimeout(tick, 3500)

    return () => {
      clearTimeout(idleTimer)
      intervals.forEach(iv => clearInterval(iv))
    }
  }, [])

  function handleClick(hs) {
    const img = imgRef.current
    img.style.transition = 'filter 0.15s ease'
    img.style.filter = 'brightness(2) saturate(0)'
    setTimeout(() => navigate(hs.path), 180)
  }

  return (
    <div className="room-wrapper">
      <div className="room-container" ref={containerRef}>
        <img src={`${import.meta.env.BASE_URL}room.jpg`} alt="Pixel art room" className="room-image" ref={imgRef} />

        {HOTSPOTS.map(hs => (
          <div
            key={hs.id}
            className="hotspot"
            id={hs.id}
            style={{ ...hs.style, '--c': hs.color }}
            onClick={() => handleClick(hs)}
          >
            <div className="hs-clip" style={{ clipPath: hs.clip }}>
              <img className="hs-img" src={`${import.meta.env.BASE_URL}room.jpg`} alt="" style={hs.imgStyle} />
            </div>
            <div className="hs-label">
              <span className="hs-label-icon">{hs.icon}</span>
              <span className="hs-label-text">{hs.label}</span>
            </div>
          </div>
        ))}

        <div className="room-hint">✦ click an object to explore ✦</div>
      </div>
    </div>
  )
}

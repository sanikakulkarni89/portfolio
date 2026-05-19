import { useEffect } from 'react'

const CHARS = ['✦', '✧', '·', '★', '✶']
const DEFAULT_COLOR = '#f5a7c7'

export default function CursorTrail() {
  useEffect(() => {
    let trailColor = DEFAULT_COLOR
    let lastDrop = 0

    function dropTrail(x, y) {
      const now = performance.now()
      if (now - lastDrop < 68) return
      lastDrop = now

      const t = document.createElement('span')
      t.className = 'pxc-trail'
      t.textContent = CHARS[Math.floor(Math.random() * CHARS.length)]
      t.style.left = x + (Math.random() - 0.5) * 10 + 'px'
      t.style.top = y + (Math.random() - 0.5) * 10 + 'px'
      t.style.color = trailColor
      t.style.fontSize = 6 + Math.random() * 8 + 'px'
      document.body.appendChild(t)
      t.addEventListener('animationend', () => t.remove(), { once: true })
    }

    function onMouseMove(e) { dropTrail(e.clientX, e.clientY) }

    function onMouseEnter(e) {
      const el = e.currentTarget
      const c =
        getComputedStyle(el).getPropertyValue('--c').trim() ||
        getComputedStyle(el).getPropertyValue('--pc').trim() ||
        DEFAULT_COLOR
      if (c) trailColor = c
    }
    function onMouseLeave() { trailColor = DEFAULT_COLOR }

    document.addEventListener('mousemove', onMouseMove)

    const targets = document.querySelectorAll('.hotspot, a, button, .back-btn, .project-link')
    targets.forEach(el => {
      el.addEventListener('mouseenter', onMouseEnter)
      el.addEventListener('mouseleave', onMouseLeave)
    })

    return () => {
      document.removeEventListener('mousemove', onMouseMove)
      targets.forEach(el => {
        el.removeEventListener('mouseenter', onMouseEnter)
        el.removeEventListener('mouseleave', onMouseLeave)
      })
    }
  }, [])

  return null
}

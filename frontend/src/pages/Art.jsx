import { useState } from 'react'
import SectionPage from '../components/SectionPage'

const ART_ITEMS = [
  { cat: 'digital',     icon: '🖌️', label: 'DIGITAL 01',    text: 'Piece Title · Digital · 2024' },
  { cat: 'traditional', icon: '✏️', label: 'SKETCH 01',     text: 'Sketch Title · Pencil · 2024' },
  { cat: 'pixel',       icon: '🎮', label: 'PIXEL 01',      text: 'Pixel Art · Aseprite · 2023' },
  { cat: 'digital',     icon: '🌸', label: 'DIGITAL 02',    text: 'Piece Title · Procreate · 2023' },
  { cat: 'traditional', icon: '🖊️', label: 'SKETCH 02',     text: 'Sketch Title · Ink · 2023' },
  { cat: 'pixel',       icon: '⭐', label: 'PIXEL 02',      text: 'Pixel Scene · Aseprite · 2023' },
  { cat: 'digital',     icon: '🌙', label: 'DIGITAL 03',    text: 'Night Scene · Digital · 2022' },
  { cat: 'traditional', icon: '🍂', label: 'WATERCOLOR 01', text: 'Autumn · Watercolour · 2022' },
  { cat: 'pixel',       icon: '🏡', label: 'PIXEL 03',      text: 'Cozy Room · Aseprite · 2022' },
]

const FILTERS = ['all', 'digital', 'traditional', 'pixel']

export default function Art() {
  const [active, setActive] = useState('all')

  const visible = ART_ITEMS.filter(a => active === 'all' || a.cat === active)

  return (
    <SectionPage color="#e8c4a4" icon="🎨" title="ART">
      <p className="section-label">// creative work</p>

      <p className="art-intro">
        I draw, sketch, and occasionally paint. Mostly characters and cozy scenes.
        Replace these placeholders with your own pieces.
      </p>

      <div className="art-filter-row">
        {FILTERS.map(f => (
          <button
            key={f}
            className={`art-filter${active === f ? ' active' : ''}`}
            onClick={() => setActive(f)}
          >
            {f.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="art-grid">
        {visible.map((a, i) => (
          <div className="art-item" key={i}>
            <div className="art-placeholder">
              <span className="art-ph-icon">{a.icon}</span>
              <span className="art-ph-text">{a.label}</span>
            </div>
            <div className="art-label">{a.text}</div>
          </div>
        ))}
      </div>
    </SectionPage>
  )
}

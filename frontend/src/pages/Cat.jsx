import SectionPage from '../components/SectionPage'

import cheese       from '../assets/cheese.jpg'
import babyCheeseA  from '../assets/baby cheese.jpg'
import babyCheeseB  from '../assets/baby cheese 2.jpg'
import tshirt       from '../assets/cheese in a tshirt.jpg'
import tub          from '../assets/cheese in a tub.jpg'
import garden       from '../assets/cheese in the garden.jpg'
import sun          from '../assets/cheese in the sun.jpg'

const STATS = [
  { label: 'BREED',          val: 'Doll Faced Persian' },
  { label: 'AGE',            val: 'almost 6 years old (probably ancient in cat years)' },
  { label: 'FAVOURITE SPOT', val: 'My keyboard, mid-meeting' },
  { label: 'MOOD',           val: 'Perpetually unimpressed' },
  { label: 'FAVOURITE TREAT',val: 'Fancy Feast, only the pâté' },
  { label: 'HOBBY',          val: 'Knocking things off the desk at 3am' },
]

const FACTS = [
  'Walks across the laptop keyboard during video calls approximately 3x per week.',
  'Has a specific chirp he does when he sees birds(especially a sparrow), it sounds exactly like a broken printer.',
  'Refuses to drink water unless it\'s moving. Has trained me to leave the tap dripping.',
  'Sits on any paper or book I\'m currently trying to read. Always. It\'s personal.',
  'Absolutely loses his mind over a specific toy a crinkle ball from the dollar store.',
  'Has never once knocked something off a surface accidentally.',
]

const PHOTOS = [
  { src: babyCheeseA, alt: 'Baby Cheese' },
  { src: babyCheeseB, alt: 'Baby Cheese 2' },
  { src: tshirt,      alt: 'Cheese in a t-shirt' },
  { src: tub,         alt: 'Cheese in a tub' },
  { src: garden,      alt: 'Cheese in the garden' },
  { src: sun,         alt: 'Cheese in the sun' },
]

export default function Cat() {
  return (
    <SectionPage color="#e8e0a4" icon="🐱" title="MY CAT">
      <p className="section-label">// the real boss</p>

      <div className="cat-hero">
        <img src={cheese} alt="Cheese" />
      </div>

      <div className="cat-name-hero">
        <h2>CHEESE</h2>
        <p>chief nap officer · professional lap warmer · 2020 – present</p>
      </div>

      <div className="cat-quote">
        &ldquo;He supervises every commit I push. So far he has approved none of them.&rdquo;
      </div>

      <div className="cat-stats">
        {STATS.map(s => (
          <div className="cat-stat" key={s.label}>
            <div className="cat-stat-label">{s.label}</div>
            <div className="cat-stat-val">{s.val}</div>
          </div>
        ))}
      </div>

      <div className="cat-facts-title">// documented behaviours</div>
      <ul className="cat-facts">
        {FACTS.map((f, i) => <li key={i}>{f}</li>)}
      </ul>

      <div className="cat-facts-title">// photo album</div>
      <div className="cat-gallery">
        {PHOTOS.map((p, i) => (
          <div className="cat-photo" key={i}>
            <img src={p.src} alt={p.alt} />
          </div>
        ))}
      </div>
    </SectionPage>
  )
}

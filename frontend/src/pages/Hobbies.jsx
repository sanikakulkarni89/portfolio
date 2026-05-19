import SectionPage from '../components/SectionPage'

const HOBBIES = [
  { emoji: '🎮', name: 'GAMING', desc: 'Stray(obvi), It takes two, Split Fiction, Spider-Man, Unravled' },
  { emoji: '📖', name: 'READING', desc: 'Mythology, Self-help, Guided journaling' },
  { emoji: '🍵', name: 'MATCHA', desc: 'Yes it tastes like  grass. Yes I love it.' },
  { emoji: '🎵', name: 'MUSIC', desc: 'Learning piano. Playlists for every mood, every codebase.' },
  { emoji: '🍜', name: 'COOKING', desc: 'I believe its living to eat is better than eating to live.' },
  { emoji: '🚶', name: 'RUNNING', desc: 'The most consuming hobby ever.' },
  { emoji: '🎬', name: 'FILMS', desc: 'Studio Ghibli marathons and offbeat indie cinema. Keeping a Letterboxd.' },
  { emoji: '✍️', name: 'JOURNALING', desc: 'Bullet journaling since 2020. Keeps the chaos organised-ish.' },
  { emoji: '🌙', name: 'STARGAZING', desc: 'Amateur astronomer. Can name seven constellations on a good night.' },
  { emoji: '🧩', name: 'PUZZLES', desc: '1000-piece jigsaws and NYT crossword. Both very therapeutic.' },
]

export default function Hobbies() {
  return (
    <SectionPage color="#a8e8c4" icon="🌿" title="HOBBIES">
      <p className="section-label">// outside of work</p>

      <p className="hobbies-intro">
        When I&apos;m not debugging things at 2am, you&apos;ll probably find me doing one of these.
        I believe what you do for fun says a lot about how you think.
      </p>

      <div className="hobbies-grid">
        {HOBBIES.map(h => (
          <div className="hobby-card" key={h.name}>
            <span className="hobby-emoji">{h.emoji}</span>
            <div className="hobby-name">{h.name}</div>
            <div className="hobby-desc">{h.desc}</div>
          </div>
        ))}
      </div>
    </SectionPage>
  )
}

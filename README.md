# ✦ sanika's portfolio

> *a cozy little corner of the internet, built in a pixel art room*

**[visit the site →](https://sanikakulkarni89.github.io/portfolio/)**

---

```
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   📚 education    💻 work    🗂️ projects              │
  │                                                      │
  │         ✦ click an object to explore ✦               │
  │                                                      │
  │   🌿 hobbies                       🐱 my cat          │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

---

## what is this

a personal portfolio disguised as a pixel art bedroom. instead of a nav bar, you get a room. click on things. sparks fly.

built because a standard portfolio felt too formal for someone whose cat sits on the keyboard during every single video call.

---

## tech stack

| layer | tools |
|---|---|
| frontend | React 19, Vite 7, Tailwind CSS 4 |
| routing | React Router (HashRouter for GitHub Pages) |
| animations | pure CSS — sparkles, glows, clip-path hotspots |
| deployment | GitHub Pages via `gh-pages` |

no frameworks were harmed in the making of the cursor trail.

---

## running locally

```bash
cd frontend
npm install
npm run dev
```

site runs at `http://localhost:5173`

---

## deploying

```bash
cd frontend
npm run deploy
```

pushes the built `dist/` to the `gh-pages` branch. live in ~60 seconds.

---

## project structure

```
portfolio/
├── frontend/               # the React app
│   ├── src/
│   │   ├── pages/          # Home, Education, Work, Projects, Hobbies, Cat
│   │   ├── components/     # SectionPage, CursorTrail
│   │   └── assets/         # cheese photos (essential)
│   └── public/
│       └── room.jpg        # the pixel art room
└── data/                   # portfolio RAG system (separate backend)
```


---

*built with React, Tailwind, and the moral support of a perpetually unimpressed cat*

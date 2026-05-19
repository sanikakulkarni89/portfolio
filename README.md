# ✦ sanika's portfolio

> *a cozy little corner of the internet, built in a pixel art room*

**[visit the site →](https://sanikakulkarni89.github.io/portfolio/)**

---

```
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   📚 education    💻 work    🗂️ projects             │
  │                                                      │
  │         ✦ click an object to explore ✦              │
  │                                                      │
  │   🌿 hobbies                       🐱 my cat        │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

---

## what is this

a personal portfolio disguised as a pixel art bedroom. instead of a nav bar, you get a room. click on things. sparks fly.

built because a standard portfolio felt too formal for someone whose cat sits on the keyboard during every single video call.

---

## who is behind this

**Sanika Kulkarni** — MS Computer Science @ Syracuse University (May 2025), currently doing research on microservices and distributed systems.

previously: built payment integrations, automated architecture pipelines, deployed things on three different clouds. currently: architecting event-driven systems with more databases than most people have houseplants.

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

## the most important section

**Cheese** is a Doll Faced Persian who is almost 6 years old and has supervised every commit in this repo. he has approved none of them. his favourite spot is the keyboard, his favourite treat is Fancy Feast pâté (only the pâté), and he has never once knocked something off a surface *accidentally*.

he has a dedicated page. he earned it.

---

*built with React, Tailwind, and the moral support of a perpetually unimpressed cat*

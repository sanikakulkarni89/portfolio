import { HashRouter, Routes, Route } from 'react-router-dom'
import CursorTrail from './components/CursorTrail'
import Home from './pages/Home'
import Education from './pages/Education'
import Work from './pages/Work'
import Projects from './pages/Projects'
import Hobbies from './pages/Hobbies'
import Cat from './pages/Cat'

export default function App() {
  return (
    <HashRouter>
      <CursorTrail />
      <Routes>
        <Route path="/"          element={<Home />} />
        <Route path="/education" element={<Education />} />
        <Route path="/work"      element={<Work />} />
        <Route path="/projects"  element={<Projects />} />
        <Route path="/hobbies"   element={<Hobbies />} />
        <Route path="/cat"       element={<Cat />} />
      </Routes>
    </HashRouter>
  )
}

import { Toaster } from '@/components/ui/toaster'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'
import Analysis from './pages/Analysis'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Analysis />} />
        <Route path="*" element={<Analysis />} />
      </Routes>
      <Toaster />
    </Router>
  )
}

export default App

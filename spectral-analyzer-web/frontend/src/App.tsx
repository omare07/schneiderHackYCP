import { Box, Container, AppBar, Toolbar, Typography, Button } from '@mui/material'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import ScienceIcon from '@mui/icons-material/Science'
import SettingsIcon from '@mui/icons-material/Settings'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import Dashboard from './pages/Dashboard'
import Settings from './pages/Settings'
import Demo from './pages/Demo'

function App() {
  return (
    <BrowserRouter>
      <Box sx={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)' }}>
        <AppBar 
          position="static" 
          sx={{ 
            background: 'rgba(30, 41, 59, 0.8)',
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid rgba(148, 163, 184, 0.1)'
          }}
        >
          <Toolbar>
            <ScienceIcon sx={{ mr: 2, color: '#818cf8' }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 700 }}>
              SpectroView
            </Typography>
            <Button
              color="inherit"
              component={Link}
              to="/"
              sx={{ mr: 2 }}
            >
              Dashboard
            </Button>
            <Button
              color="inherit"
              component={Link}
              to="/demo"
              startIcon={<PlayArrowIcon />}
              sx={{ mr: 2 }}
            >
              Demo
            </Button>
            <Button
              color="inherit"
              component={Link}
              to="/settings"
              startIcon={<SettingsIcon />}
            >
              Settings
            </Button>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ py: 4 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/demo" element={<Demo />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Container>
      </Box>
    </BrowserRouter>
  )
}

export default App
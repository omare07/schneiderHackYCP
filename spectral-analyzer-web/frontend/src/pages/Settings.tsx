import { Box, Card, CardContent, Typography, TextField, Button, Divider } from '@mui/material'
import SaveIcon from '@mui/icons-material/Save'
import toast from 'react-hot-toast'

const Settings = () => {
  const handleSave = () => {
    toast.success('Settings saved successfully!')
  }

  return (
    <Box>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 4 }}>
        Settings
      </Typography>

      <Card sx={{ 
        background: 'rgba(30, 41, 59, 0.7)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(148, 163, 184, 0.1)',
        mb: 3 
      }}>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
            AI Configuration
          </Typography>
          <TextField
            fullWidth
            label="OpenRouter API Key"
            type="password"
            variant="outlined"
            placeholder="Enter your API key"
            sx={{ mb: 2 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Your API key is stored securely and used for AI-powered normalization
          </Typography>
          <Divider sx={{ my: 3 }} />
          
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
            Cache Configuration
          </Typography>
          <TextField
            fullWidth
            label="Cache TTL (hours)"
            type="number"
            defaultValue={24}
            variant="outlined"
            sx={{ mb: 2 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Time to live for cached normalization plans
          </Typography>
          
          <Box sx={{ mt: 4, display: 'flex', gap: 2 }}>
            <Button 
              variant="contained" 
              startIcon={<SaveIcon />}
              onClick={handleSave}
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)',
                }
              }}
            >
              Save Settings
            </Button>
            <Button variant="outlined">
              Reset to Defaults
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  )
}

export default Settings
import { useState, useEffect } from 'react'
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Alert,
  CircularProgress,
  Paper,
  Stack,
  Divider
} from '@mui/material'
import { motion } from 'framer-motion'
import PaletteIcon from '@mui/icons-material/Palette'
import ScienceIcon from '@mui/icons-material/Science'
import InfoIcon from '@mui/icons-material/Info'
import axios from 'axios'

interface ColorData {
  rgb: { r: number; g: number; b: number }
  hex: string
  description: string
  analysis: {
    ch_stretch_max: number
    ch2_rocking_max: number
    carbonyl_max: number
    max_absorbance: number
    darkness_factor: number
    spectral_features: {
      hydrocarbon_content: string
      oxidation_level: string
      sample_concentration: string
    }
    notes: string[]
  }
}

interface Props {
  fileId: string | null
  fileName: string
}

const GreaseColorPanel = ({ fileId, fileName }: Props) => {
  const [colorData, setColorData] = useState<ColorData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (fileId) {
      fetchColorData(fileId)
    } else {
      setColorData(null)
      setError(null)
    }
  }, [fileId])

  const fetchColorData = async (id: string) => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(
        `http://localhost:8000/api/analysis/color?file_id=${id}`
      )

      if (response.data.success) {
        setColorData(response.data.data)
      } else {
        setError('Failed to analyze color')
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to analyze color')
    } finally {
      setLoading(false)
    }
  }

  if (!fileId) {
    return (
      <Card
        sx={{
          background: 'rgba(30, 41, 59, 0.7)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(148, 163, 184, 0.1)',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <PaletteIcon sx={{ fontSize: 64, color: 'rgba(148, 163, 184, 0.3)', mb: 2 }} />
            <Typography variant="body1" color="text.secondary">
              Select a sample to analyze grease color
            </Typography>
          </Box>
        </CardContent>
      </Card>
    )
  }

  if (loading) {
    return (
      <Card
        sx={{
          background: 'rgba(30, 41, 59, 0.7)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(148, 163, 184, 0.1)',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <CircularProgress sx={{ mb: 2 }} />
            <Typography variant="body1" color="text.secondary">
              Analyzing grease color...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card
        sx={{
          background: 'rgba(30, 41, 59, 0.7)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(148, 163, 184, 0.1)',
          height: '100%'
        }}
      >
        <CardContent>
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        </CardContent>
      </Card>
    )
  }

  if (!colorData) {
    return null
  }

  const { rgb, hex, description, analysis } = colorData

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Card
        sx={{
          background: 'rgba(30, 41, 59, 0.7)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(148, 163, 184, 0.1)',
          height: '100%'
        }}
      >
        <CardContent>
          {/* Title */}
          <Typography
            variant="h6"
            sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}
          >
            <PaletteIcon color="primary" />
            Grease Color Analysis
          </Typography>

          {/* Sample Name */}
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Sample: {fileName}
          </Typography>

          {/* Color Swatch */}
          <Box
            sx={{
              width: '100%',
              height: 150,
              backgroundColor: hex,
              borderRadius: 2,
              mb: 2,
              border: '2px solid rgba(255, 255, 255, 0.1)',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
              position: 'relative',
              overflow: 'hidden'
            }}
          >
            <Box
              sx={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                background: 'linear-gradient(to top, rgba(0,0,0,0.7), transparent)',
                p: 1
              }}
            >
              <Typography
                variant="caption"
                sx={{
                  color: 'white',
                  fontWeight: 600,
                  textShadow: '0 1px 2px rgba(0,0,0,0.8)'
                }}
              >
                {hex}
              </Typography>
            </Box>
          </Box>

          {/* RGB & Description */}
          <Paper
            sx={{
              p: 2,
              mb: 2,
              background: 'rgba(99, 102, 241, 0.1)',
              border: '1px solid rgba(99, 102, 241, 0.2)'
            }}
          >
            <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
              RGB({rgb.r}, {rgb.g}, {rgb.b})
            </Typography>
            <Typography variant="h6" color="primary">
              {description}
            </Typography>
          </Paper>

          <Divider sx={{ my: 2 }} />

          {/* Spectral Features */}
          <Box sx={{ mb: 2 }}>
            <Typography
              variant="subtitle2"
              sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}
            >
              <ScienceIcon fontSize="small" color="secondary" />
              Spectral Features
            </Typography>
            <Stack spacing={1}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Hydrocarbon Content:
                </Typography>
                <Chip
                  label={analysis.spectral_features.hydrocarbon_content}
                  size="small"
                  color={
                    analysis.spectral_features.hydrocarbon_content === 'High'
                      ? 'success'
                      : analysis.spectral_features.hydrocarbon_content === 'Moderate'
                      ? 'warning'
                      : 'error'
                  }
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Oxidation Level:
                </Typography>
                <Chip
                  label={analysis.spectral_features.oxidation_level}
                  size="small"
                  color={
                    analysis.spectral_features.oxidation_level === 'Low'
                      ? 'success'
                      : analysis.spectral_features.oxidation_level === 'Moderate'
                      ? 'warning'
                      : 'error'
                  }
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Sample Concentration:
                </Typography>
                <Chip
                  label={analysis.spectral_features.sample_concentration}
                  size="small"
                  color="info"
                />
              </Box>
            </Stack>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Technical Values */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Technical Parameters
            </Typography>
            <Paper
              sx={{
                p: 1.5,
                background: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid rgba(148, 163, 184, 0.1)'
              }}
            >
              <Stack spacing={0.5}>
                <Typography variant="caption" color="text.secondary">
                  C-H Stretch Max: {analysis.ch_stretch_max.toFixed(4)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  CH₂ Rocking Max: {analysis.ch2_rocking_max.toFixed(4)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Carbonyl Max: {analysis.carbonyl_max.toFixed(4)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Max Absorbance: {analysis.max_absorbance.toFixed(4)}
                </Typography>
              </Stack>
            </Paper>
          </Box>

          {/* Analysis Notes */}
          {analysis.notes && analysis.notes.length > 0 && (
            <Box>
              <Typography
                variant="subtitle2"
                sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}
              >
                <InfoIcon fontSize="small" color="info" />
                Analysis Notes
              </Typography>
              <Paper
                sx={{
                  p: 1.5,
                  background: 'rgba(34, 211, 238, 0.1)',
                  border: '1px solid rgba(34, 211, 238, 0.2)'
                }}
              >
                <Stack spacing={1}>
                  {analysis.notes.map((note, index) => (
                    <Typography key={index} variant="caption" color="text.secondary">
                      • {note}
                    </Typography>
                  ))}
                </Stack>
              </Paper>
            </Box>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default GreaseColorPanel
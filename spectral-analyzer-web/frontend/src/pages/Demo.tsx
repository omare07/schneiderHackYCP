import { useState, useEffect } from 'react'
import {
  Container,
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  AlertTitle,
  Chip,
  Stack,
  Divider,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material'
import { motion } from 'framer-motion'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import DownloadIcon from '@mui/icons-material/Download'
import NavigateNextIcon from '@mui/icons-material/NavigateNext'
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore'
import ScienceIcon from '@mui/icons-material/Science'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import CompareIcon from '@mui/icons-material/Compare'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import CodeIcon from '@mui/icons-material/Code'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import toast from 'react-hot-toast'
import { getDemoScenarios, runDemo } from '../services/api'
import type { DemoScenario, DemoResponse, DemoStep } from '../services/api'

const Demo = () => {
  const [scenarios, setScenarios] = useState<DemoScenario[]>([])
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingScenarios, setLoadingScenarios] = useState(true)
  const [currentStep, setCurrentStep] = useState(0)
  const [steps, setSteps] = useState<DemoStep[]>([])
  const [executionTime, setExecutionTime] = useState(0)
  const [totalCost, setTotalCost] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [demoResponse, setDemoResponse] = useState<DemoResponse | null>(null)

  // Load available demo scenarios on mount
  useEffect(() => {
    const loadScenarios = async () => {
      try {
        const data = await getDemoScenarios()
        setScenarios(data)
      } catch (error: any) {
        toast.error('Failed to load demo scenarios')
        console.error(error)
      } finally {
        setLoadingScenarios(false)
      }
    }
    loadScenarios()
  }, [])

  const handleRunDemo = async (scenarioId: string) => {
    setSelectedScenario(scenarioId)
    setLoading(true)
    setError(null)
    setCurrentStep(0)
    setSteps([])
    setDemoResponse(null)

    try {
      toast.loading('Running demo...', { id: 'demo' })
      const response = await runDemo(scenarioId)
      
      setDemoResponse(response)
      setSteps(response.steps)
      setExecutionTime(response.execution_time)
      setTotalCost(response.total_cost)
      
      toast.success(`Demo complete! ${response.steps.length} steps executed`, { id: 'demo' })
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message
      setError(errorMsg)
      toast.error(`Demo failed: ${errorMsg}`, { id: 'demo' })
    } finally {
      setLoading(false)
    }
  }

  const handleNextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handlePrevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleDownloadGraph = () => {
    if (!steps[3] || !steps[3].data.graph_data) return
    
    const link = document.createElement('a')
    link.href = steps[3].data.graph_data
    link.download = `demo_graph_${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    toast.success('Graph downloaded!')
  }

  return (
    <Container maxWidth="xl">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box 
          sx={{ 
            textAlign: 'center', 
            mb: 4,
            p: 4,
            borderRadius: 3,
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
          }}
        >
          <Typography 
            variant="h2" 
            sx={{ 
              fontWeight: 700,
              background: 'linear-gradient(135deg, #818cf8 0%, #f472b6 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              mb: 2
            }}
          >
            AI Normalization Demo
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Experience the 4-step AI-powered spectral data normalization process
          </Typography>
        </Box>
      </motion.div>

      {/* Scenario Selection */}
      {loadingScenarios ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
            Select a Demo Scenario
          </Typography>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {scenarios.map((scenario) => (
              <Grid item xs={12} md={6} lg={4} key={scenario.id}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    background: selectedScenario === scenario.id 
                      ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%)'
                      : 'rgba(30, 41, 59, 0.7)',
                    backdropFilter: 'blur(10px)',
                    border: selectedScenario === scenario.id 
                      ? '2px solid #818cf8'
                      : '1px solid rgba(148, 163, 184, 0.1)',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      borderColor: '#818cf8',
                      boxShadow: '0 8px 24px rgba(99, 102, 241, 0.3)',
                    },
                  }}
                  onClick={() => !loading && handleRunDemo(scenario.id)}
                >
                  <CardContent>
                    <Stack direction="row" justifyContent="space-between" alignItems="flex-start" sx={{ mb: 2 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {scenario.name}
                      </Typography>
                      {selectedScenario === scenario.id && (
                        <CheckCircleIcon color="success" />
                      )}
                    </Stack>
                    <Typography variant="body2" color="text.secondary">
                      {scenario.description}
                    </Typography>
                    <Button
                      variant="contained"
                      size="small"
                      startIcon={loading ? <CircularProgress size={16} /> : <PlayArrowIcon />}
                      disabled={loading}
                      sx={{ mt: 2 }}
                      onClick={(e) => {
                        e.stopPropagation()
                        handleRunDemo(scenario.id)
                      }}
                    >
                      {loading && selectedScenario === scenario.id ? 'Running...' : 'Run Demo'}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </motion.div>
      )}

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          <AlertTitle>Demo Failed</AlertTitle>
          {error}
        </Alert>
      )}

      {/* Demo Visualization */}
      {steps.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Stepper */}
          <Card sx={{ 
            mb: 4,
            background: 'rgba(30, 41, 59, 0.7)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
          }}>
            <CardContent sx={{ p: 4 }}>
              <Stepper activeStep={currentStep} alternativeLabel>
                <Step>
                  <StepLabel 
                    icon={<ScienceIcon />}
                    StepIconProps={{
                      sx: { fontSize: 32 }
                    }}
                  >
                    Raw Data
                  </StepLabel>
                </Step>
                <Step>
                  <StepLabel 
                    icon={<AutoFixHighIcon />}
                    StepIconProps={{
                      sx: { fontSize: 32 }
                    }}
                  >
                    AI Analysis
                  </StepLabel>
                </Step>
                <Step>
                  <StepLabel 
                    icon={<CompareIcon />}
                    StepIconProps={{
                      sx: { fontSize: 32 }
                    }}
                  >
                    Normalized
                  </StepLabel>
                </Step>
                <Step>
                  <StepLabel 
                    icon={<ShowChartIcon />}
                    StepIconProps={{
                      sx: { fontSize: 32 }
                    }}
                  >
                    Graph
                  </StepLabel>
                </Step>
              </Stepper>
            </CardContent>
          </Card>

          {/* Step Content */}
          <Card sx={{ 
            mb: 4,
            background: 'rgba(30, 41, 59, 0.7)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
            minHeight: 400
          }}>
            <CardContent sx={{ p: 4 }}>
              {currentStep === 0 && <RawDataStep data={steps[0]} />}
              {currentStep === 1 && <AIAnalysisStep data={steps[1]} />}
              {currentStep === 2 && <NormalizedDataStep data={steps[2]} />}
              {currentStep === 3 && <GraphStep data={steps[3]} onDownload={handleDownloadGraph} />}

              {/* Navigation */}
              <Divider sx={{ my: 3 }} />
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                <Button
                  startIcon={<NavigateBeforeIcon />}
                  onClick={handlePrevStep}
                  disabled={currentStep === 0}
                >
                  Previous
                </Button>
                <Typography variant="body2" color="text.secondary">
                  Step {currentStep + 1} of {steps.length} • {steps[currentStep].processing_time.toFixed(2)}s
                </Typography>
                <Button
                  endIcon={<NavigateNextIcon />}
                  onClick={handleNextStep}
                  disabled={currentStep === steps.length - 1}
                >
                  Next
                </Button>
              </Stack>
            </CardContent>
          </Card>

          {/* Stats Footer */}
          {demoResponse && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card sx={{
                  background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(99, 102, 241, 0.05) 100%)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(99, 102, 241, 0.2)',
                }}>
                  <CardContent>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#818cf8' }}>
                      {executionTime.toFixed(2)}s
                    </Typography>
                    <Typography color="text.secondary">Total Execution Time</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card sx={{
                  background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.2) 0%, rgba(236, 72, 153, 0.05) 100%)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(236, 72, 153, 0.2)',
                }}>
                  <CardContent>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#f472b6' }}>
                      ${totalCost.toFixed(4)}
                    </Typography>
                    <Typography color="text.secondary">AI Processing Cost</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card sx={{
                  background: 'linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(34, 211, 238, 0.05) 100%)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(34, 211, 238, 0.2)',
                }}>
                  <CardContent>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#22d3ee' }}>
                      {demoResponse.cache_hit ? 'Yes' : 'No'}
                    </Typography>
                    <Typography color="text.secondary">Cache Hit</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}
        </motion.div>
      )}

      {/* Loading State */}
      {loading && (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 8 }}>
          <CircularProgress size={60} sx={{ mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            Running demo scenario...
          </Typography>
          <LinearProgress sx={{ width: '60%', mt: 2 }} />
        </Box>
      )}
    </Container>
  )
}

// Step 1: Raw Data Display
const RawDataStep = ({ data }: { data: DemoStep }) => {
  const stepData = data.data

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        {data.title}
      </Typography>

      {/* File Information */}
      <Paper sx={{ p: 3, mb: 3, background: 'rgba(99, 102, 241, 0.1)' }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Typography variant="body2" color="text.secondary">Filename</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {stepData.file_info?.filename || 'N/A'}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2" color="text.secondary">Delimiter</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {stepData.file_info?.delimiter || 'N/A'}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2" color="text.secondary">Encoding</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {stepData.file_info?.encoding || 'N/A'}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Raw File Contents */}
      {stepData.raw_text ? (
        <>
          <Typography variant="h6" sx={{ mb: 2 }}>File Contents (first 15 lines)</Typography>
          <Paper sx={{
            p: 0,
            mb: 3,
            bgcolor: '#1e1e1e',
            border: '1px solid rgba(148, 163, 184, 0.2)',
            overflow: 'hidden'
          }}>
            <Box sx={{
              fontFamily: 'Consolas, Monaco, "Courier New", monospace',
              fontSize: '0.85rem',
              color: '#d4d4d4',
              p: 2,
              overflow: 'auto',
              maxHeight: 400,
              lineHeight: 1.6,
              whiteSpace: 'pre'
            }}>
              {stepData.raw_text.map((line: string, i: number) => (
                <Box
                  key={i}
                  sx={{
                    display: 'flex',
                    '&:hover': { bgcolor: 'rgba(99, 102, 241, 0.1)' }
                  }}
                >
                  <Box
                    component="span"
                    sx={{
                      display: 'inline-block',
                      minWidth: '3em',
                      color: '#858585',
                      textAlign: 'right',
                      pr: 2,
                      userSelect: 'none'
                    }}
                  >
                    {i + 1}
                  </Box>
                  <Box component="span" sx={{ flex: 1 }}>
                    {line || ' '}
                  </Box>
                </Box>
              ))}
            </Box>
          </Paper>
        </>
      ) : stepData.raw_preview && stepData.raw_preview.length > 0 ? (
        <>
          <Typography variant="h6" sx={{ mb: 2 }}>Data Preview</Typography>
          <TableContainer component={Paper} sx={{ mb: 3, maxHeight: 300 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  {Object.keys(stepData.raw_preview[0] || {}).map((key) => (
                    <TableCell key={key} sx={{ fontWeight: 600, background: 'rgba(99, 102, 241, 0.1)' }}>
                      {key}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {stepData.raw_preview.map((row: any, idx: number) => (
                  <TableRow key={idx}>
                    {Object.values(row).map((value: any, cellIdx: number) => (
                      <TableCell key={cellIdx}>{String(value)}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      ) : null}

      {/* Detected Issues */}
      {stepData.detected_issues && stepData.detected_issues.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mb: 2 }}>Detected Issues</Typography>
          <Stack spacing={2}>
            {stepData.detected_issues.map((issue: string, idx: number) => (
              <Alert key={idx} severity="warning">
                {issue}
              </Alert>
            ))}
          </Stack>
        </>
      )}
    </Box>
  )
}

// Step 2: AI Analysis Display
const AIAnalysisStep = ({ data }: { data: DemoStep }) => {
  const stepData = data.data

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        {data.title}
      </Typography>

      {/* AI Model Info */}
      <Paper sx={{ p: 3, mb: 3, background: 'rgba(99, 102, 241, 0.1)' }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              AI Model
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {stepData.ai_model || 'N/A'}
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Confidence Score
            </Typography>
            <Stack direction="row" alignItems="center" spacing={2}>
              <CircularProgress 
                variant="determinate" 
                value={stepData.confidence_score || 0} 
                size={60}
                thickness={6}
                sx={{ color: stepData.confidence_score > 80 ? '#22d3ee' : '#f59e0b' }}
              />
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                {stepData.confidence_score?.toFixed(1) || 0}%
              </Typography>
            </Stack>
          </Grid>
        </Grid>
      </Paper>

      {/* Column Mappings */}
      {stepData.column_mappings && stepData.column_mappings.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mb: 2 }}>Column Mappings</Typography>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            {stepData.column_mappings.map((mapping: any, idx: number) => (
              <Grid item xs={12} sm={6} md={4} key={idx}>
                <Card sx={{ 
                  background: 'rgba(236, 72, 153, 0.1)',
                  border: '1px solid rgba(236, 72, 153, 0.3)'
                }}>
                  <CardContent>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {mapping.original_name}
                    </Typography>
                    <Typography variant="body1" sx={{ fontWeight: 600, mb: 1 }}>
                      → {mapping.target_name}
                    </Typography>
                    <Chip
                      label={`${mapping.confidence?.toFixed(0) || 0}%`}
                      size="small"
                      color={mapping.confidence > 80 ? 'success' : 'warning'}
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </>
      )}

      {/* Transformations Planned */}
      {stepData.transformations_planned && stepData.transformations_planned.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mb: 2 }}>Transformations Planned</Typography>
          <Stack spacing={1}>
            {stepData.transformations_planned.map((transform: string, idx: number) => (
              <Paper key={idx} sx={{ p: 2, background: 'rgba(34, 211, 238, 0.1)' }}>
                <Stack direction="row" alignItems="center" spacing={2}>
                  <CheckCircleIcon color="success" />
                  <Typography variant="body2">{transform}</Typography>
                </Stack>
              </Paper>
            ))}
          </Stack>
        </>
      )}

      {/* Raw AI Response */}
      {stepData.raw_ai_response && (
        <Accordion sx={{ mt: 3, background: 'rgba(99, 102, 241, 0.05)', border: '1px solid rgba(99, 102, 241, 0.2)' }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CodeIcon />
              Raw AI Response (Click to expand)
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              This is the actual JSON response from the AI model showing how it analyzed and normalized the data:
            </Typography>
            <Box
              sx={{
                bgcolor: '#1e1e1e',
                color: '#d4d4d4',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                maxHeight: 400,
                fontFamily: 'monospace',
                fontSize: '0.875rem',
                lineHeight: 1.5
              }}
            >
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {JSON.stringify(stepData.raw_ai_response, null, 2)}
              </pre>
            </Box>
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  )
}

// Step 3: Normalized Data Display
const NormalizedDataStep = ({ data }: { data: DemoStep }) => {
  const stepData = data.data

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        {data.title}
      </Typography>

      {/* Statistics */}
      {stepData.statistics && (
        <Paper sx={{ p: 3, mb: 3, background: 'rgba(99, 102, 241, 0.1)' }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Dataset Statistics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="text.secondary">Rows</Typography>
              <Typography variant="h6">{stepData.statistics.row_count || 0}</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="text.secondary">Columns</Typography>
              <Typography variant="h6">{stepData.statistics.column_count || 0}</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="text.secondary">Wavenumber Range</Typography>
              <Typography variant="body2">
                {stepData.statistics.wavenumber_range
                  ? `${stepData.statistics.wavenumber_range.min.toFixed(0)} - ${stepData.statistics.wavenumber_range.max.toFixed(0)}`
                  : 'N/A'}
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="body2" color="text.secondary">Intensity Range</Typography>
              <Typography variant="body2">
                {stepData.statistics.intensity_range
                  ? `${stepData.statistics.intensity_range.min.toFixed(2)} - ${stepData.statistics.intensity_range.max.toFixed(2)}`
                  : 'N/A'}
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Normalized Data Preview */}
      {stepData.preview && stepData.preview.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mb: 2 }}>Normalized Data Preview</Typography>
          <TableContainer component={Paper} sx={{ maxHeight: 400, mb: 3 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  {Object.keys(stepData.preview[0] || {}).map((key) => (
                    <TableCell key={key} sx={{ fontWeight: 600, background: 'rgba(34, 211, 238, 0.1)' }}>
                      {key}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {stepData.preview.map((row: any, idx: number) => (
                  <TableRow key={idx}>
                    {Object.values(row).map((value: any, cellIdx: number) => (
                      <TableCell key={cellIdx}>{String(value)}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Download CSV Button */}
          {stepData.normalized_csv && (
            <Box sx={{ textAlign: 'center' }}>
              <Button
                variant="contained"
                startIcon={<DownloadIcon />}
                onClick={() => {
                  const blob = new Blob([stepData.normalized_csv], { type: 'text/csv' })
                  const url = window.URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = stepData.normalized_csv_filename || 'normalized_data.csv'
                  a.click()
                  window.URL.revokeObjectURL(url)
                  toast.success('CSV downloaded!')
                }}
                sx={{
                  background: 'linear-gradient(135deg, #22d3ee 0%, #06b6d4 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%)',
                  }
                }}
              >
                Download Normalized CSV
              </Button>
            </Box>
          )}
        </>
      )}
    </Box>
  )
}

// Step 4: Graph Display
const GraphStep = ({ data, onDownload }: { data: DemoStep; onDownload: () => void }) => {
  const stepData = data.data

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        {data.title}
      </Typography>

      {/* Graph Metadata */}
      <Paper sx={{ p: 3, mb: 3, background: 'rgba(99, 102, 241, 0.1)' }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="text.secondary">Baseline</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {stepData.baseline_name || 'N/A'}
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="text.secondary">Sample</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {stepData.sample_name || 'N/A'}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Graph Display */}
      {stepData.graph_data && (
        <Box sx={{ textAlign: 'center' }}>
          <Paper 
            sx={{ 
              p: 2, 
              mb: 3, 
              background: '#ffffff',
              display: 'inline-block',
              maxWidth: '100%'
            }}
          >
            <img 
              src={stepData.graph_data} 
              alt="Spectral Comparison Graph"
              style={{ 
                maxWidth: '100%', 
                height: 'auto',
                display: 'block'
              }}
            />
          </Paper>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={onDownload}
            sx={{
              background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)',
              }
            }}
          >
            Download Graph
          </Button>
        </Box>
      )}
    </Box>
  )
}

export default Demo
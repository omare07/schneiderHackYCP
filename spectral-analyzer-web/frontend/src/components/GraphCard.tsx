import { useState, useRef } from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Button,
  IconButton,
  Menu,
  MenuItem,
  Collapse,
  CircularProgress,
  Alert,
  Stack,
  Divider,
  MobileStepper
} from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import ExpandIcon from '@mui/icons-material/Fullscreen';
import DownloadIcon from '@mui/icons-material/Download';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import KeyboardArrowLeft from '@mui/icons-material/KeyboardArrowLeft';
import KeyboardArrowRight from '@mui/icons-material/KeyboardArrowRight';
import InterpretationReportView from './InterpretationReportView';
import GreaseColorPanel from './GreaseColorPanel';

interface GraphWithInterpretation {
  id: string;
  sampleName: string;
  sampleFileId: string;
  data: string;
  interpretation: any | null;
  interpretationLoading: boolean;
  interpretationError: string | null;
  reportExpanded: boolean;
  modalOpen: boolean;
}

interface GraphCardProps {
  graph: GraphWithInterpretation;
  baselineFileId: string;
  onAnalyze: (graphId: string) => Promise<void>;
  onDownloadGraph: (graphId: string) => void;
  onDownloadWithReport: (graphId: string) => void;
  onExpandView: (graphId: string) => void;
  onToggleReport: (graphId: string) => void;
}

export const GraphCard: React.FC<GraphCardProps> = ({
  graph,
  baselineFileId,
  onAnalyze,
  onDownloadGraph,
  onDownloadWithReport,
  onExpandView,
  onToggleReport
}) => {
  const [downloadMenuAnchor, setDownloadMenuAnchor] = useState<null | HTMLElement>(null);
  const [activeStep, setActiveStep] = useState(0);
  const graphRef = useRef<HTMLDivElement>(null);

  const hasInterpretation = !!graph.interpretation;
  const isLoading = graph.interpretationLoading;
  const hasError = !!graph.interpretationError;

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  return (
    <Card sx={{ 
      background: 'rgba(30, 41, 59, 0.7)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(148, 163, 184, 0.1)',
      height: '100%',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header with actions */}
        <Stack 
          direction="row" 
          justifyContent="space-between" 
          alignItems="center" 
          sx={{ mb: 2 }}
        >
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {graph.sampleName}
          </Typography>
          
          <Stack direction="row" spacing={1}>
            {/* Analyze with AI button */}
            {!hasInterpretation && !isLoading && (
              <Button
                variant="outlined"
                size="small"
                startIcon={<ScienceIcon />}
                onClick={() => onAnalyze(graph.id)}
                sx={{
                  borderColor: 'rgba(244, 114, 182, 0.5)',
                  color: '#f472b6',
                  '&:hover': {
                    borderColor: '#f472b6',
                    background: 'rgba(244, 114, 182, 0.1)'
                  }
                }}
              >
                Analyze with AI
              </Button>
            )}
            
            {/* Loading indicator */}
            {isLoading && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CircularProgress size={20} />
                <Typography variant="body2" color="text.secondary">
                  Analyzing...
                </Typography>
              </Box>
            )}
            
            {/* Expand button */}
            <IconButton 
              size="small" 
              onClick={() => onExpandView(graph.id)}
              title="View Full Screen"
              sx={{ color: 'rgba(148, 163, 184, 0.8)' }}
            >
              <ExpandIcon />
            </IconButton>
            
            {/* Download menu */}
            <IconButton
              size="small"
              onClick={(e) => setDownloadMenuAnchor(e.currentTarget)}
              title="Download Options"
              sx={{ color: 'rgba(148, 163, 184, 0.8)' }}
            >
              <DownloadIcon />
            </IconButton>
            
            <Menu
              anchorEl={downloadMenuAnchor}
              open={Boolean(downloadMenuAnchor)}
              onClose={() => setDownloadMenuAnchor(null)}
            >
              <MenuItem onClick={() => {
                onDownloadGraph(graph.id);
                setDownloadMenuAnchor(null);
              }}>
                Download Graph Only (PNG)
              </MenuItem>
              <MenuItem 
                onClick={() => {
                  onDownloadWithReport(graph.id);
                  setDownloadMenuAnchor(null);
                }}
                disabled={!hasInterpretation}
              >
                Download with Report (PDF)
              </MenuItem>
            </Menu>
          </Stack>
        </Stack>

        <Divider sx={{ mb: 2 }} />

        {/* Carousel Content */}
        <Box sx={{ mb: 2, minHeight: 400 }}>
          {activeStep === 0 && (
            /* Graph Image */
            <Box
              ref={graphRef}
              sx={{
                textAlign: 'center',
                background: 'white',
                borderRadius: 2,
                p: 1,
                '& img': {
                  maxWidth: '100%',
                  height: 'auto',
                  borderRadius: 1
                }
              }}
            >
              <img
                src={graph.data}
                alt={`Spectral Comparison - ${graph.sampleName}`}
                id={`graph-${graph.id}`}
              />
            </Box>
          )}

          {activeStep === 1 && (
            /* Color Analysis Panel */
            <GreaseColorPanel
              fileId={graph.sampleFileId}
              fileName={graph.sampleName}
            />
          )}
        </Box>

        {/* Carousel Navigation */}
        <MobileStepper
          steps={2}
          position="static"
          activeStep={activeStep}
          sx={{
            background: 'transparent',
            mb: 2,
            '& .MuiMobileStepper-dot': {
              backgroundColor: 'rgba(148, 163, 184, 0.3)',
            },
            '& .MuiMobileStepper-dotActive': {
              backgroundColor: '#818cf8',
            }
          }}
          nextButton={
            <Button
              size="small"
              onClick={handleNext}
              disabled={activeStep === 1}
              sx={{
                color: activeStep === 1 ? 'rgba(148, 163, 184, 0.3)' : '#818cf8',
              }}
            >
              Color Analysis
              <KeyboardArrowRight />
            </Button>
          }
          backButton={
            <Button
              size="small"
              onClick={handleBack}
              disabled={activeStep === 0}
              sx={{
                color: activeStep === 0 ? 'rgba(148, 163, 184, 0.3)' : '#818cf8',
              }}
            >
              <KeyboardArrowLeft />
              Graph
            </Button>
          }
        />

        {/* Interpretation Report Section */}
        {hasError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {graph.interpretationError}
          </Alert>
        )}

        {hasInterpretation && (
          <>
            <Stack 
              direction="row" 
              justifyContent="space-between" 
              alignItems="center"
              sx={{ 
                cursor: 'pointer',
                p: 1,
                borderRadius: 1,
                '&:hover': { background: 'rgba(255, 255, 255, 0.05)' }
              }}
              onClick={() => onToggleReport(graph.id)}
            >
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                AI Interpretation Report
              </Typography>
              <IconButton size="small">
                {graph.reportExpanded ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
              </IconButton>
            </Stack>

            <Collapse in={graph.reportExpanded}>
              <Box sx={{ mt: 2 }}>
                <InterpretationReportView report={graph.interpretation} />
              </Box>
            </Collapse>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default GraphCard;
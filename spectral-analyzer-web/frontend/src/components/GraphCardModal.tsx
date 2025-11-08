import { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  IconButton,
  Grid,
  Alert,
  Typography,
  MobileStepper
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DownloadIcon from '@mui/icons-material/Download';
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

interface GraphCardModalProps {
  graph: GraphWithInterpretation | null;
  open: boolean;
  onClose: () => void;
  onDownloadGraph: (graphId: string) => void;
  onDownloadWithReport: (graphId: string) => void;
}

export const GraphCardModal: React.FC<GraphCardModalProps> = ({
  graph,
  open,
  onClose,
  onDownloadGraph,
  onDownloadWithReport
}) => {
  const [activeStep, setActiveStep] = useState(0);

  if (!graph) return null;

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleClose = () => {
    setActiveStep(0); // Reset to first slide when closing
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          background: 'rgba(15, 23, 42, 0.95)',
          backdropFilter: 'blur(10px)',
          minHeight: '90vh',
          border: '1px solid rgba(148, 163, 184, 0.2)'
        }
      }}
    >
      <DialogTitle sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
        pb: 2
      }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          {graph.sampleName} - Full View
        </Typography>
        <IconButton onClick={handleClose} sx={{ color: 'rgba(148, 163, 184, 0.8)' }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 3 }}>
        {/* Carousel Navigation at Top */}
        <MobileStepper
          steps={3}
          position="static"
          activeStep={activeStep}
          sx={{
            background: 'transparent',
            mb: 3,
            justifyContent: 'center',
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
              disabled={activeStep === 2}
              sx={{
                color: activeStep === 2 ? 'rgba(148, 163, 184, 0.3)' : '#818cf8',
              }}
            >
              {activeStep === 0 ? 'Color Analysis' : 'AI Report'}
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
              {activeStep === 2 ? 'Color Analysis' : 'Graph'}
            </Button>
          }
        />

        {/* Carousel Content */}
        <Grid container spacing={3}>
          {activeStep === 0 && (
            /* Slide 1: Graph */
            <Grid item xs={12}>
              <Box
                sx={{
                  background: 'white',
                  borderRadius: 2,
                  p: 2,
                  '& img': {
                    width: '100%',
                    height: 'auto',
                    borderRadius: 1
                  }
                }}
              >
                <img
                  src={graph.data}
                  alt={graph.sampleName}
                  id={`modal-graph-${graph.id}`}
                />
              </Box>
            </Grid>
          )}

          {activeStep === 1 && (
            /* Slide 2: Color Analysis */
            <Grid item xs={12}>
              <Box sx={{ maxWidth: 800, mx: 'auto' }}>
                <GreaseColorPanel
                  fileId={graph.sampleFileId}
                  fileName={graph.sampleName}
                />
              </Box>
            </Grid>
          )}

          {activeStep === 2 && (
            /* Slide 3: AI Report */
            <Grid item xs={12}>
              {graph.interpretation ? (
                <Box sx={{
                  background: 'rgba(30, 41, 59, 0.5)',
                  borderRadius: 2,
                  p: 3,
                  maxHeight: '70vh',
                  overflowY: 'auto',
                  maxWidth: 1200,
                  mx: 'auto'
                }}>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    AI Interpretation Report
                  </Typography>
                  <InterpretationReportView report={graph.interpretation} compact={false} />
                </Box>
              ) : (
                <Alert severity="info">
                  No interpretation available. Close this modal and click "Analyze with AI" to generate a report.
                </Alert>
              )}
            </Grid>
          )}
        </Grid>
      </DialogContent>

      <DialogActions sx={{ 
        borderTop: '1px solid rgba(148, 163, 184, 0.1)',
        p: 2,
        gap: 1 
      }}>
        <Button 
          onClick={() => onDownloadGraph(graph.id)}
          startIcon={<DownloadIcon />}
          variant="outlined"
        >
          Download Graph (PNG)
        </Button>
        <Button
          onClick={() => onDownloadWithReport(graph.id)}
          disabled={!graph.interpretation}
          variant="contained"
          startIcon={<DownloadIcon />}
          sx={{
            background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)',
            },
            '&:disabled': {
              background: 'rgba(148, 163, 184, 0.2)',
            }
          }}
        >
          Download with Report (PDF)
        </Button>
        <Button onClick={handleClose} variant="text">
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default GraphCardModal;
import {
  Box,
  Grid,
  Paper,
  Typography,
  Alert,
  AlertTitle,
  Chip,
  Stack
} from '@mui/material';

interface InterpretationReportViewProps {
  report: any;
  compact?: boolean;
}

export const InterpretationReportView: React.FC<InterpretationReportViewProps> = ({
  report,
  compact = false
}) => {
  // Safe accessor helper to prevent crashes from inconsistent AI response structures
  const safeGet = (obj: any, path: string, defaultValue: any = null) => {
    try {
      return path.split('.').reduce((current, prop) => current?.[prop], obj) ?? defaultValue;
    } catch {
      return defaultValue;
    }
  };

  if (!report) {
    return (
      <Alert severity="info">
        No interpretation available. Click "Analyze with AI" to generate a report.
      </Alert>
    );
  }

  return (
    <Box>
      <Alert severity="info" sx={{ mb: 3 }}>
        <AlertTitle>{safeGet(report, 'analysis.summary', 'Analysis Complete')}</AlertTitle>
        {safeGet(report, 'analysis.spectrum_type') && `Spectrum Type: ${safeGet(report, 'analysis.spectrum_type')}`}
        {safeGet(report, 'analysis.confidence') && ` | Confidence: ${safeGet(report, 'analysis.confidence')}`}
      </Alert>

      <Grid container spacing={3}>
        {/* Grease Condition Assessment */}
        <Grid item xs={12} md={compact ? 12 : 6}>
          <Paper sx={{ p: 3, background: 'rgba(99, 102, 241, 0.1)', height: '100%' }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
              Grease Condition Assessment
            </Typography>
            {safeGet(report, 'analysis.grease_condition_assessment') ? (
              <>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Overall Health:</strong> {
                    safeGet(report, 'analysis.grease_condition_assessment.overall_grease_health') ||
                    safeGet(report, 'analysis.grease_condition_assessment.overall_health', 'N/A')
                  }
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Oxidation Signs:</strong> {safeGet(report, 'analysis.grease_condition_assessment.oxidation_signs', 'None detected')}
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  <strong>Base Oil Condition:</strong> {safeGet(report, 'analysis.grease_condition_assessment.base_oil_condition', 'N/A')}
                </Typography>
                {safeGet(report, 'analysis.grease_condition_assessment.notes') && (
                  <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
                    {safeGet(report, 'analysis.grease_condition_assessment.notes')}
                  </Typography>
                )}
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Grease condition data not available
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Contamination Detection */}
        <Grid item xs={12} md={compact ? 12 : 6}>
          <Paper sx={{ p: 3, background: 'rgba(236, 72, 153, 0.1)', height: '100%' }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
              Contamination Detection
            </Typography>
            {safeGet(report, 'analysis.contamination_detection') ? (
              <>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Water Contamination:</strong> {safeGet(report, 'analysis.contamination_detection.water_contamination', 'Not detected')}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Fuel Dilution:</strong> {safeGet(report, 'analysis.contamination_detection.fuel_dilution', 'Not detected')}
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  <strong>Particulate:</strong> {safeGet(report, 'analysis.contamination_detection.particulate_contamination', 'Not detected')}
                </Typography>
                {safeGet(report, 'analysis.contamination_detection.notes') && (
                  <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
                    {safeGet(report, 'analysis.contamination_detection.notes')}
                  </Typography>
                )}
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Contamination data not available
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Functional Groups */}
        {(() => {
          const groups = safeGet(report, 'analysis.functional_group_analysis.key_functional_groups');
          if (!groups) return null;
          
          return (
            <Grid item xs={12}>
              <Paper sx={{ p: 3, background: 'rgba(34, 211, 238, 0.1)' }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                  Identified Functional Groups
                </Typography>
                {(() => {
                  // Handle string type
                  if (typeof groups === 'string') {
                    return <Typography variant="body2">{groups}</Typography>;
                  }
                  
                  // Handle array format
                  if (Array.isArray(groups)) {
                    return groups.map((group: any, idx: number) => {
                      if (typeof group === 'string') {
                        return (
                          <Chip
                            key={idx}
                            label={group}
                            sx={{ m: 0.5 }}
                          />
                        )
                      } else {
                        return (
                          <Chip
                            key={idx}
                            label={`${group.functional_group || group.name || `Group ${idx + 1}`}: ${group.expected_wavenumber || group.wavenumber_range || 'N/A'}`}
                            sx={{ m: 0.5 }}
                          />
                        )
                      }
                    });
                  }
                  
                  // Handle object format
                  if (typeof groups === 'object') {
                    return Object.entries(groups).map(([key, value]: [string, any]) => (
                      <Chip
                        key={key}
                        label={`${key}: ${typeof value === 'string' ? value : value?.expected_wavenumber || 'N/A'}`}
                        sx={{ m: 0.5 }}
                      />
                    ));
                  }
                  
                  return null;
                })()}
              </Paper>
            </Grid>
          );
        })()}

        {/* Recommendations */}
        {safeGet(report, 'analysis.recommendations') && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3, background: 'rgba(251, 191, 36, 0.1)' }}>
              <Alert severity="info" sx={{ mb: 2 }}>
                <AlertTitle>Equipment Health & Recommendations</AlertTitle>
                {safeGet(report, 'analysis.recommendations.equipment_health_assessment', 'Assessment in progress')}
              </Alert>
              {safeGet(report, 'analysis.recommendations.recommended_actions') && (
                <>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>Recommended Actions:</Typography>
                  <Stack spacing={1} sx={{ mb: 2 }}>
                    {(() => {
                      const actions = safeGet(report, 'analysis.recommendations.recommended_actions');
                      if (Array.isArray(actions)) {
                        return actions.map((action: string, idx: number) => (
                          <Chip key={idx} label={`${idx + 1}. ${action}`} size="small" color="warning" />
                        ));
                      } else if (typeof actions === 'string') {
                        return <Chip label={actions} size="small" color="warning" />;
                      }
                      return null;
                    })()}
                  </Stack>
                </>
              )}
              {safeGet(report, 'analysis.recommendations.urgency_level') && (
                <Typography variant="body2">
                  <strong>Urgency Level:</strong>{' '}
                  <Chip
                    label={safeGet(report, 'analysis.recommendations.urgency_level')}
                    size="small"
                    color={
                      safeGet(report, 'analysis.recommendations.urgency_level') === 'High' ? 'error' :
                      safeGet(report, 'analysis.recommendations.urgency_level') === 'Medium' ? 'warning' : 'success'
                    }
                  />
                </Typography>
              )}
            </Paper>
          </Grid>
        )}

        {/* Key Spectral Features / Peak Analysis */}
        {safeGet(report, 'analysis.key_spectral_features.major_peaks') && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3, background: 'rgba(74, 222, 128, 0.1)' }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Key Spectral Features
              </Typography>
              <Typography variant="body2" sx={{ mb: 2, fontStyle: 'italic' }}>
                {safeGet(report, 'analysis.key_spectral_features.baseline_characteristics', 'Analyzing spectral features...')}
              </Typography>
              <Grid container spacing={2}>
                {(() => {
                  const peaks = safeGet(report, 'analysis.key_spectral_features.major_peaks');
                  if (Array.isArray(peaks)) {
                    return peaks.slice(0, 6).map((peak: any, idx: number) => (
                      <Grid item xs={12} sm={6} md={4} key={idx}>
                        <Paper sx={{ p: 2, background: 'rgba(255, 255, 255, 0.05)' }}>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {peak.wavenumber || 'N/A'} cm⁻¹
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {peak.functional_group}
                          </Typography>
                          <Typography variant="body2" sx={{ mt: 1, fontSize: '0.85rem' }}>
                            {peak.significance}
                          </Typography>
                        </Paper>
                      </Grid>
                    ));
                  } else if (typeof peaks === 'object') {
                    return Object.entries(peaks).slice(0, 6).map(([key, value]: [string, any]) => (
                      <Grid item xs={12} sm={6} md={4} key={key}>
                        <Paper sx={{ p: 2, background: 'rgba(255, 255, 255, 0.05)' }}>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {key}
                          </Typography>
                          <Typography variant="body2" sx={{ mt: 1, fontSize: '0.85rem' }}>
                            {typeof value === 'string' ? value : value?.significance || 'No description'}
                          </Typography>
                        </Paper>
                      </Grid>
                    ));
                  }
                  return null;
                })()}
              </Grid>
            </Paper>
          </Grid>
        )}

        {/* Anomalies / Unusual Patterns */}
        {(() => {
          const anomalies = safeGet(report, 'analysis.key_spectral_features.anomalies');
          if (!anomalies || (Array.isArray(anomalies) && anomalies.length === 0)) return null;
          
          return (
            <Grid item xs={12}>
              <Paper sx={{ p: 3, background: 'rgba(251, 146, 60, 0.1)' }}>
                <Alert severity="warning" sx={{ mb: 2 }}>
                  <AlertTitle>Anomalies or Unusual Patterns Detected</AlertTitle>
                  {Array.isArray(anomalies) ? `${anomalies.length} anomaly(ies) found in the spectrum` : 'Anomalies detected in the spectrum'}
                </Alert>
                <Stack spacing={1}>
                  {Array.isArray(anomalies) ? (
                    anomalies.map((anomaly: string, idx: number) => (
                      <Paper key={idx} sx={{ p: 2, background: 'rgba(255, 255, 255, 0.05)' }}>
                        <Typography variant="body2">
                          • {anomaly}
                        </Typography>
                      </Paper>
                    ))
                  ) : typeof anomalies === 'string' ? (
                    <Paper sx={{ p: 2, background: 'rgba(255, 255, 255, 0.05)' }}>
                      <Typography variant="body2">
                        • {anomalies}
                      </Typography>
                    </Paper>
                  ) : null}
                </Stack>
              </Paper>
            </Grid>
          );
        })()}
      </Grid>
    </Box>
  );
};

export default InterpretationReportView;
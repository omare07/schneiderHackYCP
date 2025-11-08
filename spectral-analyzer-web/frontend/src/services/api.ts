import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface FileInfo {
  file_id: string;
  filename: string;
  size: number;
}

export interface NormalizationPlan {
  file_hash: string;
  column_mappings: Array<{
    original_name: string;
    target_name: string;
    confidence: number;
  }>;
  confidence_score: number;
  ai_model: string;
}

export interface CacheStats {
  total_requests: number;
  cache_hits: number;
  hit_rate: number;
  memory_usage_mb: number;
}

export interface InterpretationReport {
  spectrum_type: string;
  quality_assessment: {
    overall_quality: string;
    quality_score: number;
    issues: string[];
    recommendations: string[];
  };
  chemical_composition: {
    likely_compounds: string[];
    functional_groups: Array<{
      name: string;
      wavenumber_range: string;
      confidence: string;
      notes: string;
    }>;
    composition_notes: string;
  };
  peak_analysis: {
    major_peaks: Array<{
      wavenumber: number;
      intensity: number;
      assignment: string;
      significance: string;
    }>;
    peak_patterns: string;
  };
  anomalies: {
    detected: boolean;
    anomaly_list: Array<{
      type: string;
      location: string;
      severity: string;
      description: string;
    }>;
  };
  comparison_insights: {
    typical_characteristics: string;
    unusual_features: string;
    reference_comparisons: string;
  };
  summary: string;
  confidence: string;
}

// File operations
export const uploadFile = async (file: File): Promise<FileInfo> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post('/files/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data.data;
};

export const uploadBatchFiles = async (files: File[]) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  
  const response = await apiClient.post('/files/upload-batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data.data;
};

export const listFiles = async () => {
  const response = await apiClient.get('/files/list');
  return response.data.data;
};

export const deleteFile = async (fileId: string) => {
  const response = await apiClient.delete(`/files/${fileId}`);
  return response.data;
};

export const getFileInfo = async (fileId: string) => {
  const response = await apiClient.get(`/files/${fileId}/info`);
  return response.data.data;
};

// Analysis operations
export const parseCSV = async (fileId: string) => {
  const response = await apiClient.post(`/analysis/parse?file_id=${fileId}`);
  return response.data.data;
};

export const normalizeCSV = async (fileId: string, forceRefresh = false) => {
  const response = await apiClient.post('/analysis/normalize', {
    file_id: fileId,
    force_refresh: forceRefresh,
  });
  return response.data.data;
};

export const validateData = async (fileId: string) => {
  const response = await apiClient.post(`/analysis/validate?file_id=${fileId}`);
  return response.data.data;
};

export const getAIStatus = async () => {
  const response = await apiClient.get('/analysis/ai-status');
  return response.data.data;
};

export const interpretSpectralData = async (fileId: string) => {
  const response = await apiClient.post(`/analysis/interpret?file_id=${fileId}`);
  return response.data.data;
};

export const analyzeGreaseColor = async (fileId: string) => {
  const response = await apiClient.post(`/analysis/color?file_id=${fileId}`);
  return response.data.data;
};

// Graph operations
export const generateGraph = async (
  baselineFileId: string,
  sampleFileId: string,
  format = 'png'
) => {
  const response = await apiClient.post('/graphs/generate', {
    baseline_file_id: baselineFileId,
    sample_file_id: sampleFileId,
    format,
  });
  return response.data.data;
};

export const generateBatchGraphs = async (
  baselineFileId: string,
  sampleFileIds: string[],
  format = 'png'
) => {
  const response = await apiClient.post('/graphs/generate-batch', {
    baseline_file_id: baselineFileId,
    sample_file_ids: sampleFileIds,
    format,
  });
  return response.data.data;
};

export const getGraph = async (graphId: string) => {
  const response = await apiClient.get(`/graphs/${graphId}/base64`);
  return response.data.data;
};

// Statistics operations
export const getCacheStats = async (): Promise<CacheStats> => {
  const response = await apiClient.get('/stats/cache');
  return response.data.data;
};

export const getCostStats = async () => {
  const response = await apiClient.get('/stats/costs');
  return response.data.data;
};

export const clearCache = async () => {
  const response = await apiClient.post('/stats/cache/clear');
  return response.data;
};

export const cleanupExpired = async () => {
  const response = await apiClient.delete('/stats/cache/expired');
  return response.data;
};

// Demo interfaces
export interface DemoScenario {
  id: string
  name: string
  description: string
}

export interface DemoStep {
  step_number: number
  step_name: string
  title: string
  data: any
  timestamp: string
  processing_time: number
}

export interface DemoResponse {
  scenario_id: string
  scenario_name: string
  scenario_description: string
  steps: DemoStep[]
  execution_time: number
  total_cost: number
  cache_hit: boolean
}

// Demo API functions
export const getDemoScenarios = async (): Promise<DemoScenario[]> => {
  const response = await apiClient.get('/demo/scenarios')
  return response.data.data.scenarios
}

export const runDemo = async (scenarioId: string): Promise<DemoResponse> => {
  const response = await apiClient.post('/demo/run', {
    scenario: scenarioId
  })
  
  // Map backend response to frontend interface
  const backendData = response.data.data
  return {
    scenario_id: backendData.scenario.id,
    scenario_name: backendData.scenario.name,
    scenario_description: backendData.scenario.description,
    steps: backendData.steps.map((step: any) => ({
      step_number: step.step_number,
      step_name: step.step_name,
      title: step.description,
      data: step.data || {},
      timestamp: new Date().toISOString(),
      processing_time: step.execution_time
    })),
    execution_time: backendData.total_execution_time,
    total_cost: backendData.total_cost,
    cache_hit: backendData.steps[1]?.data?.cache_hit || false
  }
}

// WebSocket for progress updates
export const createProgressWebSocket = (sessionId: string) => {
  const ws = new WebSocket(`ws://localhost:8000/api/ws/progress/${sessionId}`);
  return ws;
};

export default apiClient;
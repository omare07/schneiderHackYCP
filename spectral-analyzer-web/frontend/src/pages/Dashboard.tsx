// File System Access API types
declare global {
  interface Window {
    showSaveFilePicker?: (options?: {
      suggestedName?: string
      types?: Array<{
        description: string
        accept: Record<string, string[]>
      }>
    }) => Promise<FileSystemFileHandle>
  }
  
  interface FileSystemFileHandle {
    createWritable(): Promise<FileSystemWritableFileStream>
  }
  
  interface FileSystemWritableFileStream {
    write(data: Blob | BufferSource | string): Promise<void>
    close(): Promise<void>
  }
}

import { useState, useCallback, useEffect } from 'react'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Paper,
  Chip,
  Alert,
  AlertTitle,
  Divider,
  IconButton,
  Stack,
  Menu,
  MenuItem,
  CircularProgress,
  FormControl,
  InputLabel,
  Select
} from '@mui/material'
import { motion } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import ScienceIcon from '@mui/icons-material/Science'
import ShowChartIcon from '@mui/icons-material/ShowChart'
import DeleteIcon from '@mui/icons-material/Delete'
import DownloadIcon from '@mui/icons-material/Download'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import toast from 'react-hot-toast'
import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'
import JSZip from 'jszip'
import axios from 'axios'
import { uploadFile, normalizeCSV, generateGraph, getCacheStats, interpretSpectralData } from '../services/api'
import type { FileInfo, CacheStats } from '../services/api'
import GraphCard from '../components/GraphCard'
import GraphCardModal from '../components/GraphCardModal'
import GreaseColorPanel from '../components/GreaseColorPanel'

interface UploadedFile extends FileInfo {
  type: 'baseline' | 'sample'
}

interface NormalizationResult {
  file_hash: string
  column_mappings: Array<{
    original_name: string
    target_name: string
    confidence: number
  }>
  confidence_score: number
  ai_model: string
  transformations_applied?: string[]
}

interface GraphWithInterpretation {
  id: string
  sampleName: string
  sampleFileId: string
  data: string
  interpretation: any | null
  interpretationLoading: boolean
  interpretationError: string | null
  reportExpanded: boolean
  modalOpen: boolean
}

const Dashboard = () => {
  const [baselineFile, setBaselineFile] = useState<UploadedFile | null>(null)
  const [sampleFiles, setSampleFiles] = useState<UploadedFile[]>([])
  const [uploading, setUploading] = useState(false)
  const [normalizing, setNormalizing] = useState(false)
  const [generatingGraph, setGeneratingGraph] = useState(false)
  const [normalizationResult, setNormalizationResult] = useState<NormalizationResult | null>(null)
  const [graphs, setGraphs] = useState<GraphWithInterpretation[]>([])
  const [expandedGraphId, setExpandedGraphId] = useState<string | null>(null)
  const [stats, setStats] = useState<CacheStats | null>(null)
  const [uploadProgress, setUploadProgress] = useState({ baseline: 0, sample: 0 })
  const [generatingReports, setGeneratingReports] = useState(false)
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null)
  const [downloadMenuAnchor, setDownloadMenuAnchor] = useState<null | HTMLElement>(null)
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null)
  const [previewGraph, setPreviewGraph] = useState<GraphWithInterpretation | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)

  // Safe accessor helper to prevent crashes from inconsistent AI response structures
  const safeGet = (obj: any, path: string, defaultValue: any = null) => {
    try {
      return path.split('.').reduce((current, prop) => current?.[prop], obj) ?? defaultValue;
    } catch {
      return defaultValue;
    }
  }

  // Load cache stats
  const loadStats = async () => {
    try {
      const cacheStats = await getCacheStats()
      setStats(cacheStats)
    } catch (error) {
      console.error('Error loading stats:', error)
    }
  }

  // Helper: Download with Save As dialog (File System Access API)
  const downloadWithSaveAs = async (blob: Blob, suggestedFilename: string): Promise<void> => {
    // Try File System Access API first (Chrome, Edge 86+)
    if ('showSaveFilePicker' in window) {
      try {
        const fileHandle = await (window as any).showSaveFilePicker({
          suggestedName: suggestedFilename,
          types: [
            {
              description: suggestedFilename.endsWith('.zip') ? 'ZIP Archive' :
                          suggestedFilename.endsWith('.pdf') ? 'PDF Document' : 'PNG Image',
              accept: suggestedFilename.endsWith('.zip') ? { 'application/zip': ['.zip'] } :
                     suggestedFilename.endsWith('.pdf') ? { 'application/pdf': ['.pdf'] } :
                     { 'image/png': ['.png'] }
            }
          ]
        })
        
        const writable = await fileHandle.createWritable()
        await writable.write(blob)
        await writable.close()
        
        toast.success(`Saved ${suggestedFilename}`)
        return
      } catch (error: any) {
        if (error.name === 'AbortError') {
          // User cancelled
          return
        }
        // Fall through to fallback
        console.warn('Save file picker failed:', error)
      }
    }
    
    // Fallback: regular download (may or may not show Save As depending on browser settings)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = suggestedFilename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    toast.success(`Downloaded ${suggestedFilename}`)
  }

  // Helper: Generate PDF report for a single graph
  const generatePDFReport = async (graph: GraphWithInterpretation): Promise<Blob> => {
    // Fetch color analysis data
    let colorData: any = null
    try {
      const colorResponse = await axios.post(
        `http://localhost:8000/api/analysis/color?file_id=${graph.sampleFileId}`
      )
      if (colorResponse.data.success) {
        colorData = colorResponse.data.data
      }
    } catch (error) {
      console.warn('Could not fetch color data for PDF:', error)
    }

    // Create temporary container
    const container = document.createElement('div')
    container.style.position = 'absolute'
    container.style.left = '-9999px'
    container.style.width = '210mm'
    container.style.padding = '20px'
    container.style.background = 'white'
    container.style.color = 'black'
    container.style.fontFamily = 'Arial, sans-serif'
    
    // Add graph image
    const graphImg = document.createElement('img')
    graphImg.src = graph.data
    graphImg.style.width = '100%'
    graphImg.style.marginBottom = '20px'
    container.appendChild(graphImg)
    
    // Add report header
    const header = document.createElement('h2')
    header.textContent = `AI Interpretation Report - ${graph.sampleName}`
    header.style.marginBottom = '20px'
    header.style.borderBottom = '2px solid #333'
    header.style.paddingBottom = '10px'
    container.appendChild(header)
    
    // Add color analysis section if available
    if (colorData) {
      const colorSection = document.createElement('div')
      colorSection.style.marginBottom = '20px'
      colorSection.style.padding = '15px'
      colorSection.style.border = '1px solid #ddd'
      colorSection.style.borderRadius = '8px'
      colorSection.style.backgroundColor = '#f9f9f9'
      
      colorSection.innerHTML = `
        <h3 style="margin-top: 0; color: #6366F1;">Color Analysis</h3>
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
          <div style="width: 100px; height: 100px; background-color: ${colorData.hex}; border: 2px solid #333; border-radius: 4px; margin-right: 15px;"></div>
          <div>
            <p style="margin: 5px 0;"><strong>Color:</strong> ${colorData.description}</p>
            <p style="margin: 5px 0;"><strong>Hex:</strong> ${colorData.hex}</p>
            <p style="margin: 5px 0;"><strong>RGB:</strong> (${colorData.rgb.r}, ${colorData.rgb.g}, ${colorData.rgb.b})</p>
          </div>
        </div>
        
        <h4 style="margin-bottom: 10px;">Spectral Features</h4>
        <p><strong>Hydrocarbon Content:</strong> ${colorData.analysis.spectral_features.hydrocarbon_content}</p>
        <p><strong>Oxidation Level:</strong> ${colorData.analysis.spectral_features.oxidation_level}</p>
        <p><strong>Sample Concentration:</strong> ${colorData.analysis.spectral_features.sample_concentration}</p>
        
        <h4 style="margin-top: 15px; margin-bottom: 10px;">Technical Parameters</h4>
        <p style="font-size: 12px; line-height: 1.4;">
          C-H Stretch Max: ${colorData.analysis.ch_stretch_max.toFixed(4)} |
          CH₂ Rocking Max: ${colorData.analysis.ch2_rocking_max.toFixed(4)} |
          Carbonyl Max: ${colorData.analysis.carbonyl_max.toFixed(4)} |
          Max Absorbance: ${colorData.analysis.max_absorbance.toFixed(4)}
        </p>
      `
      container.appendChild(colorSection)
    }
    
    // Add simplified report content
    const reportDiv = document.createElement('div')
    reportDiv.innerHTML = `
      <h3>Analysis Summary</h3>
      <p><strong>Spectrum Type:</strong> ${graph.interpretation?.analysis?.spectrum_type || 'N/A'}</p>
      <p><strong>Confidence:</strong> ${graph.interpretation?.analysis?.confidence || 'N/A'}</p>
      
      <h3>Grease Condition Assessment</h3>
      <p><strong>Overall Health:</strong> ${graph.interpretation?.analysis?.grease_condition_assessment?.overall_grease_health || 'N/A'}</p>
      <p><strong>Oxidation Signs:</strong> ${graph.interpretation?.analysis?.grease_condition_assessment?.oxidation_signs || 'N/A'}</p>
      <p><strong>Base Oil Condition:</strong> ${graph.interpretation?.analysis?.grease_condition_assessment?.base_oil_condition || 'N/A'}</p>
      
      <h3>Contamination Detection</h3>
      <p><strong>Water:</strong> ${graph.interpretation?.analysis?.contamination_detection?.water_contamination || 'Not detected'}</p>
      <p><strong>Fuel Dilution:</strong> ${graph.interpretation?.analysis?.contamination_detection?.fuel_dilution || 'Not detected'}</p>
      <p><strong>Particulate:</strong> ${graph.interpretation?.analysis?.contamination_detection?.particulate_contamination || 'Not detected'}</p>
    `
    reportDiv.style.lineHeight = '1.6'
    container.appendChild(reportDiv)
    
    // Add timestamp
    const timestamp = document.createElement('p')
    timestamp.textContent = `Generated: ${new Date().toLocaleString()}`
    timestamp.style.marginTop = '20px'
    timestamp.style.fontSize = '12px'
    timestamp.style.color = '#666'
    container.appendChild(timestamp)
    
    document.body.appendChild(container)

    // Wait for image to load
    await new Promise(resolve => {
      if (graphImg.complete) {
        resolve(null)
      } else {
        graphImg.onload = () => resolve(null)
      }
    })

    // Capture as canvas
    const canvas = await html2canvas(container, {
      scale: 2,
      useCORS: true,
      backgroundColor: '#ffffff'
    })

    // Create PDF
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    })

    const imgWidth = 210
    const imgHeight = (canvas.height * imgWidth) / canvas.width
    const imgData = canvas.toDataURL('image/png')

    let heightLeft = imgHeight
    let position = 0

    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight)
    heightLeft -= 297

    while (heightLeft > 0) {
      position = heightLeft - imgHeight
      pdf.addPage()
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight)
      heightLeft -= 297
    }

    document.body.removeChild(container)
    
    return pdf.output('blob')
  }

  // Batch: Download all graphs
  const handleDownloadAllGraphs = async () => {
    setDownloadMenuAnchor(null)
    
    const graphsWithData = graphs.filter(g => g.data)
    if (graphsWithData.length === 0) {
      toast.error('No graphs to download')
      return
    }

    try {
      toast.loading(`Creating ZIP with ${graphsWithData.length} graphs...`, { id: 'download-graphs' })
      
      // Create ZIP
      const zip = new JSZip()
      
      // Add all graphs to ZIP
      for (const graph of graphsWithData) {
        if (graph.data) {
          // Convert base64 to blob
          const response = await fetch(graph.data)
          const blob = await response.blob()
          
          // Add to ZIP with clean filename
          const filename = `${graph.sampleName.replace(/\.csv$/i, '')}_vs_baseline.png`
          zip.file(filename, blob)
        }
      }
      
      // Generate ZIP file
      const zipBlob = await zip.generateAsync({ type: 'blob' })
      
      // Create filename with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
      const zipFilename = `spectral_graphs_${timestamp}.zip`
      
      // Use Save As dialog
      await downloadWithSaveAs(zipBlob, zipFilename)
      toast.dismiss('download-graphs')
    } catch (error: any) {
      toast.error(`Failed to create ZIP: ${error.message}`, { id: 'download-graphs' })
    }
  }

  // Batch: Download all reports
  const handleDownloadAllReports = async () => {
    setDownloadMenuAnchor(null)
    
    const graphsWithReports = graphs.filter(g => g.interpretation)
    if (graphsWithReports.length === 0) {
      toast('No AI reports available', { icon: '⚠️' })
      return
    }

    try {
      toast.loading(`Creating ZIP with ${graphsWithReports.length} reports...`, { id: 'download-reports' })
      
      // Create ZIP
      const zip = new JSZip()
      
      // Generate and add PDFs to ZIP
      for (let i = 0; i < graphsWithReports.length; i++) {
        const graph = graphsWithReports[i]
        setProgress({ current: i + 1, total: graphsWithReports.length })
        
        const pdfBlob = await generatePDFReport(graph)
        const filename = `${graph.sampleName.replace(/\.csv$/i, '')}_report.pdf`
        zip.file(filename, pdfBlob)
      }
      
      // Generate ZIP file
      const zipBlob = await zip.generateAsync({ type: 'blob' })
      
      // Create filename with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
      const zipFilename = `spectral_reports_${timestamp}.zip`
      
      // Use Save As dialog
      await downloadWithSaveAs(zipBlob, zipFilename)
      toast.dismiss('download-reports')
    } catch (error: any) {
      toast.error(`Failed to create ZIP: ${error.message}`, { id: 'download-reports' })
    } finally {
      setProgress(null)
    }
  }

  // Batch: Generate AI reports for all graphs
  async function handleGenerateAllReports() {
    setGeneratingReports(true)
    
    try {
      const graphsNeedingReports = graphs.filter(g => !g.interpretation)
      
      if (graphsNeedingReports.length === 0) {
        toast.success('All graphs already have AI reports')
        setGeneratingReports(false)
        return
      }
      
      const totalGraphs = graphsNeedingReports.length
      const BATCH_SIZE = 5 // Process 5 at a time
      let completed = 0
      
      // Split into batches of 5
      for (let i = 0; i < graphsNeedingReports.length; i += BATCH_SIZE) {
        const batch = graphsNeedingReports.slice(i, i + BATCH_SIZE)
        
        // Update progress
        setProgress({
          current: Math.min(i + BATCH_SIZE, totalGraphs),
          total: totalGraphs
        })
        
        // Process batch in parallel
        const batchPromises = batch.map(graph =>
          handleAnalyzeGraph(graph.id).catch(error => {
            console.error(`Failed to generate report for ${graph.sampleName}:`, error)
            return null // Don't fail entire batch
          })
        )
        
        // Wait for all in batch to complete
        await Promise.all(batchPromises)
        
        completed += batch.length
        
        // Small delay between batches to avoid rate limiting
        if (i + BATCH_SIZE < graphsNeedingReports.length) {
          await new Promise(resolve => setTimeout(resolve, 1000))
        }
      }
      
      toast.success(`Generated ${completed} AI reports`)
    } catch (error: any) {
      toast.error(`Failed to generate reports: ${error.message}`)
    } finally {
      setGeneratingReports(false)
      setProgress(null)
    }
  }

  // Baseline file dropzone
  const onDropBaseline = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return
    
    const file = acceptedFiles[0]
    if (!file.name.endsWith('.csv')) {
      toast.error('Please upload a CSV file')
      return
    }

    // Clear existing baseline and all dependent state before loading new one
    handleClearBaseline()

    setUploading(true)
    setUploadProgress(prev => ({ ...prev, baseline: 0 }))
    
    try {
      // Simulate progress for UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => ({ ...prev, baseline: Math.min(prev.baseline + 20, 90) }))
      }, 200)

      const uploadedFile = await uploadFile(file)
      
      clearInterval(progressInterval)
      setUploadProgress(prev => ({ ...prev, baseline: 100 }))
      
      setBaselineFile({
        ...uploadedFile,
        type: 'baseline'
      })
      
      toast.success(`Baseline file uploaded: ${uploadedFile.filename}`)
      loadStats()
    } catch (error: any) {
      toast.error(`Upload failed: ${error.response?.data?.detail || error.message}`)
    } finally {
      setUploading(false)
      setTimeout(() => setUploadProgress(prev => ({ ...prev, baseline: 0 })), 1000)
    }
  }, [])

  // Sample file dropzone - now supports multiple files
  const onDropSample = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return
    
    // Clear existing samples and dependent state before loading new set
    setGraphs([])
    setSelectedSampleId(null)
    setPreviewGraph(null)
    setNormalizationResult(null)
    
    setUploading(true)
    
    try {
      const uploadedSamples: UploadedFile[] = []
      
      for (const file of acceptedFiles) {
        if (!file.name.endsWith('.csv')) {
          toast.error(`Skipping ${file.name}: not a CSV file`)
          continue
        }
        
        setUploadProgress(prev => ({ ...prev, sample: 0 }))
        
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => ({ ...prev, sample: Math.min(prev.sample + 20, 90) }))
        }, 200)

        const uploadedFile = await uploadFile(file)
        
        clearInterval(progressInterval)
        setUploadProgress(prev => ({ ...prev, sample: 100 }))
        
        uploadedSamples.push({
          ...uploadedFile,
          type: 'sample'
        })
        
        toast.success(`Sample file uploaded: ${uploadedFile.filename}`)
      }
      
      setSampleFiles(uploadedSamples)
      loadStats()
    } catch (error: any) {
      toast.error(`Upload failed: ${error.response?.data?.detail || error.message}`)
    } finally {
      setUploading(false)
      setTimeout(() => setUploadProgress(prev => ({ ...prev, sample: 0 })), 1000)
    }
  }, [])

  const { getRootProps: getBaselineRootProps, getInputProps: getBaselineInputProps, isDragActive: isBaselineDragActive } = useDropzone({
    onDrop: onDropBaseline,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
    disabled: uploading
  })

  const { getRootProps: getSampleRootProps, getInputProps: getSampleInputProps, isDragActive: isSampleDragActive } = useDropzone({
    onDrop: onDropSample,
    accept: { 'text/csv': ['.csv'] },
    multiple: true,
    disabled: uploading
  })

  // AI Normalization (keeping old behavior for manual normalization without interpretation)
  const handleAnalyze = async () => {
    if (!baselineFile || sampleFiles.length === 0) {
      toast.error('Please upload baseline and at least one sample file first')
      return
    }

    setNormalizing(true)
    setNormalizationResult(null)
    
    try {
      toast.loading('Normalizing spectral data with AI...', { id: 'normalize' })
      
      // Normalize baseline
      const baselineResult = await normalizeCSV(baselineFile.file_id)
      
      // Normalize first sample for display
      const firstSampleResult = await normalizeCSV(sampleFiles[0].file_id)
      
      // Use sample result for display
      if (firstSampleResult.plan) {
        setNormalizationResult({
          file_hash: firstSampleResult.plan.file_hash,
          column_mappings: firstSampleResult.plan.column_mappings,
          confidence_score: firstSampleResult.plan.confidence_score,
          ai_model: firstSampleResult.plan.ai_model,
          transformations_applied: firstSampleResult.plan.data_transformations
        })
        
        toast.success(
          `AI normalization complete! Confidence: ${firstSampleResult.plan.confidence_score.toFixed(1)}%`,
          { id: 'normalize', duration: 4000 }
        )
      }
      
      loadStats()
    } catch (error: any) {
      console.error('Analysis error:', error)
      toast.error(`Normalization failed: ${error.response?.data?.detail || error.message}`, { id: 'normalize' })
    } finally {
      setNormalizing(false)
    }
  }

  // Batch Graph Generation
  const handleGenerateGraph = async () => {
    if (!baselineFile || sampleFiles.length === 0) {
      toast.error('Please upload baseline and sample files first')
      return
    }

    setGeneratingGraph(true)
    setGraphs([])
    
    try {
      toast.loading(`Generating ${sampleFiles.length} comparison graph(s)...`, { id: 'graph' })
      
      const newGraphs: GraphWithInterpretation[] = []
      
      for (const sampleFile of sampleFiles) {
        try {
          const result = await generateGraph(baselineFile.file_id, sampleFile.file_id, 'png')
          newGraphs.push({
            id: result.graph_id,
            sampleName: sampleFile.filename,
            sampleFileId: sampleFile.file_id,
            data: result.graph_data,
            interpretation: null,
            interpretationLoading: false,
            interpretationError: null,
            reportExpanded: false,
            modalOpen: false
          })
        } catch (error: any) {
          toast.error(`Failed to generate graph for ${sampleFile.filename}`)
        }
      }
      
      setGraphs(newGraphs)
      toast.success(`Successfully generated ${newGraphs.length} graph(s)!`, { id: 'graph' })
      loadStats()
    } catch (error: any) {
      toast.error(`Graph generation failed: ${error.response?.data?.detail || error.message}`, { id: 'graph' })
    } finally {
      setGeneratingGraph(false)
    }
  }

  // Per-graph analyze handler
  const handleAnalyzeGraph = async (graphId: string) => {
    const graph = graphs.find(g => g.id === graphId)
    if (!graph) return

    setGraphs(prev => prev.map(g =>
      g.id === graphId
        ? { ...g, interpretationLoading: true, interpretationError: null }
        : g
    ))

    try {
      const interpretation = await interpretSpectralData(graph.sampleFileId)
      
      if (!interpretation || !interpretation.interpretation) {
        throw new Error('Invalid interpretation data')
      }

      setGraphs(prev => prev.map(g =>
        g.id === graphId
          ? {
              ...g,
              interpretation: interpretation.interpretation,
              interpretationLoading: false,
              reportExpanded: true
            }
          : g
      ))

      toast.success('AI analysis complete!')
    } catch (error: any) {
      setGraphs(prev => prev.map(g =>
        g.id === graphId
          ? {
              ...g,
              interpretationLoading: false,
              interpretationError: error.response?.data?.detail || error.message
            }
          : g
      ))
      toast.error('AI analysis failed')
    }
  }

  // Toggle report expansion
  const handleToggleReport = (graphId: string) => {
    setGraphs(prev => prev.map(g =>
      g.id === graphId
        ? { ...g, reportExpanded: !g.reportExpanded }
        : g
    ))
  }

  // Open modal view
  const handleExpandView = (graphId: string) => {
    setExpandedGraphId(graphId)
  }

  // Download single graph
  const handleDownloadGraphById = async (graphId: string) => {
    const graph = graphs.find(g => g.id === graphId)
    if (!graph) return

    try {
      const response = await fetch(graph.data)
      const blob = await response.blob()
      const filename = `${graph.sampleName.replace(/\.csv$/i, '')}_vs_baseline.png`
      await downloadWithSaveAs(blob, filename)
    } catch (error: any) {
      toast.error(`Failed to download: ${error.message}`)
    }
  }

  // Download graph with report as PDF
  const handleDownloadWithReport = async (graphId: string) => {
    const graph = graphs.find(g => g.id === graphId)
    if (!graph || !graph.interpretation) {
      toast.error('No interpretation available')
      return
    }

    try {
      toast.loading('Generating PDF...', { id: 'pdf' })

      const pdfBlob = await generatePDFReport(graph)
      const filename = `${graph.sampleName.replace(/\.csv$/i, '')}_report.pdf`
      
      await downloadWithSaveAs(pdfBlob, filename)
      
      toast.success('PDF ready!', { id: 'pdf' })
    } catch (error) {
      console.error('PDF generation error:', error)
      toast.error('Failed to generate PDF', { id: 'pdf' })
    }
  }

  // Clear files
  const handleClearBaseline = () => {
    setBaselineFile(null)
    setNormalizationResult(null)
    setGraphs([])
  }

  const handleClearSample = (index: number) => {
    setSampleFiles(prev => prev.filter((_, i) => i !== index))
    setGraphs([])
  }
  
  const handleClearAllSamples = () => {
    setSampleFiles([])
    setNormalizationResult(null)
    setGraphs([])
  }

  // Sample selection handler for real-time preview
  const handleSampleSelection = async (graphId: string) => {
    setPreviewLoading(true)
    setSelectedSampleId(graphId)
    
    try {
      const selected = graphs.find(g => g.id === graphId)
      if (selected) {
        // Small delay to show loading (graphs are already generated, so instant)
        await new Promise(resolve => setTimeout(resolve, 100))
        setPreviewGraph(selected)
      }
    } finally {
      setPreviewLoading(false)
    }
  }

  // Auto-select first sample when graphs are loaded
  useEffect(() => {
    if (graphs.length > 0 && !selectedSampleId) {
      // Auto-select first sample
      handleSampleSelection(graphs[0].id)
    }
  }, [graphs])

  // Keyboard navigation for samples
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (!graphs.length) return
      
      const currentIndex = graphs.findIndex(g => g.id === selectedSampleId)
      
      if (e.key === 'ArrowLeft' && currentIndex > 0) {
        handleSampleSelection(graphs[currentIndex - 1].id)
      } else if (e.key === 'ArrowRight' && currentIndex < graphs.length - 1) {
        handleSampleSelection(graphs[currentIndex + 1].id)
      }
    }
    
    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [selectedSampleId, graphs])

  return (
    <Box>
      {/* Hero Section */}
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
            Spectral Analyzer
          </Typography>
          <Typography variant="h6" color="text.secondary">
            AI-Powered Spectroscopy Analysis Platform
          </Typography>
        </Box>
      </motion.div>

      {/* File Upload Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(30, 41, 59, 0.7)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
            height: '100%'
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <CloudUploadIcon color="primary" />
                Baseline File
              </Typography>
              
              {!baselineFile ? (
                <Paper
                  {...getBaselineRootProps()}
                  sx={{
                    p: 4,
                    textAlign: 'center',
                    cursor: 'pointer',
                    border: '2px dashed',
                    borderColor: isBaselineDragActive ? 'primary.main' : 'rgba(148, 163, 184, 0.3)',
                    background: isBaselineDragActive ? 'rgba(99, 102, 241, 0.1)' : 'rgba(30, 41, 59, 0.5)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      borderColor: 'primary.main',
                      background: 'rgba(99, 102, 241, 0.05)',
                    }
                  }}
                >
                  <input {...getBaselineInputProps()} />
                  <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                  <Typography variant="body1" sx={{ mb: 1 }}>
                    {isBaselineDragActive ? 'Drop the file here' : 'Drag & drop baseline CSV file'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    or click to browse
                  </Typography>
                </Paper>
              ) : (
                <Paper sx={{ p: 3, background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="body1" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <CheckCircleIcon color="success" fontSize="small" />
                        {baselineFile.filename}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {(baselineFile.size / 1024).toFixed(2)} KB
                      </Typography>
                    </Box>
                    <IconButton onClick={handleClearBaseline} size="small" color="error">
                      <DeleteIcon />
                    </IconButton>
                  </Stack>
                </Paper>
              )}
              
              {uploadProgress.baseline > 0 && uploadProgress.baseline < 100 && (
                <LinearProgress variant="determinate" value={uploadProgress.baseline} sx={{ mt: 2 }} />
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ 
            background: 'rgba(30, 41, 59, 0.7)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
            height: '100%'
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <CloudUploadIcon color="secondary" />
                Sample Files ({sampleFiles.length})
              </Typography>
              
              <Paper
                {...getSampleRootProps()}
                sx={{
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  border: '2px dashed',
                  borderColor: isSampleDragActive ? 'secondary.main' : 'rgba(148, 163, 184, 0.3)',
                  background: isSampleDragActive ? 'rgba(236, 72, 153, 0.1)' : 'rgba(30, 41, 59, 0.5)',
                  transition: 'all 0.3s ease',
                  mb: 2,
                  '&:hover': {
                    borderColor: 'secondary.main',
                    background: 'rgba(236, 72, 153, 0.05)',
                  }
                }}
              >
                <input {...getSampleInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 48, color: 'secondary.main', mb: 2 }} />
                <Typography variant="body1" sx={{ mb: 1 }}>
                  {isSampleDragActive ? 'Drop files here' : 'Drag & drop sample CSV files (multiple)'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  or click to browse
                </Typography>
              </Paper>
              
              {sampleFiles.length > 0 && (
                <>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      {sampleFiles.length} sample file(s) uploaded
                    </Typography>
                    <Button size="small" color="error" onClick={handleClearAllSamples}>
                      Clear All
                    </Button>
                  </Stack>
                  <Stack spacing={1}>
                    {sampleFiles.map((file, index) => (
                      <Paper key={file.file_id} sx={{ p: 2, background: 'rgba(236, 72, 153, 0.1)', border: '1px solid rgba(236, 72, 153, 0.3)' }}>
                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                          <Box>
                            <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <CheckCircleIcon color="success" fontSize="small" />
                              {file.filename}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {(file.size / 1024).toFixed(2)} KB
                            </Typography>
                          </Box>
                          <IconButton onClick={() => handleClearSample(index)} size="small" color="error">
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Stack>
                      </Paper>
                    ))}
                  </Stack>
                </>
              )}
              
              {uploadProgress.sample > 0 && uploadProgress.sample < 100 && (
                <LinearProgress variant="determinate" value={uploadProgress.sample} sx={{ mt: 2 }} />
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Action Buttons */}
      <Box sx={{ mb: 4, display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
        <Button
          variant="contained"
          size="large"
          startIcon={<ShowChartIcon />}
          onClick={handleGenerateGraph}
          disabled={!baselineFile || sampleFiles.length === 0 || generatingGraph}
          sx={{
            background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            px: 4,
            py: 1.5,
            '&:hover': {
              background: 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)',
            }
          }}
        >
          {generatingGraph ? 'Generating...' : 'Generate Graphs'}
        </Button>

        <Button
          variant="contained"
          size="large"
          startIcon={generatingReports ? <CircularProgress size={20} color="inherit" /> : <AutoAwesomeIcon />}
          onClick={handleGenerateAllReports}
          disabled={graphs.length === 0 || generatingReports}
          sx={{
            background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            px: 4,
            py: 1.5,
            '&:hover': {
              background: 'linear-gradient(135deg, #f5576c 0%, #f093fb 100%)',
            }
          }}
        >
          {generatingReports
            ? `Generating... ${progress ? `${progress.current}/${progress.total}` : ''}`
            : 'Generate AI Reports for All'}
        </Button>
        
        <Button
          variant="contained"
          size="large"
          startIcon={<DownloadIcon />}
          onClick={(e) => setDownloadMenuAnchor(e.currentTarget)}
          disabled={graphs.length === 0}
          sx={{
            background: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
            color: '#333',
            px: 4,
            py: 1.5,
            '&:hover': {
              background: 'linear-gradient(135deg, #fed6e3 0%, #a8edea 100%)',
            }
          }}
        >
          Download All
        </Button>
        <Menu
          anchorEl={downloadMenuAnchor}
          open={Boolean(downloadMenuAnchor)}
          onClose={() => setDownloadMenuAnchor(null)}
        >
          <MenuItem
            onClick={handleDownloadAllGraphs}
            disabled={graphs.length === 0 || !graphs.every(g => g.data)}
          >
            <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
            Download All Graphs
          </MenuItem>
          <MenuItem
            onClick={handleDownloadAllReports}
            disabled={graphs.length === 0 || !graphs.every(g => g.interpretation)}
          >
            <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
            Download All Reports
          </MenuItem>
        </Menu>
      </Box>

      {/* Normalization Results */}
      {normalizationResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card sx={{ 
            mb: 4,
            background: 'rgba(30, 41, 59, 0.7)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <ScienceIcon color="secondary" />
                AI Normalization Results
              </Typography>
              
              <Alert severity="success" sx={{ mb: 2 }}>
                <AlertTitle>Normalization Complete</AlertTitle>
                Model: {normalizationResult.ai_model} | Confidence: {normalizationResult.confidence_score.toFixed(1)}%
              </Alert>

              <Typography variant="subtitle2" sx={{ mb: 2 }}>Column Mappings:</Typography>
              <Grid container spacing={2}>
                {normalizationResult.column_mappings.map((mapping, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Paper sx={{ p: 2, background: 'rgba(99, 102, 241, 0.1)' }}>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                        {mapping.original_name}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        → {mapping.target_name}
                      </Typography>
                      <Chip
                        label={`${mapping.confidence.toFixed(0)}%`}
                        size="small"
                        color="primary"
                        sx={{ mt: 1 }}
                      />
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Live Preview Section */}
      {graphs.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <ShowChartIcon color="info" />
              Live Preview & Color Analysis
            </Typography>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Select Sample</InputLabel>
              <Select
                value={selectedSampleId || ''}
                onChange={(e) => handleSampleSelection(e.target.value)}
                label="Select Sample"
              >
                {graphs.map((graph) => (
                  <MenuItem key={graph.id} value={graph.id}>
                    {graph.sampleName} vs {baselineFile?.filename}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            {previewLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
              </Box>
            )}
            
            {!previewLoading && previewGraph && (
              <Grid container spacing={2}>
                <Grid item xs={12} md={8}>
                  <Paper elevation={3} sx={{
                    p: 2,
                    background: 'rgba(30, 41, 59, 0.7)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(148, 163, 184, 0.1)',
                  }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                      <Box sx={{ width: 20, height: 3, bgcolor: '#2E7D32' }} />
                      <Typography variant="body2">Baseline: {baselineFile?.filename}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                      <Box sx={{ width: 20, height: 3, bgcolor: '#1976D2' }} />
                      <Typography variant="body2">Sample: {previewGraph.sampleName}</Typography>
                    </Box>
                    
                    <img
                      src={previewGraph.data}
                      alt="Live Preview"
                      style={{ width: '100%', height: 'auto' }}
                    />
                    
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      LIVE PREVIEW - Select different samples to update graph in real-time
                      {graphs.length > 1 && ' (Use ← → arrow keys to navigate)'}
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={4}>
                  <GreaseColorPanel
                    fileId={previewGraph.sampleFileId}
                    fileName={previewGraph.sampleName}
                  />
                </Grid>
              </Grid>
            )}
          </Box>
        </motion.div>
      )}

      {/* Graph Gallery with per-graph AI */}
      {graphs.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{ mb: 4 }}>
            <Typography variant="h5" sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
              <ShowChartIcon color="info" />
              Spectral Comparison Graphs ({graphs.length})
            </Typography>
            
            <Grid container spacing={3}>
              {graphs.map((graph) => (
                <Grid item xs={12} md={graphs.length === 1 ? 12 : 6} key={graph.id}>
                  <GraphCard
                    graph={graph}
                    baselineFileId={baselineFile!.file_id}
                    onAnalyze={handleAnalyzeGraph}
                    onDownloadGraph={handleDownloadGraphById}
                    onDownloadWithReport={handleDownloadWithReport}
                    onExpandView={handleExpandView}
                    onToggleReport={handleToggleReport}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>
        </motion.div>
      )}

      {/* Modal for expanded view */}
      <GraphCardModal
        graph={graphs.find(g => g.id === expandedGraphId) || null}
        open={!!expandedGraphId}
        onClose={() => setExpandedGraphId(null)}
        onDownloadGraph={handleDownloadGraphById}
        onDownloadWithReport={handleDownloadWithReport}
      />

      {/* Stats Section */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card sx={{
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(99, 102, 241, 0.05) 100%)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
          }}>
            <CardContent>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#818cf8' }}>
                {(baselineFile ? 1 : 0) + sampleFiles.length}
              </Typography>
              <Typography color="text.secondary">Files Uploaded</Typography>
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
                {graphs.filter(g => g.interpretation).length}
              </Typography>
              <Typography color="text.secondary">AI Interpretations</Typography>
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
                {graphs.length}
              </Typography>
              <Typography color="text.secondary">Graphs Generated</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Dashboard
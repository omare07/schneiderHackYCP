# Spectral Analyzer Web Application

A modern, AI-powered spectral analysis web application built with React and FastAPI. This application replaces the PyQt6 desktop version with a beautiful, responsive web interface while preserving all powerful Python backend functionality.

## 🌟 Features

### Modern React Frontend
- **Beautiful UI**: Glass-morphism effects, gradient backgrounds, smooth animations
- **Material-UI**: Professional component library with dark mode
- **Real-time Updates**: WebSocket integration for progress tracking
- **Drag-and-Drop**: Intuitive file upload with react-dropzone
- **Interactive Graphs**: Plotly.js for high-quality spectral visualization
- **Responsive Design**: Works on desktop, tablet, and mobile

### Powerful FastAPI Backend
- **AI Normalization**: OpenRouter integration for intelligent CSV mapping
- **CSV Parsing**: Handles diverse spectroscopy formats
- **Graph Generation**: Professional matplotlib-based visualizations
- **Caching System**: Multi-tier caching for cost optimization
- **Cost Tracking**: Comprehensive API usage monitoring
- **Batch Processing**: Process multiple samples efficiently

## 📁 Project Structure

```
spectral-analyzer-web/
├── backend/                    # FastAPI Python backend
│   ├── api/
│   │   ├── main.py            # FastAPI app entry point
│   │   └── routes/            # API endpoints
│   │       ├── files.py       # File upload/management
│   │       ├── analysis.py    # CSV parsing & normalization
│   │       ├── graphs.py      # Graph generation
│   │       └── stats.py       # Cache & cost tracking
│   ├── core/                  # Core Python modules
│   │   ├── csv_parser.py      # Advanced CSV parsing
│   │   ├── ai_normalizer.py   # AI-powered normalization
│   │   ├── graph_generator.py # Graph creation
│   │   └── data_validator.py  # Data validation
│   ├── utils/                 # Utility modules
│   │   ├── cache_manager.py   # Multi-tier caching
│   │   ├── cost_tracker.py    # Cost monitoring
│   │   └── api_client.py      # API integrations
│   ├── requirements.txt       # Python dependencies
│   └── run.py                 # Development server
│
└── frontend/                   # React TypeScript frontend
    ├── src/
    │   ├── components/        # Reusable components
    │   ├── pages/             # Page components
    │   ├── services/          # API client
    │   ├── App.tsx            # Main app component
    │   └── main.tsx           # Entry point
    ├── package.json           # Node dependencies
    └── vite.config.ts         # Vite configuration
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** 
- **npm or yarn**

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd spectral-analyzer-web/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (create `.env` file):
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```

5. **Run the backend server**:
   ```bash
   python run.py
   ```

   The API will be available at `http://localhost:8000`
   API docs at `http://localhost:8000/api/docs`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd spectral-analyzer-web/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:5173`

## 📖 Usage Guide

### 1. Upload Files

- Click the upload area or drag CSV files
- Supports single or batch uploads
- Automatic format detection

### 2. Analyze Data

- Select baseline and sample files
- Click "Normalize" for AI-powered analysis
- Review confidence scores and mappings

### 3. Generate Graphs

- Choose baseline and samples for comparison
- Generate publication-quality graphs
- Export as PNG, PDF, or SVG

### 4. Monitor Performance

- View cache statistics
- Track API costs
- Monitor processing times

## 🎨 Design Highlights

### Modern UI Features

- **Gradient Backgrounds**: Beautiful color transitions
- **Glass-morphism**: Frosted glass card effects
- **Smooth Animations**: Framer Motion for fluid transitions
- **Toast Notifications**: Real-time feedback
- **Loading States**: Skeleton screens and spinners
- **Dark Mode**: Easy on the eyes

### Professional Styling

- **Color Scheme**: Indigo (#6366f1) and Pink (#ec4899)
- **Typography**: Inter font family
- **Spacing**: Consistent 8px grid system
- **Responsive**: Mobile-first design

## 🔌 API Endpoints

### File Management
- `POST /api/files/upload` - Upload CSV file
- `POST /api/files/upload-batch` - Upload multiple files
- `GET /api/files/list` - List uploaded files
- `DELETE /api/files/{file_id}` - Delete file
- `GET /api/files/{file_id}/info` - Get file info

### Analysis
- `POST /api/analysis/parse` - Parse CSV structure
- `POST /api/analysis/normalize` - AI normalization
- `POST /api/analysis/validate` - Validate data quality
- `GET /api/analysis/ai-status` - Check AI service

### Graphs
- `POST /api/graphs/generate` - Generate comparison graph
- `POST /api/graphs/generate-batch` - Batch generation
- `GET /api/graphs/{graph_id}` - Get graph file
- `GET /api/graphs/{graph_id}/base64` - Get as base64

### Statistics
- `GET /api/stats/cache` - Cache performance
- `GET /api/stats/costs` - Cost tracking
- `POST /api/stats/cache/clear` - Clear cache
- `DELETE /api/stats/cache/expired` - Cleanup

### WebSocket
- `WS /api/ws/progress/{session_id}` - Real-time progress

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 🚀 Production Deployment

### Backend (Docker)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend (Build)
```bash
cd frontend
npm run build
# Serve dist/ with nginx or any static server
```

## 🆚 Improvements Over PyQt6 Version

### User Experience
✅ **Web-based**: Access from any device, no installation
✅ **Modern UI**: Beautiful gradients and animations
✅ **Responsive**: Works on mobile and tablet
✅ **Real-time Updates**: WebSocket progress tracking
✅ **Toast Notifications**: Better user feedback

### Performance
✅ **Faster**: Async FastAPI backend
✅ **Caching**: Multi-tier cache system
✅ **Batch Processing**: More efficient workflows
✅ **Better Error Handling**: Comprehensive validation

### Developer Experience
✅ **TypeScript**: Type safety in frontend
✅ **API Documentation**: Auto-generated Swagger docs
✅ **Modular**: Better code organization
✅ **Modern Stack**: Latest technologies

## 📊 Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pandas** - Data manipulation
- **Matplotlib** - Graph generation
- **OpenRouter** - AI API integration
- **Redis** - Caching (optional)

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Material-UI** - Component library
- **React Query** - Data fetching
- **Plotly.js** - Interactive graphs
- **Axios** - HTTP client
- **Framer Motion** - Animations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project preserves all functionality from the original spectral analyzer while providing a modern web interface.

## 🙏 Acknowledgments

- Original spectral analyzer Python backend
- OpenRouter for AI capabilities
- Material-UI for beautiful components
- The open-source community

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Made with ❤️ for spectroscopy analysis**
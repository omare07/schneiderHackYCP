# SpectroView

Advanced FTIR Spectroscopy Analysis Platform for Grease Condition Monitoring

SpectroView is a comprehensive dual-platform solution designed for MRG Labs grease condition monitoring using FTIR (Fourier Transform Infrared) spectroscopy. The platform provides AI-powered data normalization, professional visualization, and advanced analysis capabilities for spectroscopic data interpretation.

## Overview

SpectroView addresses the critical need for accurate and efficient analysis of FTIR spectral data in industrial lubrication monitoring. The platform combines sophisticated data processing algorithms with intuitive interfaces to transform raw spectroscopic measurements into actionable insights for maintenance and quality control applications.

### Purpose

- **Primary Application**: MRG Labs grease condition monitoring using FTIR spectroscopy
- **Target Use Case**: Real-time analysis and interpretation of spectral data for predictive maintenance
- **Key Innovation**: AI-powered normalization with confidence scoring and custom spectroscopic color analysis

### Key Innovation Points

- Dual-mode deployment supporting both desktop and web-based workflows
- Intelligent CSV parsing with support for multiple formats and encoding standards
- AI-driven data normalization with confidence metrics
- Custom grease color analysis algorithm based on spectroscopic principles
- Multi-tier caching system for optimized performance and cost reduction
- Comprehensive cost tracking and API usage monitoring

## Features

### Core Capabilities

- **Dual-Platform Architecture**: Desktop application (PyQt6) and web application (React/FastAPI)
- **AI-Powered Data Normalization**: Leverages OpenRouter API for intelligent spectral data processing
- **Professional Graph Generation**: High-quality comparison visualizations with customizable parameters
- **Custom Color Analysis**: Spectroscopic-based grease color determination algorithm
- **Multi-Tier Caching**: Intelligent caching system reducing API costs and improving response times
- **Cost Tracking and Monitoring**: Comprehensive API usage and cost metrics
- **Batch Processing**: Efficient processing of multiple spectral files
- **Real-Time Analysis**: Immediate interpretation and visual feedback

### Data Processing

- Support for multiple CSV delimiters (comma, semicolon, tab)
- Automatic detection of European and US number formats
- UTF-8 and Latin-1 encoding detection
- Intelligent header recognition and metadata extraction
- Robust error handling and validation

## Architecture

SpectroView employs a modular architecture designed for flexibility and scalability:

### Desktop Application Stack

- **Language**: Python 3.8+
- **GUI Framework**: PyQt6 for modern, responsive interface
- **Visualization**: matplotlib for professional graph generation
- **Data Processing**: pandas, numpy for spectral data manipulation
- **AI Integration**: OpenRouter API client for normalization

### Web Application Stack

**Backend**:
- **Framework**: FastAPI for high-performance async API
- **Runtime**: Python 3.8+ with uvicorn ASGI server
- **Data Processing**: Shared core modules with desktop application
- **API**: RESTful endpoints for file upload, analysis, and visualization

**Frontend**:
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for optimized development and production builds
- **Styling**: Modern CSS with responsive design
- **State Management**: React hooks and context

### Core Components

The system is built around several key modules:

- **CSV Parser**: Intelligent spectral data extraction and normalization
- **AI Normalizer**: OpenRouter integration with confidence scoring
- **Graph Generator**: Professional visualization engine
- **Color Analyzer**: Spectroscopic color determination
- **Cache Manager**: Multi-tier caching for performance optimization
- **Cost Tracker**: API usage monitoring and cost analysis

## Installation and Setup

### Desktop Application

```bash
cd spectral_analyzer
pip install -r requirements.txt
python main.py
```

### Web Application

**Backend Setup**:

```bash
cd spectral-analyzer-web/backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env and configure your OPENROUTER_API_KEY
python3 run.py
```

**Frontend Setup**:

```bash
cd spectral-analyzer-web/frontend
npm install
npm run dev
```

The web application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Configuration

### Environment Variables

The following environment variables are required:

**OPENROUTER_API_KEY**: Your OpenRouter API key for AI normalization features

Configuration files are located in:
- Desktop: `spectral_analyzer/.env`
- Web Backend: `spectral-analyzer-web/backend/.env`

Reference the `.env.example` files in each directory for configuration templates.

### API Configuration

The platform uses OpenRouter API with the following default settings:
- Model: `deepseek/deepseek-chat` (configurable)
- Temperature: 0.1 (for consistent results)
- Max Tokens: 1000

## Usage

### Uploading Spectral Data

1. **Desktop Application**: Drag and drop CSV files into the application window or use the file selection dialog
2. **Web Application**: Use the upload interface to select spectral data files

### Generating Comparison Graphs

The platform automatically processes uploaded files and generates comparison visualizations showing:
- Baseline spectral data
- Sample spectral data
- Overlay comparison with difference highlighting
- Statistical analysis metrics

### Interpreting Results

Each analysis provides:
- AI-generated interpretation with confidence scores
- Spectral comparison graphs
- Grease color analysis (when applicable)
- Data quality metrics
- Processing cost information

### Color Analysis Feature

The custom color analysis algorithm determines grease color based on spectroscopic absorption patterns:
- Analyzes visible spectrum range (400-700 nm)
- Calculates transmission percentages
- Determines effective color using spectroscopic principles
- Provides confidence metrics

## Technical Details

### CSV Format Support

The parser handles diverse spectral data formats:
- **Delimiters**: Comma, semicolon, tab
- **Number Formats**: European (comma decimal) and US (period decimal)
- **Encoding**: UTF-8, Latin-1 with automatic detection
- **Headers**: Intelligent detection and metadata extraction
- **Data Validation**: Robust error checking and normalization

### AI Normalization

The AI normalization system provides:
- Intelligent column mapping and header interpretation
- Data format standardization
- Confidence scoring for each normalization decision
- Caching of normalized results for performance

### Graph Generation

Professional visualizations with:
- High-resolution output (300 DPI)
- Customizable dimensions and styling
- Overlay comparison capabilities
- Statistical annotations
- Export to PNG format

### Color Analysis Algorithm

Spectroscopic basis for color determination:
- Integration of transmission across visible spectrum
- Weighted color contribution calculation
- Dominant wavelength identification
- Color space conversion and representation

## Core Components

### CSV Parser (`core/csv_parser.py`)

Robust parser for spectral data files with support for multiple formats, encodings, and delimiter types. Includes intelligent header detection and metadata extraction.

### AI Normalizer (`core/ai_normalizer.py`)

OpenRouter API integration for intelligent data normalization. Provides confidence scoring and caching for optimized performance.

### Graph Generator (`core/graph_generator.py`)

Professional visualization engine for spectral comparison graphs. Supports customizable styling and high-resolution output.

### Color Analyzer (`core/color_analyzer.py`)

Custom spectroscopic color analysis algorithm for grease condition assessment based on visible spectrum absorption patterns.

### Cache Manager (`utils/cache_manager.py`)

Multi-tier caching system with file-based and memory-based caching strategies. Reduces API calls and improves response times.

### Cost Tracker (`utils/cost_tracker.py`)

Comprehensive tracking of API usage and associated costs. Provides detailed analytics and usage reports.

## Testing

### Running Tests

```bash
cd spectral_analyzer
pytest tests/
```

### Test Coverage

The project includes comprehensive test suites with:
- Integration test coverage: 81.8%
- Unit tests for core components
- End-to-end workflow testing
- Performance benchmarks

### Test Data

The `tests/test_data/` directory contains diverse spectral data samples for validation:
- Perfect baseline data
- European format samples
- Raman spectroscopy data
- Edge cases and problematic formats

## Project Structure

```
schneiderHackYCP/
├── spectral_analyzer/              # Desktop Application
│   ├── core/                       # Core processing modules
│   ├── ui/                         # PyQt6 user interface
│   ├── utils/                      # Utility modules
│   ├── config/                     # Configuration
│   ├── tests/                      # Test suites
│   └── resources/                  # Icons, styles, templates
│
├── spectral-analyzer-web/          # Web Application
│   ├── backend/                    # FastAPI backend
│   │   ├── api/                    # API routes
│   │   ├── core/                   # Shared core modules
│   │   ├── utils/                  # Utility modules
│   │   └── config/                 # Configuration
│   └── frontend/                   # React frontend
│       └── src/                    # Source code
│           ├── components/         # React components
│           ├── pages/              # Application pages
│           └── services/           # API services
│
└── Documentation/                  # Project documentation
```

## Contributing

Contributions to SpectroView are welcome. When contributing, please:

1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Ensure all tests pass
5. Submit a pull request with detailed description

### Code Standards

- Follow PEP 8 style guidelines for Python code
- Use TypeScript strict mode for frontend code
- Maintain test coverage above 80%
- Document all public APIs and functions

## License

This project is developed as part of academic and research initiatives. License information will be determined based on project deployment and distribution requirements.

## Acknowledgments

This project was developed for the **Schneider Prize for Technology Innovation 2025**, addressing real-world challenges in industrial lubrication monitoring.

Special thanks to **MRG Labs** for providing domain expertise and use case requirements that guided the development of this platform.

## Contact and Support

For questions, issues, or contributions, please use the project's issue tracker or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Platform**: Cross-platform (Windows, macOS, Linux)
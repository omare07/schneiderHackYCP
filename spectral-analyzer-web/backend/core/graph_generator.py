"""
Professional spectral graph generation system for laboratory-quality reports.

Creates high-quality spectral graphs with baseline + sample overlay visualization,
batch processing capabilities, and comprehensive styling for scientific publications.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
try:
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        # Fallback for headless environments
        FigureCanvas = None
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from scipy import interpolate
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class ExportFormat(Enum):
    """Supported export formats."""
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    PDF = "pdf"
    SVG = "svg"


class ThemeMode(Enum):
    """Graph theme modes."""
    LIGHT = "light"
    DARK = "dark"
    PUBLICATION = "publication"


@dataclass
class GraphConfig:
    """Professional graph configuration for spectral analysis."""
    figure_size: Tuple[float, float] = (10, 6)
    dpi: int = 300
    baseline_color: str = '#2E7D32'  # Professional green
    sample_color: str = '#1976D2'    # Professional blue
    background_color: str = 'white'
    text_color: str = 'black'
    grid_color: str = '#E0E0E0'
    grid_alpha: float = 0.3
    show_grid: bool = True
    font_family: str = 'Arial'
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    line_width: float = 1.5
    legend_size: int = 10
    margin_inches: float = 0.1
    theme: ThemeMode = ThemeMode.LIGHT


@dataclass
class BatchResult:
    """Result of batch graph generation."""
    total_graphs: int
    successful: int
    failed: int
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    average_time_per_graph: float = 0.0


class SpectralGraphGenerator:
    """
    Professional spectral graph generation system.
    
    Features:
    - Baseline + sample overlay visualization
    - Batch processing with progress reporting
    - Professional styling for laboratory reports
    - High-quality export (PNG/JPG/PDF)
    - Automatic filename conflict resolution
    - Memory-efficient processing
    - Theme support (light/dark/publication)
    """
    
    def __init__(self, theme_manager=None):
        """
        Initialize the spectral graph generator.
        
        Args:
            theme_manager: Optional theme manager for UI integration
        """
        self.logger = logging.getLogger(__name__)
        self.theme_manager = theme_manager
        
        # Set matplotlib style for professional output
        plt.style.use('default')  # Start with clean default
        
        # Configure matplotlib for high-quality output
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'font.size': 12,
            'axes.linewidth': 1.0,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': 100,  # Display DPI
            'savefig.dpi': 300,  # Save DPI
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        self.logger.info("SpectralGraphGenerator initialized")
    
    def generate_spectral_graph(self, data: pd.DataFrame, output_path: Union[str, Path],
                               title: str = "Spectral Analysis",
                               config: Optional[GraphConfig] = None) -> bool:
        """
        Generate and save a single spectral graph.
        
        Args:
            data: DataFrame with spectral data
            output_path: Path to save the graph
            title: Graph title
            config: Optional graph configuration
            
        Returns:
            True if graph generated and saved successfully
        """
        try:
            if config is None:
                config = GraphConfig()
            
            self.logger.info(f"Generating spectral graph: {title}")
            
            # Create figure with professional styling
            fig = Figure(figsize=config.figure_size, dpi=config.dpi)
            fig.patch.set_facecolor(config.background_color)
            
            ax = fig.add_subplot(111)
            ax.set_facecolor(config.background_color)
            
            # Extract spectral data
            x_data, y_data = self._extract_spectral_data(data)
            
            if x_data is None or y_data is None:
                self.logger.error("Failed to extract spectral data")
                return False
            
            # Plot data
            ax.plot(x_data, y_data,
                   color=config.baseline_color,
                   linewidth=config.line_width,
                   label='Spectral Data',
                   alpha=0.9)
            
            # Apply professional styling
            self._apply_professional_styling(ax, config)
            
            # Set title
            ax.set_title(title, fontsize=config.title_size, fontweight='bold',
                        color=config.text_color, pad=20)
            
            # Set axis labels
            ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=config.label_size,
                         color=config.text_color)
            ax.set_ylabel('Absorbance', fontsize=config.label_size,
                         color=config.text_color)
            
            # Invert x-axis for IR spectroscopy convention (4000 → 400)
            ax.invert_xaxis()
            
            # Grid
            if config.show_grid:
                ax.grid(True, alpha=config.grid_alpha, color=config.grid_color)
            
            # Tight layout
            fig.tight_layout(pad=config.margin_inches)
            
            # Save the graph
            success = self.save_graph(fig, output_path, 'png', config.dpi)
            
            # Clean up
            plt.close(fig)
            
            if success:
                self.logger.info(f"Spectral graph saved successfully: {output_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to generate spectral graph: {e}")
            return False
    
    def generate_comparison_graph(self, baseline_data: pd.DataFrame,
                                sample_data: pd.DataFrame,
                                baseline_name: str, sample_name: str,
                                config: Optional[GraphConfig] = None) -> Figure:
        """
        Generate single comparison graph with baseline and sample overlay.
        
        Args:
            baseline_data: DataFrame with baseline spectral data
            sample_data: DataFrame with sample spectral data
            baseline_name: Name/filename of baseline
            sample_name: Name/filename of sample
            config: Optional graph configuration
            
        Returns:
            Matplotlib Figure object
        """
        if config is None:
            config = GraphConfig()
        
        try:
            self.logger.info(f"Generating comparison graph: {sample_name} vs {baseline_name}")
            
            # Create figure with professional styling
            fig = Figure(figsize=config.figure_size, dpi=config.dpi)
            fig.patch.set_facecolor(config.background_color)
            
            ax = fig.add_subplot(111)
            ax.set_facecolor(config.background_color)
            
            # Process and align data
            baseline_x, baseline_y = self._extract_spectral_data(baseline_data)
            sample_x, sample_y = self._extract_spectral_data(sample_data)
            
            if baseline_x is None or sample_x is None:
                return self._create_error_figure("Invalid spectral data format", config)
            
            # Interpolate data to common wavenumber range if needed
            common_x, aligned_baseline_y, aligned_sample_y = self._align_spectral_data(
                baseline_x, baseline_y, sample_x, sample_y
            )
            
            # Plot baseline (green)
            ax.plot(common_x, aligned_baseline_y,
                   color=config.baseline_color,
                   linewidth=config.line_width,
                   label=f'Baseline: {baseline_name}',
                   alpha=0.9)
            
            # Plot sample (blue)
            ax.plot(common_x, aligned_sample_y,
                   color=config.sample_color,
                   linewidth=config.line_width,
                   label=f'Sample: {sample_name}',
                   alpha=0.9)
            
            # Apply professional styling
            self._apply_professional_styling(ax, config)
            
            # Set title
            title = f"Spectral Comparison: {sample_name} vs {baseline_name}"
            ax.set_title(title, fontsize=config.title_size, fontweight='bold',
                        color=config.text_color, pad=20)
            
            # Set axis labels
            ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=config.label_size,
                         color=config.text_color)
            ax.set_ylabel('Absorbance', fontsize=config.label_size,
                         color=config.text_color)
            
            # Invert x-axis for IR spectroscopy convention (4000 → 400)
            ax.invert_xaxis()
            
            # Add legend
            legend = ax.legend(fontsize=config.legend_size,
                             frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            
            # Grid
            if config.show_grid:
                ax.grid(True, alpha=config.grid_alpha, color=config.grid_color)
            
            # Tight layout
            fig.tight_layout(pad=config.margin_inches)
            
            self.logger.info("Comparison graph generated successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison graph: {e}")
            return self._create_error_figure(f"Graph generation failed: {e}", config)
    
    def generate_batch_graphs(self, baseline_data: pd.DataFrame,
                            sample_datasets: List[Tuple[pd.DataFrame, str]],
                            output_dir: str, format: str = 'png',
                            baseline_name: str = "baseline",
                            progress_callback: Optional[Callable[[str, float], None]] = None) -> BatchResult:
        """
        Generate multiple graphs for batch processing.
        
        Args:
            baseline_data: DataFrame with baseline spectral data
            sample_datasets: List of (DataFrame, filename) tuples for samples
            output_dir: Directory to save graphs
            format: Export format (png, jpg, pdf)
            baseline_name: Name of baseline file
            progress_callback: Optional callback for progress updates (message, percentage)
            
        Returns:
            BatchResult with processing statistics
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        result = BatchResult(
            total_graphs=len(sample_datasets),
            successful=0,
            failed=0
        )
        
        try:
            self.logger.info(f"Starting batch generation of {len(sample_datasets)} graphs")
            
            if progress_callback:
                progress_callback("Starting batch processing...", 0.0)
            
            for i, (sample_data, sample_filename) in enumerate(sample_datasets):
                try:
                    # Update progress
                    progress = (i / len(sample_datasets)) * 100
                    if progress_callback:
                        progress_callback(f"Processing {sample_filename}...", progress)
                    
                    # Generate graph
                    fig = self.generate_comparison_graph(
                        baseline_data, sample_data, baseline_name, sample_filename
                    )
                    
                    # Create output filename with conflict resolution
                    output_filename = self._create_safe_filename(
                        sample_filename, baseline_name, format, output_path
                    )
                    output_file_path = output_path / output_filename
                    
                    # Save graph
                    success = self.save_graph(fig, output_file_path, format)
                    
                    if success:
                        result.successful += 1
                        result.output_files.append(str(output_file_path))
                        self.logger.debug(f"Saved graph: {output_file_path}")
                    else:
                        result.failed += 1
                        result.errors.append(f"Failed to save graph for {sample_filename}")
                    
                    # Clean up figure to free memory
                    plt.close(fig)
                    
                except Exception as e:
                    result.failed += 1
                    error_msg = f"Failed to process {sample_filename}: {e}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Final progress update
            if progress_callback:
                progress_callback("Batch processing complete", 100.0)
            
            # Calculate timing statistics
            result.processing_time = time.time() - start_time
            if result.total_graphs > 0:
                result.average_time_per_graph = result.processing_time / result.total_graphs
            
            self.logger.info(
                f"Batch processing complete: {result.successful}/{result.total_graphs} successful "
                f"in {result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            result.errors.append(f"Batch processing error: {e}")
            result.processing_time = time.time() - start_time
            return result
    
    def save_graph(self, figure: Figure, filepath: Union[str, Path],
                  format: str = 'png', dpi: int = 300) -> bool:
        """
        Save graph with high quality settings.
        
        Args:
            figure: Matplotlib Figure object
            filepath: Output file path
            format: Export format
            dpi: Resolution for raster formats
            
        Returns:
            True if save successful
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format
            if format.lower() in ['jpg', 'jpeg']:
                # For JPEG, set background to white (no transparency)
                figure.patch.set_facecolor('white')
                save_format = 'jpeg'
            else:
                save_format = format.lower()
            
            # Save with high quality settings
            figure.savefig(
                filepath,
                format=save_format,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.1,
                facecolor='white' if save_format in ['jpeg', 'jpg'] else None,
                edgecolor='none',
                transparent=False if save_format in ['jpeg', 'jpg'] else None
            )
            
            self.logger.debug(f"Graph saved: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save graph to {filepath}: {e}")
            return False
    
    def _extract_spectral_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract wavenumber and intensity data from DataFrame."""
        try:
            # Look for wavenumber column
            wavenumber_col = None
            intensity_col = None
            
            for col in data.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['wavenumber', 'wave', 'cm-1', 'frequency']):
                    wavenumber_col = col
                elif any(keyword in col_lower for keyword in ['absorbance', 'abs', 'intensity', 'transmittance']):
                    intensity_col = col
            
            # If not found by name, assume first two columns
            if wavenumber_col is None or intensity_col is None:
                if len(data.columns) >= 2:
                    wavenumber_col = data.columns[0]
                    intensity_col = data.columns[1]
                else:
                    return None, None
            
            # Extract data and convert to numeric
            x_data = pd.to_numeric(data[wavenumber_col], errors='coerce').dropna()
            y_data = pd.to_numeric(data[intensity_col], errors='coerce').dropna()
            
            # Ensure same length
            min_length = min(len(x_data), len(y_data))
            x_data = x_data.iloc[:min_length]
            y_data = y_data.iloc[:min_length]
            
            return x_data.values, y_data.values
            
        except Exception as e:
            self.logger.error(f"Failed to extract spectral data: {e}")
            return None, None
    
    def _align_spectral_data(self, baseline_x: np.ndarray, baseline_y: np.ndarray,
                           sample_x: np.ndarray, sample_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align spectral data to common wavenumber range using interpolation."""
        try:
            # Find overlapping range
            x_min = max(baseline_x.min(), sample_x.min())
            x_max = min(baseline_x.max(), sample_x.max())
            
            # Create common x-axis (descending for IR spectroscopy)
            num_points = min(len(baseline_x), len(sample_x), 2000)  # Limit for performance
            common_x = np.linspace(x_max, x_min, num_points)
            
            # Interpolate baseline data
            baseline_interp = interpolate.interp1d(
                baseline_x, baseline_y, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )
            aligned_baseline_y = baseline_interp(common_x)
            
            # Interpolate sample data
            sample_interp = interpolate.interp1d(
                sample_x, sample_y, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )
            aligned_sample_y = sample_interp(common_x)
            
            return common_x, aligned_baseline_y, aligned_sample_y
            
        except Exception as e:
            self.logger.warning(f"Data alignment failed, using original data: {e}")
            # Fallback: use baseline x-axis
            return baseline_x, baseline_y, np.interp(baseline_x, sample_x, sample_y)
    
    def _apply_professional_styling(self, ax, config: GraphConfig):
        """Apply professional styling to the plot."""
        # Set colors
        ax.tick_params(colors=config.text_color, labelsize=config.tick_size)
        ax.spines['bottom'].set_color(config.text_color)
        ax.spines['left'].set_color(config.text_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set font properties
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontfamily(config.font_family)
    
    def _create_safe_filename(self, sample_name: str, baseline_name: str,
                            format: str, output_dir: Path) -> str:
        """Create safe filename with conflict resolution."""
        # Clean filenames
        sample_clean = self._sanitize_filename(sample_name)
        baseline_clean = self._sanitize_filename(baseline_name)
        
        # Create base filename
        base_filename = f"{sample_clean}_vs_{baseline_clean}.{format.lower()}"
        
        # Check for conflicts and add numeric suffix if needed
        counter = 1
        filename = base_filename
        while (output_dir / filename).exists():
            name_part = f"{sample_clean}_vs_{baseline_clean}_{counter:03d}"
            filename = f"{name_part}.{format.lower()}"
            counter += 1
            
            # Prevent infinite loop
            if counter > 999:
                filename = f"{sample_clean}_vs_{baseline_clean}_{int(time.time())}.{format.lower()}"
                break
        
        return filename
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Remove file extension if present
        if '.' in filename:
            filename = Path(filename).stem
        
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        return filename[:50]
    
    def _create_error_figure(self, error_message: str, config: GraphConfig) -> Figure:
        """Create error figure when generation fails."""
        fig = Figure(figsize=config.figure_size, dpi=config.dpi)
        fig.patch.set_facecolor(config.background_color)
        
        ax = fig.add_subplot(111)
        ax.set_facecolor(config.background_color)
        
        ax.text(0.5, 0.5, f'❌ Graph Generation Error\n\n{error_message}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='red', fontfamily=config.font_family,
                bbox=dict(boxstyle='round,pad=1', facecolor='white',
                         alpha=0.8, edgecolor='red'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Graph Generation Failed", fontsize=config.title_size,
                    color='red', fontfamily=config.font_family)
        
        fig.tight_layout()
        return fig


# Legacy compatibility classes
class PlotType(Enum):
    """Available plot types for spectral data."""
    LINE = "line"
    SCATTER = "scatter"
    FILLED = "filled"
    STACKED = "stacked"
    OVERLAY = "overlay"
    DIFFERENCE = "difference"
    WATERFALL = "waterfall"


class ColorScheme(Enum):
    """Available color schemes."""
    DEFAULT = "default"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    SCIENTIFIC = "scientific"
    MONOCHROME = "monochrome"
    CUSTOM = "custom"


@dataclass
class PlotStyle:
    """Plot styling configuration."""
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    line_width: float = 1.5
    marker_size: float = 3.0
    alpha: float = 1.0
    grid: bool = True
    legend: bool = True
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    dpi: int = 300
    figure_size: Tuple[float, float] = (10, 6)


@dataclass
class AxisConfig:
    """Axis configuration."""
    x_label: str = "Wavenumber (cm⁻¹)"
    y_label: str = "Absorbance"
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    x_scale: str = "linear"
    y_scale: str = "linear"
    invert_x: bool = False
    invert_y: bool = False


class GraphGenerator:
    """
    Professional spectral graph generation system.
    
    Features:
    - Multiple plot types and styles
    - Interactive Qt integration
    - High-quality export options
    - Batch processing capabilities
    - Customizable themes and layouts
    """
    
    def __init__(self):
        """Initialize the graph generator."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize the underlying spectral graph generator
        self.spectral_generator = SpectralGraphGenerator()
        
        # Set matplotlib backend and style
        plt.style.use('seaborn-v0_8')
        
        # Color palettes
        self.color_palettes = {
            ColorScheme.DEFAULT: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            ColorScheme.VIRIDIS: plt.cm.viridis(np.linspace(0, 1, 10)),
            ColorScheme.PLASMA: plt.cm.plasma(np.linspace(0, 1, 10)),
            ColorScheme.SCIENTIFIC: ['#0173b2', '#de8f05', '#029e73', '#cc78bc', '#ca9161'],
            ColorScheme.MONOCHROME: ['#000000', '#404040', '#808080', '#c0c0c0', '#e0e0e0']
        }
        
        # Default configurations
        self.default_config = GraphConfig()
        
        # Peak detection parameters
        self.peak_detection_params = {
            'height': 0.01,
            'distance': 10,
            'prominence': 0.005,
            'width': 1
        }
    
    def generate_spectral_graph(self, data: pd.DataFrame, output_path: Union[str, Path],
                               title: str = "Spectral Analysis",
                               config: Optional[GraphConfig] = None) -> bool:
        """
        Generate and save a spectral graph - delegates to SpectralGraphGenerator.
        
        Args:
            data: DataFrame with spectral data
            output_path: Path to save the graph
            title: Graph title
            config: Optional graph configuration
            
        Returns:
            True if graph generated and saved successfully
        """
        return self.spectral_generator.generate_spectral_graph(data, output_path, title, config)
    
    def _extract_spectral_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract spectral data - delegates to SpectralGraphGenerator."""
        return self.spectral_generator._extract_spectral_data(data)
    
    def _apply_professional_styling(self, ax, config: GraphConfig):
        """Apply professional styling - delegates to SpectralGraphGenerator."""
        return self.spectral_generator._apply_professional_styling(ax, config)
    
    def save_graph(self, figure: Figure, filepath: Union[str, Path],
                  format: str = 'png', dpi: int = 300) -> bool:
        """Save graph - delegates to SpectralGraphGenerator."""
        return self.spectral_generator.save_graph(figure, filepath, format, dpi)
    
    def create_spectral_plot(self, data: pd.DataFrame, config: Optional[GraphConfig] = None) -> Figure:
        """
        Create a spectral plot from data.
        
        Args:
            data: DataFrame with spectral data
            config: Graph configuration options
            
        Returns:
            Matplotlib Figure object
        """
        if config is None:
            config = self.default_config
        
        try:
            self.logger.info(f"Creating {config.plot_type.value} spectral plot")
            
            # Create figure and axis
            fig = Figure(figsize=config.style.figure_size, dpi=config.style.dpi)
            ax = fig.add_subplot(111)
            
            # Apply preprocessing if needed
            processed_data = self._preprocess_data(data, config)
            
            # Generate plot based on type
            if config.plot_type == PlotType.LINE:
                self._create_line_plot(ax, processed_data, config)
            elif config.plot_type == PlotType.SCATTER:
                self._create_scatter_plot(ax, processed_data, config)
            elif config.plot_type == PlotType.FILLED:
                self._create_filled_plot(ax, processed_data, config)
            elif config.plot_type == PlotType.STACKED:
                self._create_stacked_plot(ax, processed_data, config)
            elif config.plot_type == PlotType.OVERLAY:
                self._create_overlay_plot(ax, processed_data, config)
            elif config.plot_type == PlotType.WATERFALL:
                self._create_waterfall_plot(ax, processed_data, config)
            else:
                self._create_line_plot(ax, processed_data, config)
            
            # Apply styling
            self._apply_styling(fig, ax, config)
            
            # Add annotations
            self._add_annotations(ax, processed_data, config)
            
            self.logger.info("Spectral plot created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create spectral plot: {e}")
            return self._create_error_plot(str(e))
    
    def create_comparison_plot(self, datasets: List[Tuple[pd.DataFrame, str]], 
                             config: Optional[GraphConfig] = None) -> Figure:
        """
        Create comparison plot with multiple datasets.
        
        Args:
            datasets: List of (DataFrame, label) tuples
            config: Graph configuration options
            
        Returns:
            Matplotlib Figure object
        """
        if config is None:
            config = self.default_config
        
        try:
            self.logger.info(f"Creating comparison plot with {len(datasets)} datasets")
            
            fig = Figure(figsize=config.style.figure_size, dpi=config.style.dpi)
            ax = fig.add_subplot(111)
            
            colors = self._get_color_palette(config.style.color_scheme, len(datasets))
            
            for i, (data, label) in enumerate(datasets):
                processed_data = self._preprocess_data(data, config)
                
                if 'wavenumber' in processed_data.columns:
                    x_data = processed_data['wavenumber']
                    
                    # Find intensity columns
                    intensity_cols = [col for col in processed_data.columns 
                                    if col in ['absorbance', 'transmittance', 'intensity']]
                    
                    if intensity_cols:
                        y_data = processed_data[intensity_cols[0]]
                        
                        ax.plot(x_data, y_data, 
                               color=colors[i % len(colors)],
                               linewidth=config.style.line_width,
                               alpha=config.style.alpha,
                               label=label)
            
            # Apply styling
            self._apply_styling(fig, ax, config)
            
            # Ensure legend is shown for comparison
            ax.legend(fontsize=config.style.label_size)
            
            self.logger.info("Comparison plot created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison plot: {e}")
            return self._create_error_plot(str(e))
    
    def create_batch_plots(self, datasets: List[Tuple[pd.DataFrame, str]], 
                          output_dir: Path, config: Optional[GraphConfig] = None) -> List[Path]:
        """
        Create batch plots for multiple datasets.
        
        Args:
            datasets: List of (DataFrame, filename) tuples
            output_dir: Directory to save plots
            config: Graph configuration options
            
        Returns:
            List of saved file paths
        """
        if config is None:
            config = self.default_config
        
        saved_files = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Creating batch plots for {len(datasets)} datasets")
            
            for data, filename in datasets:
                try:
                    # Create individual plot
                    fig = self.create_spectral_plot(data, config)
                    
                    # Save plot
                    output_path = output_dir / f"{filename}.png"
                    fig.savefig(output_path, dpi=config.style.dpi, bbox_inches='tight')
                    saved_files.append(output_path)
                    
                    # Close figure to free memory
                    plt.close(fig)
                    
                except Exception as e:
                    self.logger.error(f"Failed to create plot for {filename}: {e}")
            
            self.logger.info(f"Batch plotting completed: {len(saved_files)} files saved")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Batch plotting failed: {e}")
            return saved_files
    
    def _preprocess_data(self, data: pd.DataFrame, config: GraphConfig) -> pd.DataFrame:
        """Preprocess data before plotting."""
        processed_data = data.copy()
        
        try:
            # Apply baseline correction if requested
            if config.baseline_correction:
                processed_data = self._apply_baseline_correction(processed_data)
            
            # Apply normalization if requested
            if config.normalization:
                processed_data = self._apply_normalization(processed_data, config.normalization)
            
            # Sort by wavenumber if present
            if 'wavenumber' in processed_data.columns:
                processed_data = processed_data.sort_values('wavenumber', ascending=False)
            
            return processed_data
            
        except Exception as e:
            self.logger.warning(f"Data preprocessing failed: {e}")
            return data
    
    def _create_line_plot(self, ax, data: pd.DataFrame, config: GraphConfig):
        """Create line plot."""
        if 'wavenumber' not in data.columns:
            return
        
        x_data = data['wavenumber']
        
        # Find intensity columns
        intensity_cols = [col for col in data.columns 
                         if col in ['absorbance', 'transmittance', 'intensity']]
        
        colors = self._get_color_palette(config.style.color_scheme, len(intensity_cols))
        
        for i, col in enumerate(intensity_cols):
            y_data = data[col]
            ax.plot(x_data, y_data,
                   color=colors[i % len(colors)],
                   linewidth=config.style.line_width,
                   alpha=config.style.alpha,
                   label=col.capitalize())
    
    def _create_scatter_plot(self, ax, data: pd.DataFrame, config: GraphConfig):
        """Create scatter plot."""
        if 'wavenumber' not in data.columns:
            return
        
        x_data = data['wavenumber']
        intensity_cols = [col for col in data.columns 
                         if col in ['absorbance', 'transmittance', 'intensity']]
        
        colors = self._get_color_palette(config.style.color_scheme, len(intensity_cols))
        
        for i, col in enumerate(intensity_cols):
            y_data = data[col]
            ax.scatter(x_data, y_data,
                      color=colors[i % len(colors)],
                      s=config.style.marker_size,
                      alpha=config.style.alpha,
                      label=col.capitalize())
    
    def _create_filled_plot(self, ax, data: pd.DataFrame, config: GraphConfig):
        """Create filled area plot."""
        if 'wavenumber' not in data.columns:
            return
        
        x_data = data['wavenumber']
        intensity_cols = [col for col in data.columns 
                         if col in ['absorbance', 'transmittance', 'intensity']]
        
        colors = self._get_color_palette(config.style.color_scheme, len(intensity_cols))
        
        for i, col in enumerate(intensity_cols):
            y_data = data[col]
            ax.fill_between(x_data, y_data,
                           color=colors[i % len(colors)],
                           alpha=config.style.alpha * 0.7,
                           label=col.capitalize())
    
    def _create_stacked_plot(self, ax, data: pd.DataFrame, config: GraphConfig):
        """Create stacked plot."""
        if 'wavenumber' not in data.columns:
            return
        
        x_data = data['wavenumber']
        intensity_cols = [col for col in data.columns 
                         if col in ['absorbance', 'transmittance', 'intensity']]
        
        colors = self._get_color_palette(config.style.color_scheme, len(intensity_cols))
        
        # Stack the data
        y_stack = np.zeros(len(x_data))
        for i, col in enumerate(intensity_cols):
            y_data = data[col].fillna(0)
            ax.fill_between(x_data, y_stack, y_stack + y_data,
                           color=colors[i % len(colors)],
                           alpha=config.style.alpha,
                           label=col.capitalize())
            y_stack += y_data
    
    def _create_overlay_plot(self, ax, data: pd.DataFrame, config: GraphConfig):
        """Create overlay plot with offset."""
        if 'wavenumber' not in data.columns:
            return
        
        x_data = data['wavenumber']
        intensity_cols = [col for col in data.columns 
                         if col in ['absorbance', 'transmittance', 'intensity']]
        
        colors = self._get_color_palette(config.style.color_scheme, len(intensity_cols))
        
        # Calculate offset
        offset_step = 0.5
        
        for i, col in enumerate(intensity_cols):
            y_data = data[col] + (i * offset_step)
            ax.plot(x_data, y_data,
                   color=colors[i % len(colors)],
                   linewidth=config.style.line_width,
                   alpha=config.style.alpha,
                   label=f"{col.capitalize()} (+{i * offset_step:.1f})")
    
    def _create_waterfall_plot(self, ax, data: pd.DataFrame, config: GraphConfig):
        """Create waterfall plot."""
        if 'wavenumber' not in data.columns:
            return
        
        x_data = data['wavenumber']
        intensity_cols = [col for col in data.columns 
                         if col in ['absorbance', 'transmittance', 'intensity']]
        
        colors = self._get_color_palette(config.style.color_scheme, len(intensity_cols))
        
        # Create 3D-like effect
        for i, col in enumerate(intensity_cols):
            y_data = data[col]
            z_offset = i * 0.1
            
            # Draw the main line
            ax.plot(x_data, y_data + z_offset,
                   color=colors[i % len(colors)],
                   linewidth=config.style.line_width,
                   alpha=config.style.alpha,
                   label=col.capitalize())
            
            # Add connecting lines for 3D effect
            if i > 0:
                for j in range(0, len(x_data), max(1, len(x_data) // 20)):
                    ax.plot([x_data.iloc[j], x_data.iloc[j]], 
                           [prev_y_data.iloc[j] + (i-1) * 0.1, y_data.iloc[j] + z_offset],
                           color='gray', alpha=0.3, linewidth=0.5)
            
            prev_y_data = y_data
    
    def _apply_styling(self, fig: Figure, ax, config: GraphConfig):
        """Apply styling to the plot."""
        # Set title and labels
        ax.set_title(config.title, fontsize=config.style.title_size, fontweight='bold')
        ax.set_xlabel(config.axis.x_label, fontsize=config.style.label_size)
        ax.set_ylabel(config.axis.y_label, fontsize=config.style.label_size)
        
        # Set axis ranges
        if config.axis.x_range:
            ax.set_xlim(config.axis.x_range)
        if config.axis.y_range:
            ax.set_ylim(config.axis.y_range)
        
        # Set axis scales
        ax.set_xscale(config.axis.x_scale)
        ax.set_yscale(config.axis.y_scale)
        
        # Invert axes if needed
        if config.axis.invert_x:
            ax.invert_xaxis()
        if config.axis.invert_y:
            ax.invert_yaxis()
        
        # Grid
        if config.style.grid:
            ax.grid(True, alpha=0.3)
        
        # Legend
        if config.style.legend:
            ax.legend(fontsize=config.style.label_size)
        
        # Tick parameters
        ax.tick_params(labelsize=config.style.tick_size)
        
        # Tight layout
        fig.tight_layout()
    
    def _add_annotations(self, ax, data: pd.DataFrame, config: GraphConfig):
        """Add annotations to the plot."""
        for annotation in config.annotations:
            try:
                if annotation['type'] == 'text':
                    ax.annotate(annotation['text'],
                               xy=(annotation['x'], annotation['y']),
                               fontsize=annotation.get('fontsize', 10),
                               color=annotation.get('color', 'black'))
                
                elif annotation['type'] == 'peak':
                    # Highlight peaks
                    self._annotate_peaks(ax, data, annotation)
                
                elif annotation['type'] == 'region':
                    # Highlight spectral regions
                    self._annotate_region(ax, annotation)
                
            except Exception as e:
                self.logger.warning(f"Failed to add annotation: {e}")
    
    def _annotate_peaks(self, ax, data: pd.DataFrame, annotation: Dict[str, Any]):
        """Annotate spectral peaks."""
        try:
            from scipy.signal import find_peaks
            
            if 'wavenumber' not in data.columns:
                return
            
            intensity_col = annotation.get('column', 'absorbance')
            if intensity_col not in data.columns:
                return
            
            x_data = data['wavenumber'].values
            y_data = data[intensity_col].values
            
            # Find peaks
            peaks, properties = find_peaks(y_data, **self.peak_detection_params)
            
            # Annotate peaks
            for peak in peaks:
                ax.annotate(f'{x_data[peak]:.0f}',
                           xy=(x_data[peak], y_data[peak]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
        
        except ImportError:
            self.logger.warning("scipy not available for peak detection")
        except Exception as e:
            self.logger.warning(f"Peak annotation failed: {e}")
    
    def _annotate_region(self, ax, annotation: Dict[str, Any]):
        """Annotate spectral regions."""
        try:
            x_range = annotation['x_range']
            y_range = ax.get_ylim()
            
            # Add shaded region
            ax.axvspan(x_range[0], x_range[1], 
                      alpha=annotation.get('alpha', 0.2),
                      color=annotation.get('color', 'gray'))
            
            # Add label
            if 'label' in annotation:
                ax.text((x_range[0] + x_range[1]) / 2, 
                       y_range[1] * 0.9,
                       annotation['label'],
                       ha='center', va='top',
                       fontsize=annotation.get('fontsize', 10))
        
        except Exception as e:
            self.logger.warning(f"Region annotation failed: {e}")
    
    def _get_color_palette(self, scheme: ColorScheme, n_colors: int) -> List[str]:
        """Get color palette for the specified scheme."""
        if scheme in self.color_palettes:
            colors = self.color_palettes[scheme]
            if len(colors) >= n_colors:
                return colors[:n_colors]
            else:
                # Repeat colors if needed
                return (colors * ((n_colors // len(colors)) + 1))[:n_colors]
        
        # Default fallback
        return self.color_palettes[ColorScheme.DEFAULT][:n_colors]
    
    def _apply_baseline_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply baseline correction to spectral data."""
        corrected_data = data.copy()
        
        try:
            intensity_cols = [col for col in data.columns 
                             if col in ['absorbance', 'transmittance', 'intensity']]
            
            for col in intensity_cols:
                y_data = corrected_data[col].values
                
                # Simple linear baseline correction
                baseline = np.linspace(y_data[0], y_data[-1], len(y_data))
                corrected_data[col] = y_data - baseline
            
            return corrected_data
            
        except Exception as e:
            self.logger.warning(f"Baseline correction failed: {e}")
            return data
    
    def _apply_normalization(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply normalization to spectral data."""
        normalized_data = data.copy()
        
        try:
            intensity_cols = [col for col in data.columns 
                             if col in ['absorbance', 'transmittance', 'intensity']]
            
            for col in intensity_cols:
                y_data = normalized_data[col].values
                
                if method == 'min_max':
                    y_min, y_max = y_data.min(), y_data.max()
                    if y_max != y_min:
                        normalized_data[col] = (y_data - y_min) / (y_max - y_min)
                
                elif method == 'z_score':
                    y_mean, y_std = y_data.mean(), y_data.std()
                    if y_std != 0:
                        normalized_data[col] = (y_data - y_mean) / y_std
                
                elif method == 'unit_vector':
                    y_norm = np.linalg.norm(y_data)
                    if y_norm != 0:
                        normalized_data[col] = y_data / y_norm
            
            return normalized_data
            
        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}")
            return data
    
    def _create_error_plot(self, error_message: str) -> Figure:
        """Create error plot when generation fails."""
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        ax.text(0.5, 0.5, f"Error creating plot:\n{error_message}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='red',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Plot Generation Error", fontsize=14)
        
        return fig
    
    def export_plot(self, fig: Figure, output_path: Path, format: str = 'png', 
                   dpi: int = 300, **kwargs) -> bool:
        """
        Export plot to file.
        
        Args:
            fig: Matplotlib Figure object
            output_path: Output file path
            format: Export format (png, pdf, svg, etc.)
            dpi: Resolution for raster formats
            **kwargs: Additional arguments for savefig
            
        Returns:
            bool: True if export successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fig.savefig(output_path, format=format, dpi=dpi, 
                       bbox_inches='tight', **kwargs)
            
            self.logger.info(f"Plot exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export plot: {e}")
            return False
    
    def create_qt_canvas(self, fig: Figure) -> FigureCanvas:
        """
        Create Qt canvas for embedding in PyQt application.
        
        Args:
            fig: Matplotlib Figure object
            
        Returns:
            FigureCanvas for Qt integration
        """
        try:
            canvas = FigureCanvas(fig)
            canvas.setParent(None)
            return canvas
            
        except Exception as e:
            self.logger.error(f"Failed to create Qt canvas: {e}")
            return None
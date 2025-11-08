"""
Comprehensive test cases for the spectral graph generation system.

Tests the SpectralGraphGenerator class functionality including:
- Professional graph generation
- Baseline + sample overlay visualization
- Batch processing capabilities
- Export functionality
- Error handling
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.graph_generator import (
    SpectralGraphGenerator, GraphConfig, BatchResult, ExportFormat, ThemeMode,
    GraphGenerator, PlotType, ColorScheme
)


class TestSpectralGraphGenerator(unittest.TestCase):
    """Test cases for SpectralGraphGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SpectralGraphGenerator()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample spectral data
        self.baseline_data = self._create_sample_data("baseline", 4000, 400, 1000)
        self.sample_data = self._create_sample_data("sample", 4000, 400, 1000)
        
        # Create sample datasets for batch testing
        self.sample_datasets = [
            (self._create_sample_data("sample1", 4000, 400, 800), "sample1.csv"),
            (self._create_sample_data("sample2", 4000, 400, 900), "sample2.csv"),
            (self._create_sample_data("sample3", 4000, 400, 1100), "sample3.csv")
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        plt.close('all')  # Close all matplotlib figures
    
    def _create_sample_data(self, name: str, start_wave: float, end_wave: float, 
                           num_points: int) -> pd.DataFrame:
        """Create sample spectral data for testing."""
        wavenumbers = np.linspace(start_wave, end_wave, num_points)
        
        # Create realistic spectral data with some peaks
        absorbance = np.random.normal(0.5, 0.1, num_points)
        
        # Add some characteristic peaks
        peak_positions = [3500, 2900, 1650, 1450, 1000]
        for peak_pos in peak_positions:
            if end_wave <= peak_pos <= start_wave:
                peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
                absorbance[peak_idx] += np.random.uniform(0.3, 0.8)
        
        # Ensure positive values
        absorbance = np.maximum(absorbance, 0.01)
        
        return pd.DataFrame({
            'Wavenumber': wavenumbers,
            'Absorbance': absorbance
        })
    
    def test_initialization(self):
        """Test SpectralGraphGenerator initialization."""
        generator = SpectralGraphGenerator()
        self.assertIsNotNone(generator.logger)
        self.assertIsNone(generator.theme_manager)
        
        # Test with theme manager
        mock_theme_manager = object()
        generator_with_theme = SpectralGraphGenerator(theme_manager=mock_theme_manager)
        self.assertEqual(generator_with_theme.theme_manager, mock_theme_manager)
    
    def test_graph_config_defaults(self):
        """Test GraphConfig default values."""
        config = GraphConfig()
        
        self.assertEqual(config.figure_size, (10, 6))
        self.assertEqual(config.dpi, 300)
        self.assertEqual(config.baseline_color, '#2E7D32')
        self.assertEqual(config.sample_color, '#1976D2')
        self.assertEqual(config.background_color, 'white')
        self.assertTrue(config.show_grid)
        self.assertEqual(config.font_family, 'Arial')
        self.assertEqual(config.theme, ThemeMode.LIGHT)
    
    def test_generate_comparison_graph_basic(self):
        """Test basic comparison graph generation."""
        fig = self.generator.generate_comparison_graph(
            self.baseline_data, self.sample_data, "baseline", "sample"
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)  # Baseline and sample lines
        
        # Check axis labels
        self.assertEqual(ax.get_xlabel(), 'Wavenumber (cm⁻¹)')
        self.assertEqual(ax.get_ylabel(), 'Absorbance')
        
        # Check if x-axis is inverted (IR spectroscopy convention)
        self.assertTrue(ax.xaxis_inverted())
        
        # Check legend
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(len(legend.get_texts()), 2)
    
    def test_generate_comparison_graph_with_config(self):
        """Test comparison graph generation with custom configuration."""
        config = GraphConfig(
            figure_size=(12, 8),
            baseline_color='red',
            sample_color='blue',
            show_grid=False,
            line_width=2.0
        )
        
        fig = self.generator.generate_comparison_graph(
            self.baseline_data, self.sample_data, "baseline", "sample", config
        )
        
        self.assertIsInstance(fig, Figure)
        self.assertEqual(fig.get_size_inches().tolist(), [12, 8])
        
        ax = fig.axes[0]
        
        # Check line colors and width
        baseline_line = ax.lines[0]
        sample_line = ax.lines[1]
        
        self.assertEqual(baseline_line.get_color(), 'red')
        self.assertEqual(sample_line.get_color(), 'blue')
        self.assertEqual(baseline_line.get_linewidth(), 2.0)
        self.assertEqual(sample_line.get_linewidth(), 2.0)
    
    def test_generate_comparison_graph_invalid_data(self):
        """Test comparison graph generation with invalid data."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        fig = self.generator.generate_comparison_graph(
            empty_data, self.sample_data, "empty", "sample"
        )
        
        self.assertIsInstance(fig, Figure)
        # Should create error figure
        ax = fig.axes[0]
        self.assertEqual(len(ax.texts), 1)  # Error message text
    
    def test_extract_spectral_data(self):
        """Test spectral data extraction from DataFrame."""
        x_data, y_data = self.generator._extract_spectral_data(self.baseline_data)
        
        self.assertIsNotNone(x_data)
        self.assertIsNotNone(y_data)
        self.assertEqual(len(x_data), len(y_data))
        self.assertTrue(np.all(x_data >= 400))
        self.assertTrue(np.all(x_data <= 4000))
        self.assertTrue(np.all(y_data >= 0))
    
    def test_extract_spectral_data_different_column_names(self):
        """Test spectral data extraction with different column names."""
        # Test with different column names
        data_alt_names = pd.DataFrame({
            'Wave': np.linspace(4000, 400, 100),
            'Intensity': np.random.uniform(0, 1, 100)
        })
        
        x_data, y_data = self.generator._extract_spectral_data(data_alt_names)
        
        self.assertIsNotNone(x_data)
        self.assertIsNotNone(y_data)
        self.assertEqual(len(x_data), 100)
    
    def test_align_spectral_data(self):
        """Test spectral data alignment."""
        baseline_x = np.linspace(4000, 500, 1000)
        baseline_y = np.random.uniform(0, 1, 1000)
        sample_x = np.linspace(3800, 400, 800)
        sample_y = np.random.uniform(0, 1, 800)
        
        common_x, aligned_baseline_y, aligned_sample_y = self.generator._align_spectral_data(
            baseline_x, baseline_y, sample_x, sample_y
        )
        
        self.assertEqual(len(common_x), len(aligned_baseline_y))
        self.assertEqual(len(common_x), len(aligned_sample_y))
        
        # Check that common range is within overlap
        self.assertTrue(np.all(common_x >= 500))  # max of minimums
        self.assertTrue(np.all(common_x <= 3800))  # min of maximums
    
    def test_save_graph_png(self):
        """Test saving graph as PNG."""
        fig = self.generator.generate_comparison_graph(
            self.baseline_data, self.sample_data, "baseline", "sample"
        )
        
        output_path = self.temp_dir / "test_graph.png"
        success = self.generator.save_graph(fig, output_path, "png")
        
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
    
    def test_save_graph_jpg(self):
        """Test saving graph as JPG."""
        fig = self.generator.generate_comparison_graph(
            self.baseline_data, self.sample_data, "baseline", "sample"
        )
        
        output_path = self.temp_dir / "test_graph.jpg"
        success = self.generator.save_graph(fig, output_path, "jpg")
        
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
    
    def test_batch_processing(self):
        """Test batch graph generation."""
        result = self.generator.generate_batch_graphs(
            self.baseline_data,
            self.sample_datasets,
            str(self.temp_dir),
            format="png",
            baseline_name="baseline"
        )
        
        self.assertIsInstance(result, BatchResult)
        self.assertEqual(result.total_graphs, 3)
        self.assertEqual(result.successful, 3)
        self.assertEqual(result.failed, 0)
        self.assertEqual(len(result.output_files), 3)
        self.assertEqual(len(result.errors), 0)
        self.assertGreater(result.processing_time, 0)
        
        # Check that files were created
        for output_file in result.output_files:
            file_path = Path(output_file)
            self.assertTrue(file_path.exists())
            self.assertGreater(file_path.stat().st_size, 0)
    
    def test_batch_processing_with_progress_callback(self):
        """Test batch processing with progress callback."""
        progress_updates = []
        
        def progress_callback(message: str, percentage: float):
            progress_updates.append((message, percentage))
        
        result = self.generator.generate_batch_graphs(
            self.baseline_data,
            self.sample_datasets[:2],  # Use fewer samples for faster test
            str(self.temp_dir),
            progress_callback=progress_callback
        )
        
        self.assertEqual(result.successful, 2)
        self.assertGreater(len(progress_updates), 0)
        
        # Check that progress goes from 0 to 100
        percentages = [update[1] for update in progress_updates]
        self.assertEqual(min(percentages), 0.0)
        self.assertEqual(max(percentages), 100.0)
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        test_cases = [
            ("normal_file.csv", "normal_file"),
            ("file with spaces.csv", "file with spaces"),
            ("file<>:\"/\\|?*.csv", "file_________.csv"),
            ("very_long_filename_that_exceeds_fifty_characters_limit.csv", 
             "very_long_filename_that_exceeds_fifty_characters")
        ]
        
        for input_name, expected in test_cases:
            result = self.generator._sanitize_filename(input_name)
            self.assertEqual(result, expected[:50])  # Ensure length limit
    
    def test_safe_filename_creation(self):
        """Test safe filename creation with conflict resolution."""
        # Create a file to test conflict resolution
        existing_file = self.temp_dir / "sample1_vs_baseline.png"
        existing_file.touch()
        
        filename = self.generator._create_safe_filename(
            "sample1", "baseline", "png", self.temp_dir
        )
        
        # Should create a different filename to avoid conflict
        self.assertNotEqual(filename, "sample1_vs_baseline.png")
        self.assertTrue(filename.startswith("sample1_vs_baseline_"))
        self.assertTrue(filename.endswith(".png"))
    
    def test_error_figure_creation(self):
        """Test error figure creation."""
        config = GraphConfig()
        error_message = "Test error message"
        
        fig = self.generator._create_error_figure(error_message, config)
        
        self.assertIsInstance(fig, Figure)
        ax = fig.axes[0]
        self.assertEqual(len(ax.texts), 1)
        
        # Check that error message is in the text
        text_content = ax.texts[0].get_text()
        self.assertIn(error_message, text_content)


class TestLegacyGraphGenerator(unittest.TestCase):
    """Test cases for legacy GraphGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = GraphGenerator()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample spectral data
        self.sample_data = pd.DataFrame({
            'wavenumber': np.linspace(4000, 400, 1000),
            'absorbance': np.random.uniform(0, 2, 1000)
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_legacy_initialization(self):
        """Test legacy GraphGenerator initialization."""
        generator = GraphGenerator()
        self.assertIsNotNone(generator.logger)
        self.assertIsNotNone(generator.spectral_generator)
        self.assertIsNotNone(generator.color_palettes)
        self.assertIsNotNone(generator.default_config)
    
    def test_create_spectral_plot(self):
        """Test legacy spectral plot creation."""
        fig = self.generator.create_spectral_plot(self.sample_data)
        
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertGreater(len(ax.lines), 0)
    
    def test_create_comparison_plot(self):
        """Test legacy comparison plot creation."""
        datasets = [
            (self.sample_data, "Sample 1"),
            (self.sample_data, "Sample 2")
        ]
        
        fig = self.generator.create_comparison_plot(datasets)
        
        self.assertIsInstance(fig, Figure)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)
        
        # Check legend
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
    
    def test_export_plot(self):
        """Test legacy plot export."""
        fig = self.generator.create_spectral_plot(self.sample_data)
        output_path = self.temp_dir / "legacy_test.png"
        
        success = self.generator.export_plot(fig, output_path)
        
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
    
    def test_create_qt_canvas(self):
        """Test Qt canvas creation."""
        fig = self.generator.create_spectral_plot(self.sample_data)
        canvas = self.generator.create_qt_canvas(fig)
        
        # Canvas creation might fail in test environment without Qt
        # Just test that method doesn't crash
        self.assertTrue(True)


class TestBatchResult(unittest.TestCase):
    """Test cases for BatchResult data class."""
    
    def test_batch_result_creation(self):
        """Test BatchResult creation and attributes."""
        result = BatchResult(
            total_graphs=5,
            successful=4,
            failed=1,
            output_files=["file1.png", "file2.png"],
            errors=["Error processing file3"],
            processing_time=10.5,
            average_time_per_graph=2.1
        )
        
        self.assertEqual(result.total_graphs, 5)
        self.assertEqual(result.successful, 4)
        self.assertEqual(result.failed, 1)
        self.assertEqual(len(result.output_files), 2)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.processing_time, 10.5)
        self.assertEqual(result.average_time_per_graph, 2.1)
    
    def test_batch_result_defaults(self):
        """Test BatchResult default values."""
        result = BatchResult(total_graphs=0, successful=0, failed=0)
        
        self.assertEqual(len(result.output_files), 0)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.processing_time, 0.0)
        self.assertEqual(result.average_time_per_graph, 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete graph generation system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = SpectralGraphGenerator()
        
        # Create more realistic test data
        self.baseline_data = self._create_realistic_spectrum("baseline")
        self.sample_data = self._create_realistic_spectrum("sample")
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def _create_realistic_spectrum(self, name: str) -> pd.DataFrame:
        """Create realistic IR spectrum data."""
        wavenumbers = np.linspace(4000, 400, 1800)
        
        # Create baseline spectrum
        absorbance = np.random.normal(0.1, 0.02, len(wavenumbers))
        
        # Add characteristic IR peaks
        peaks = {
            3500: 0.8,  # O-H stretch
            2900: 0.6,  # C-H stretch
            1650: 1.2,  # C=O stretch
            1450: 0.4,  # C-H bend
            1000: 0.7   # C-O stretch
        }
        
        for peak_pos, intensity in peaks.items():
            if 400 <= peak_pos <= 4000:
                # Add Gaussian peak
                peak_width = 50
                peak_profile = intensity * np.exp(-((wavenumbers - peak_pos) / peak_width) ** 2)
                absorbance += peak_profile
        
        # Add some noise variation for sample vs baseline
        if name == "sample":
            absorbance *= np.random.uniform(0.9, 1.1, len(wavenumbers))
            absorbance += np.random.normal(0, 0.05, len(wavenumbers))
        
        # Ensure positive values
        absorbance = np.maximum(absorbance, 0.001)
        
        return pd.DataFrame({
            'Wavenumber': wavenumbers,
            'Absorbance': absorbance
        })
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Generate comparison graph
        fig = self.generator.generate_comparison_graph(
            self.baseline_data, self.sample_data, "baseline.csv", "sample.csv"
        )
        
        self.assertIsInstance(fig, Figure)
        
        # 2. Save graph
        output_path = self.temp_dir / "comparison.png"
        success = self.generator.save_graph(fig, output_path)
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
        
        # 3. Verify graph properties
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)
        self.assertTrue(ax.xaxis_inverted())
        
        # 4. Check file size (should be reasonable for a graph)
        file_size = output_path.stat().st_size
        self.assertGreater(file_size, 10000)  # At least 10KB
        self.assertLess(file_size, 1000000)   # Less than 1MB
    
    def test_batch_processing_integration(self):
        """Test batch processing integration."""
        # Create multiple sample datasets
        sample_datasets = []
        for i in range(3):
            sample_data = self._create_realistic_spectrum(f"sample_{i}")
            sample_datasets.append((sample_data, f"sample_{i}.csv"))
        
        # Process batch
        result = self.generator.generate_batch_graphs(
            self.baseline_data,
            sample_datasets,
            str(self.temp_dir),
            format="png",
            baseline_name="baseline.csv"
        )
        
        # Verify results
        self.assertEqual(result.total_graphs, 3)
        self.assertEqual(result.successful, 3)
        self.assertEqual(result.failed, 0)
        self.assertEqual(len(result.output_files), 3)
        
        # Verify all files exist and have reasonable sizes
        for output_file in result.output_files:
            file_path = Path(output_file)
            self.assertTrue(file_path.exists())
            file_size = file_path.stat().st_size
            self.assertGreater(file_size, 5000)  # At least 5KB


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
#!/usr/bin/env python3
"""
Standalone test for the spectral graph generator.
Tests the core functionality without complex imports.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only what we need directly
from core.graph_generator import SpectralGraphGenerator, GraphConfig, BatchResult


def create_sample_data(name: str, start_wave: float = 4000, end_wave: float = 400, num_points: int = 1000):
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


def test_basic_functionality():
    """Test basic graph generation functionality."""
    print("Testing basic functionality...")
    
    # Create test data
    baseline_data = create_sample_data("baseline")
    sample_data = create_sample_data("sample")
    
    # Initialize generator
    generator = SpectralGraphGenerator()
    
    # Test 1: Basic comparison graph
    print("  ✓ Testing comparison graph generation...")
    fig = generator.generate_comparison_graph(
        baseline_data, sample_data, "baseline.csv", "sample.csv"
    )
    
    assert isinstance(fig, Figure), "Should return a Figure object"
    assert len(fig.axes) == 1, "Should have one axis"
    
    ax = fig.axes[0]
    assert len(ax.lines) == 2, "Should have two lines (baseline and sample)"
    assert ax.xaxis_inverted(), "X-axis should be inverted for IR spectroscopy"
    
    # Check axis labels
    assert ax.get_xlabel() == 'Wavenumber (cm⁻¹)', "X-axis label should be correct"
    assert ax.get_ylabel() == 'Absorbance', "Y-axis label should be correct"
    
    # Check legend
    legend = ax.get_legend()
    assert legend is not None, "Should have a legend"
    assert len(legend.get_texts()) == 2, "Legend should have two entries"
    
    plt.close(fig)
    print("  ✓ Basic comparison graph test passed")


def test_custom_configuration():
    """Test graph generation with custom configuration."""
    print("Testing custom configuration...")
    
    baseline_data = create_sample_data("baseline")
    sample_data = create_sample_data("sample")
    generator = SpectralGraphGenerator()
    
    # Test with custom config
    config = GraphConfig(
        figure_size=(12, 8),
        baseline_color='red',
        sample_color='blue',
        show_grid=False,
        line_width=2.5
    )
    
    fig = generator.generate_comparison_graph(
        baseline_data, sample_data, "baseline", "sample", config
    )
    
    assert fig.get_size_inches().tolist() == [12, 8], "Figure size should match config"
    
    ax = fig.axes[0]
    baseline_line = ax.lines[0]
    sample_line = ax.lines[1]
    
    assert baseline_line.get_color() == 'red', "Baseline color should be red"
    assert sample_line.get_color() == 'blue', "Sample color should be blue"
    assert baseline_line.get_linewidth() == 2.5, "Line width should match config"
    
    plt.close(fig)
    print("  ✓ Custom configuration test passed")


def test_file_operations():
    """Test file saving operations."""
    print("Testing file operations...")
    
    baseline_data = create_sample_data("baseline")
    sample_data = create_sample_data("sample")
    generator = SpectralGraphGenerator()
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Generate graph
        fig = generator.generate_comparison_graph(
            baseline_data, sample_data, "baseline", "sample"
        )
        
        # Test PNG save
        png_path = temp_dir / "test_graph.png"
        success = generator.save_graph(fig, png_path, "png")
        
        assert success, "PNG save should succeed"
        assert png_path.exists(), "PNG file should exist"
        assert png_path.stat().st_size > 0, "PNG file should not be empty"
        
        # Test JPG save
        jpg_path = temp_dir / "test_graph.jpg"
        success = generator.save_graph(fig, jpg_path, "jpg")
        
        assert success, "JPG save should succeed"
        assert jpg_path.exists(), "JPG file should exist"
        assert jpg_path.stat().st_size > 0, "JPG file should not be empty"
        
        plt.close(fig)
        print("  ✓ File operations test passed")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_batch_processing():
    """Test batch processing functionality."""
    print("Testing batch processing...")
    
    baseline_data = create_sample_data("baseline")
    generator = SpectralGraphGenerator()
    
    # Create sample datasets
    sample_datasets = [
        (create_sample_data("sample1"), "sample1.csv"),
        (create_sample_data("sample2"), "sample2.csv"),
        (create_sample_data("sample3"), "sample3.csv")
    ]
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test batch processing
        result = generator.generate_batch_graphs(
            baseline_data,
            sample_datasets,
            str(temp_dir),
            format="png",
            baseline_name="baseline.csv"
        )
        
        assert isinstance(result, BatchResult), "Should return BatchResult"
        assert result.total_graphs == 3, "Should process 3 graphs"
        assert result.successful == 3, "All graphs should succeed"
        assert result.failed == 0, "No graphs should fail"
        assert len(result.output_files) == 3, "Should have 3 output files"
        assert len(result.errors) == 0, "Should have no errors"
        assert result.processing_time > 0, "Should have positive processing time"
        
        # Check that files were created
        for output_file in result.output_files:
            file_path = Path(output_file)
            assert file_path.exists(), f"Output file should exist: {output_file}"
            assert file_path.stat().st_size > 0, f"Output file should not be empty: {output_file}"
        
        print("  ✓ Batch processing test passed")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_data_extraction():
    """Test spectral data extraction."""
    print("Testing data extraction...")
    
    generator = SpectralGraphGenerator()
    
    # Test with standard column names
    data1 = pd.DataFrame({
        'Wavenumber': np.linspace(4000, 400, 100),
        'Absorbance': np.random.uniform(0, 1, 100)
    })
    
    x_data, y_data = generator._extract_spectral_data(data1)
    
    assert x_data is not None, "Should extract x data"
    assert y_data is not None, "Should extract y data"
    assert len(x_data) == len(y_data), "X and Y data should have same length"
    assert len(x_data) == 100, "Should have correct number of points"
    
    # Test with alternative column names
    data2 = pd.DataFrame({
        'Wave': np.linspace(4000, 400, 50),
        'Intensity': np.random.uniform(0, 1, 50)
    })
    
    x_data2, y_data2 = generator._extract_spectral_data(data2)
    
    assert x_data2 is not None, "Should extract x data with alternative names"
    assert y_data2 is not None, "Should extract y data with alternative names"
    assert len(x_data2) == 50, "Should have correct number of points"
    
    print("  ✓ Data extraction test passed")


def test_filename_handling():
    """Test filename sanitization and conflict resolution."""
    print("Testing filename handling...")
    
    generator = SpectralGraphGenerator()
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test sanitization
        test_cases = [
            ("normal_file.csv", "normal_file"),
            ("file with spaces.csv", "file with spaces"),
            ("file<>:\"/\\|?*.csv", "file_________"),
        ]
        
        for input_name, expected_start in test_cases:
            result = generator._sanitize_filename(input_name)
            assert result.startswith(expected_start[:20]), f"Sanitization failed for {input_name}"
            assert len(result) <= 50, "Sanitized filename should not exceed 50 characters"
        
        # Test conflict resolution
        existing_file = temp_dir / "sample_vs_baseline.png"
        existing_file.touch()
        
        filename = generator._create_safe_filename("sample", "baseline", "png", temp_dir)
        
        assert filename != "sample_vs_baseline.png", "Should create different filename to avoid conflict"
        assert filename.endswith(".png"), "Should maintain file extension"
        assert "sample_vs_baseline" in filename, "Should contain original name parts"
        
        print("  ✓ Filename handling test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    generator = SpectralGraphGenerator()
    
    # Test with empty DataFrame
    empty_data = pd.DataFrame()
    sample_data = create_sample_data("sample")
    
    fig = generator.generate_comparison_graph(
        empty_data, sample_data, "empty", "sample"
    )
    
    assert isinstance(fig, Figure), "Should return Figure even with invalid data"
    
    # Should create error figure
    ax = fig.axes[0]
    assert len(ax.texts) > 0, "Should have error message text"
    
    plt.close(fig)
    print("  ✓ Error handling test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SPECTRAL GRAPH GENERATOR STANDALONE TESTS")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_custom_configuration,
        test_file_operations,
        test_batch_processing,
        test_data_extraction,
        test_filename_handling,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
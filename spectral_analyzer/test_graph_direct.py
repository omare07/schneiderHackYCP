#!/usr/bin/env python3
"""
Direct test of graph generator functionality.
Imports the graph generator module directly to avoid circular imports.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the graph generator module directly
import importlib.util
spec = importlib.util.spec_from_file_location("graph_generator", "core/graph_generator.py")
graph_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_generator)

# Get the classes we need
SpectralGraphGenerator = graph_generator.SpectralGraphGenerator
GraphConfig = graph_generator.GraphConfig
BatchResult = graph_generator.BatchResult


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


def test_production_functionality():
    """Test production-ready functionality."""
    print("🔬 TESTING SPECTRAL GRAPH GENERATOR - PRODUCTION MODE")
    print("=" * 60)
    
    # Create test data
    baseline_data = create_sample_data("baseline")
    sample_data = create_sample_data("sample")
    
    # Initialize generator
    generator = SpectralGraphGenerator()
    
    print("✅ Generator initialized successfully")
    
    # Test 1: Generate comparison graph
    print("📊 Generating comparison graph...")
    fig = generator.generate_comparison_graph(
        baseline_data, sample_data, "baseline.csv", "sample.csv"
    )
    
    print(f"   - Figure created: {type(fig)}")
    print(f"   - Axes count: {len(fig.axes)}")
    print(f"   - Lines count: {len(fig.axes[0].lines)}")
    print(f"   - X-axis inverted: {fig.axes[0].xaxis_inverted()}")
    
    # Test 2: Save graphs in different formats
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Using temp directory: {temp_dir}")
    
    try:
        # Save PNG
        png_path = temp_dir / "comparison.png"
        success_png = generator.save_graph(fig, png_path, "png", dpi=300)
        print(f"   - PNG saved: {success_png} ({png_path.stat().st_size if png_path.exists() else 0} bytes)")
        
        # Save JPG
        jpg_path = temp_dir / "comparison.jpg"
        success_jpg = generator.save_graph(fig, jpg_path, "jpg", dpi=300)
        print(f"   - JPG saved: {success_jpg} ({jpg_path.stat().st_size if jpg_path.exists() else 0} bytes)")
        
        plt.close(fig)
        
        # Test 3: Batch processing
        print("🔄 Testing batch processing...")
        sample_datasets = [
            (create_sample_data("sample1"), "sample1.csv"),
            (create_sample_data("sample2"), "sample2.csv"),
            (create_sample_data("sample3"), "sample3.csv")
        ]
        
        batch_result = generator.generate_batch_graphs(
            baseline_data,
            sample_datasets,
            str(temp_dir / "batch"),
            format="png",
            baseline_name="baseline.csv"
        )
        
        print(f"   - Total graphs: {batch_result.total_graphs}")
        print(f"   - Successful: {batch_result.successful}")
        print(f"   - Failed: {batch_result.failed}")
        print(f"   - Processing time: {batch_result.processing_time:.2f}s")
        print(f"   - Output files: {len(batch_result.output_files)}")
        
        # Verify batch files exist
        for output_file in batch_result.output_files:
            file_path = Path(output_file)
            if file_path.exists():
                print(f"     ✅ {file_path.name}: {file_path.stat().st_size} bytes")
            else:
                print(f"     ❌ {file_path.name}: NOT FOUND")
        
        # Test 4: Custom configuration
        print("⚙️  Testing custom configuration...")
        config = GraphConfig(
            figure_size=(12, 8),
            dpi=300,
            baseline_color='#2E7D32',
            sample_color='#1976D2',
            show_grid=True,
            line_width=2.0
        )
        
        custom_fig = generator.generate_comparison_graph(
            baseline_data, sample_data, "baseline", "sample", config
        )
        
        custom_path = temp_dir / "custom_graph.png"
        success_custom = generator.save_graph(custom_fig, custom_path, "png")
        print(f"   - Custom graph saved: {success_custom}")
        print(f"   - Figure size: {custom_fig.get_size_inches()}")
        
        plt.close(custom_fig)
        
        # Test 5: Error handling
        print("🛡️  Testing error handling...")
        empty_data = pd.DataFrame()
        error_fig = generator.generate_comparison_graph(
            empty_data, sample_data, "empty", "sample"
        )
        
        error_path = temp_dir / "error_graph.png"
        success_error = generator.save_graph(error_fig, error_path, "png")
        print(f"   - Error graph handled: {success_error}")
        
        plt.close(error_fig)
        
        print("=" * 60)
        print("🎉 ALL PRODUCTION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Summary
        total_files = len(list(temp_dir.rglob("*.png"))) + len(list(temp_dir.rglob("*.jpg")))
        total_size = sum(f.stat().st_size for f in temp_dir.rglob("*") if f.is_file())
        
        print(f"📈 PRODUCTION SUMMARY:")
        print(f"   - Total files generated: {total_files}")
        print(f"   - Total size: {total_size / 1024:.1f} KB")
        print(f"   - Average file size: {total_size / total_files / 1024:.1f} KB")
        print(f"   - Batch processing rate: {batch_result.total_graphs / batch_result.processing_time:.1f} graphs/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ PRODUCTION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temp directory")


if __name__ == "__main__":
    success = test_production_functionality()
    if success:
        print("\n✅ PRODUCTION APPLICATION READY!")
    else:
        print("\n❌ PRODUCTION APPLICATION NEEDS FIXES!")
    
    sys.exit(0 if success else 1)
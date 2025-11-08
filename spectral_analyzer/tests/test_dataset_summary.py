#!/usr/bin/env python3
"""
Test Dataset Summary Generator

This script generates a comprehensive summary of the test dataset,
including statistics, coverage analysis, and usage recommendations.
"""

import os
from pathlib import Path
from typing import Dict, List
import json


def generate_dataset_summary(test_data_dir: str = "test_data") -> Dict:
    """Generate comprehensive dataset summary."""
    test_dir = Path(test_data_dir)
    
    # Get all CSV files
    csv_files = list(test_dir.glob("*.csv"))
    
    # Categorize files
    categories = {
        'baseline': [],
        'sample': [],
        'legacy': [],
        'performance': []
    }
    
    legacy_files = {
        'european_format.csv', 'multi_column.csv', 'no_header.csv',
        'problematic_data.csv', 'sample_spectral.csv', 'tab_delimited.csv'
    }
    
    for file_path in csv_files:
        filename = file_path.name
        file_size = file_path.stat().st_size
        
        file_info = {
            'filename': filename,
            'size_bytes': file_size,
            'size_kb': file_size / 1024
        }
        
        if filename.startswith('baseline_'):
            categories['baseline'].append(file_info)
        elif filename.startswith('sample_'):
            categories['sample'].append(file_info)
        elif filename.startswith('performance_'):
            categories['performance'].append(file_info)
        elif filename in legacy_files:
            categories['legacy'].append(file_info)
    
    # Calculate statistics
    total_files = len(csv_files)
    total_size = sum(f.stat().st_size for f in csv_files)
    
    # Format issues covered
    format_issues = {
        'European format': any('european' in f.name for f in csv_files),
        'Tab delimited': any('tab' in f.name for f in csv_files),
        'No headers': any('no_header' in f.name for f in csv_files),
        'Extra columns': any('extra_column' in f.name for f in csv_files),
        'Wrong order': any('wrong_order' in f.name for f in csv_files),
        'Mixed issues': any('mixed' in f.name for f in csv_files),
        'Scale issues': any('scale' in f.name for f in csv_files),
        'Encoding issues': any('encoding' in f.name for f in csv_files),
        'Noisy data': any('noisy' in f.name for f in csv_files),
        'Different spectroscopy': any('raman' in f.name for f in csv_files)
    }
    
    issues_covered = sum(1 for covered in format_issues.values() if covered)
    coverage_percentage = (issues_covered / len(format_issues)) * 100
    
    summary = {
        'dataset_info': {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_kb': total_size / 1024,
            'total_size_mb': total_size / (1024 * 1024)
        },
        'categories': categories,
        'format_coverage': {
            'issues_covered': issues_covered,
            'total_issues': len(format_issues),
            'coverage_percentage': coverage_percentage,
            'format_issues': format_issues
        },
        'files_by_size': {
            'small': [f for f in csv_files if f.stat().st_size < 1000],
            'medium': [f for f in csv_files if 1000 <= f.stat().st_size < 50000],
            'large': [f for f in csv_files if 50000 <= f.stat().st_size < 200000],
            'very_large': [f for f in csv_files if f.stat().st_size >= 200000]
        }
    }
    
    return summary


def print_dataset_summary():
    """Print formatted dataset summary."""
    summary = generate_dataset_summary()
    
    print("📊 Spectral Analysis Test Dataset Summary")
    print("=" * 60)
    
    # Basic statistics
    info = summary['dataset_info']
    print(f"📁 Total Files: {info['total_files']}")
    print(f"💾 Total Size: {info['total_size_kb']:.1f} KB ({info['total_size_mb']:.2f} MB)")
    print()
    
    # Category breakdown
    print("📂 File Categories:")
    for category, files in summary['categories'].items():
        if files:
            total_size = sum(f['size_kb'] for f in files)
            print(f"   {category.title()}: {len(files)} files ({total_size:.1f} KB)")
    print()
    
    # Format coverage
    coverage = summary['format_coverage']
    print(f"🎯 Format Coverage: {coverage['coverage_percentage']:.0f}% ({coverage['issues_covered']}/{coverage['total_issues']} issues)")
    print("   Format Issues Covered:")
    for issue, covered in coverage['format_issues'].items():
        status = "✅" if covered else "❌"
        print(f"   {status} {issue}")
    print()
    
    # Size distribution
    print("📏 File Size Distribution:")
    size_categories = summary['files_by_size']
    for size_cat, files in size_categories.items():
        if files:
            print(f"   {size_cat.title()}: {len(files)} files")
    print()
    
    # Key highlights
    print("⭐ Key Highlights:")
    print("   • Realistic spectroscopic data with proper peak features")
    print("   • Multiple CSV format variations (delimiters, decimals, encodings)")
    print("   • Different spectroscopy types (FTIR, Raman)")
    print("   • Performance testing with large datasets")
    print("   • Error handling scenarios")
    print("   • Comprehensive documentation and validation")
    print()
    
    print("🚀 Usage:")
    print("   • Run validation: python3 validate_test_data.py")
    print("   • Integration tests: python3 test_integration_scenarios.py")
    print("   • Complete suite: python3 run_comprehensive_tests.py")
    print("   • Interactive demo: python3 demo_test_dataset.py")


if __name__ == "__main__":
    print_dataset_summary()
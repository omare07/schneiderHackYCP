#!/usr/bin/env python3
"""
Test Dataset Demonstration Script

This script provides an interactive demonstration of the comprehensive test dataset,
showcasing the various format challenges and AI normalization capabilities.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatasetDemo:
    """Interactive demonstration of the test dataset."""
    
    def __init__(self, test_data_dir: str = "test_data"):
        """Initialize demo with test data directory."""
        self.test_data_dir = Path(test_data_dir)
        
        # File categories for organized demonstration
        self.file_categories = {
            'baseline': {
                'title': '🟢 Baseline Files (Perfect References)',
                'files': [
                    ('baseline_perfect.csv', 'Perfect standard format'),
                    ('baseline_european.csv', 'European format (semicolon, comma decimal)'),
                    ('baseline_raman.csv', 'Raman spectroscopy data')
                ]
            },
            'format_issues': {
                'title': '🔴 Sample Files (Format Issues)',
                'files': [
                    ('sample_tab_delimited.csv', 'Tab-separated values with metadata'),
                    ('sample_no_headers.csv', 'No column headers'),
                    ('sample_extra_columns.csv', 'Many extra metadata columns'),
                    ('sample_wrong_order.csv', 'Ascending wavenumbers (wrong for FTIR)'),
                    ('sample_mixed_issues.csv', 'Multiple format problems'),
                    ('sample_scale_issues.csv', 'Wrong magnitude scaling'),
                    ('sample_encoding_test.csv', 'Latin-1 encoding with French text')
                ]
            },
            'spectroscopy_types': {
                'title': '🔬 Different Spectroscopy Types',
                'files': [
                    ('sample_raman_spectrum.csv', 'Raman spectroscopy with intensity counts'),
                    ('sample_noisy_data.csv', 'FTIR with noise and artifacts')
                ]
            },
            'performance': {
                'title': '⚡ Performance Testing',
                'files': [
                    ('performance_large_file.csv', 'Large dataset (10,000 points)')
                ]
            }
        }
    
    def display_file_preview(self, filename: str, max_lines: int = 10) -> None:
        """Display a preview of a test file."""
        file_path = self.test_data_dir / filename
        
        if not file_path.exists():
            print(f"   ❌ File not found: {filename}")
            return
        
        try:
            # Detect encoding
            encodings = ['utf-8', 'latin-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"   ❌ Could not read file: {filename}")
                return
            
            lines = content.split('\n')
            file_size = file_path.stat().st_size
            
            print(f"   📁 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"   📄 Lines: {len([l for l in lines if l.strip()])}")
            print(f"   👀 Preview (first {max_lines} lines):")
            
            for i, line in enumerate(lines[:max_lines]):
                if line.strip():
                    print(f"      {i+1:2d} | {line}")
            
            if len(lines) > max_lines:
                print(f"      ... ({len(lines) - max_lines} more lines)")
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    def analyze_format_issues(self, filename: str) -> Dict:
        """Analyze and describe format issues in a file."""
        file_path = self.test_data_dir / filename
        
        if not file_path.exists():
            return {'error': 'File not found'}
        
        issues = []
        
        # Analyze filename for clues
        if 'european' in filename:
            issues.append('European CSV format (semicolon delimiter, comma decimal)')
        if 'tab' in filename:
            issues.append('Tab-delimited instead of comma-separated')
        if 'no_header' in filename:
            issues.append('Missing column headers')
        if 'extra_column' in filename:
            issues.append('Additional metadata columns')
        if 'wrong_order' in filename:
            issues.append('Incorrect wavenumber ordering')
        if 'mixed' in filename:
            issues.append('Multiple format problems combined')
        if 'scale' in filename:
            issues.append('Incorrect magnitude scaling')
        if 'encoding' in filename:
            issues.append('Non-UTF-8 character encoding')
        if 'noisy' in filename:
            issues.append('Data quality issues (noise, spikes)')
        if 'raman' in filename:
            issues.append('Different spectroscopy type')
        if 'performance' in filename:
            issues.append('Large dataset for performance testing')
        
        return {'issues': issues}
    
    def demonstrate_category(self, category_key: str) -> None:
        """Demonstrate files in a specific category."""
        category = self.file_categories[category_key]
        
        print(f"\n{category['title']}")
        print("=" * 60)
        
        for filename, description in category['files']:
            print(f"\n📄 {filename}")
            print(f"   📝 Description: {description}")
            
            # Analyze format issues
            analysis = self.analyze_format_issues(filename)
            if 'issues' in analysis and analysis['issues']:
                print(f"   🔧 Format Issues:")
                for issue in analysis['issues']:
                    print(f"      • {issue}")
            
            # Display file preview
            self.display_file_preview(filename)
            
            print()
    
    def run_interactive_demo(self) -> None:
        """Run interactive demonstration."""
        print("🧪 Spectral Analysis Test Dataset Demonstration")
        print("=" * 60)
        print("This demo showcases the comprehensive test dataset created for")
        print("demonstrating AI normalization capabilities and system robustness.")
        print()
        
        while True:
            print("Available demonstrations:")
            print("1. 🟢 Baseline Files (Perfect References)")
            print("2. 🔴 Format Issues (Problematic Files)")
            print("3. 🔬 Spectroscopy Types (Different Instruments)")
            print("4. ⚡ Performance Testing (Large Files)")
            print("5. 📊 Complete Overview (All Files)")
            print("6. 🚀 Quick Demo (Highlights)")
            print("0. Exit")
            print()
            
            try:
                choice = input("Select demonstration (0-6): ").strip()
                
                if choice == '0':
                    print("👋 Demo completed!")
                    break
                elif choice == '1':
                    self.demonstrate_category('baseline')
                elif choice == '2':
                    self.demonstrate_category('format_issues')
                elif choice == '3':
                    self.demonstrate_category('spectroscopy_types')
                elif choice == '4':
                    self.demonstrate_category('performance')
                elif choice == '5':
                    self.demonstrate_all_files()
                elif choice == '6':
                    self.run_quick_demo()
                else:
                    print("❌ Invalid choice. Please select 0-6.")
                
                if choice != '0':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def demonstrate_all_files(self) -> None:
        """Demonstrate all files in the dataset."""
        print("\n📊 Complete Test Dataset Overview")
        print("=" * 60)
        
        for category_key in self.file_categories:
            self.demonstrate_category(category_key)
        
        # Show statistics
        total_files = sum(len(cat['files']) for cat in self.file_categories.values())
        total_size = sum(
            (self.test_data_dir / filename).stat().st_size 
            for cat in self.file_categories.values() 
            for filename, _ in cat['files']
            if (self.test_data_dir / filename).exists()
        )
        
        print(f"\n📈 Dataset Statistics:")
        print(f"   📁 Total files: {total_files}")
        print(f"   💾 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        print(f"   🎯 Coverage: Comprehensive format variation testing")
    
    def run_quick_demo(self) -> None:
        """Run a quick demonstration of key highlights."""
        print("\n🚀 Quick Demo - Test Dataset Highlights")
        print("=" * 60)
        
        highlights = [
            ('baseline_perfect.csv', 'Perfect format - no issues'),
            ('sample_mixed_issues.csv', 'Multiple format problems'),
            ('sample_extra_columns.csv', 'Extra metadata columns'),
            ('performance_large_file.csv', 'Large dataset (10K points)')
        ]
        
        for filename, description in highlights:
            print(f"\n⭐ {filename}")
            print(f"   📝 {description}")
            
            # Quick analysis
            analysis = self.analyze_format_issues(filename)
            if 'issues' in analysis and analysis['issues']:
                print(f"   🔧 Key Issues: {', '.join(analysis['issues'][:2])}")
            
            # File stats
            file_path = self.test_data_dir / filename
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"   📊 Size: {size:,} bytes")
        
        print(f"\n✨ This dataset demonstrates the application's ability to handle:")
        print(f"   • European CSV formats (semicolon, comma decimal)")
        print(f"   • Tab-delimited files with metadata")
        print(f"   • Files without headers")
        print(f"   • Multiple extra columns")
        print(f"   • Wrong data ordering")
        print(f"   • Scale and unit issues")
        print(f"   • Different spectroscopy types")
        print(f"   • Noisy and corrupted data")
        print(f"   • Large performance datasets")
        print(f"   • Various character encodings")
    
    def run_automated_demo(self) -> None:
        """Run automated demonstration without user interaction."""
        print("🤖 Automated Test Dataset Demonstration")
        print("=" * 60)
        
        for category_key in self.file_categories:
            self.demonstrate_category(category_key)
            time.sleep(1)  # Brief pause between categories
        
        self.demonstrate_all_files()


def main():
    """Main demo function."""
    demo = TestDatasetDemo()
    
    # Check if running interactively
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        demo.run_automated_demo()
    else:
        demo.run_interactive_demo()


if __name__ == "__main__":
    main()
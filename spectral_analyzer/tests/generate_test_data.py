#!/usr/bin/env python3
"""
Comprehensive Test Data Generator for Spectral Analysis Application

This script generates realistic spectroscopic test data with various format issues
to demonstrate the AI normalization capabilities and robustness of the system.
"""

import os
import csv
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import json


class SpectralTestDataGenerator:
    """Generate comprehensive test datasets for spectral analysis application."""
    
    def __init__(self, output_dir: str = "test_data"):
        """Initialize the test data generator.
        
        Args:
            output_dir: Directory to save generated test files
        """
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # Spectroscopic parameters
        self.wavenumber_ranges = {
            'ftir': (400, 4000),
            'raman': (200, 3500),
            'nir': (4000, 12000),
            'uv_vis': (200, 800)  # nm for UV-Vis
        }
        
        # Common spectroscopic peaks (wavenumber, relative intensity)
        self.common_peaks = {
            'ftir': [
                (3300, 0.8),  # O-H stretch
                (2950, 0.6),  # C-H stretch
                (1650, 0.9),  # C=O stretch
                (1450, 0.5),  # C-H bend
                (1050, 0.7),  # C-O stretch
                (800, 0.4),   # C-H out-of-plane
            ],
            'raman': [
                (3000, 0.7),  # C-H stretch
                (1600, 0.9),  # C=C stretch
                (1000, 0.6),  # C-C stretch
                (500, 0.5),   # C-C-C bend
            ],
            'nir': [
                (7000, 0.6),  # O-H overtone
                (5800, 0.4),  # C-H overtone
                (4300, 0.5),  # C-H combination
            ]
        }
        
        # Instrument manufacturers and their typical formats
        self.instruments = {
            'bruker': {'delimiter': ',', 'decimal': '.', 'encoding': 'utf-8'},
            'thermo': {'delimiter': '\t', 'decimal': '.', 'encoding': 'utf-8'},
            'agilent': {'delimiter': ',', 'decimal': '.', 'encoding': 'latin-1'},
            'perkinelmer': {'delimiter': ';', 'decimal': ',', 'encoding': 'utf-8'},
            'shimadzu': {'delimiter': ',', 'decimal': '.', 'encoding': 'shift_jis'}
        }
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def generate_realistic_spectrum(self, 
                                  spectrum_type: str = 'ftir',
                                  num_points: int = 1500,
                                  noise_level: float = 0.02,
                                  baseline_drift: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic spectroscopic data with peaks, noise, and baseline.
        
        Args:
            spectrum_type: Type of spectroscopy ('ftir', 'raman', 'nir', 'uv_vis')
            num_points: Number of data points
            noise_level: Amount of random noise to add
            baseline_drift: Amount of baseline drift
            
        Returns:
            Tuple of (wavenumbers, absorbance_values)
        """
        # Get wavenumber range for spectrum type
        wn_min, wn_max = self.wavenumber_ranges.get(spectrum_type, (400, 4000))
        
        # Generate wavenumber array (descending for IR, ascending for others)
        if spectrum_type in ['ftir', 'nir']:
            wavenumbers = np.linspace(wn_max, wn_min, num_points)
        else:
            wavenumbers = np.linspace(wn_min, wn_max, num_points)
        
        # Initialize absorbance with baseline
        absorbance = np.zeros(num_points)
        
        # Add baseline drift (polynomial)
        x_norm = (wavenumbers - wavenumbers.min()) / (wavenumbers.max() - wavenumbers.min())
        baseline = baseline_drift * (0.1 * x_norm**2 + 0.05 * x_norm + 0.02)
        absorbance += baseline
        
        # Add spectroscopic peaks
        peaks = self.common_peaks.get(spectrum_type, [])
        for peak_wn, intensity in peaks:
            if wn_min <= peak_wn <= wn_max:
                # Gaussian peak
                width = 20 + random.uniform(-10, 10)  # Peak width variation
                peak_intensity = intensity * random.uniform(0.7, 1.3)  # Intensity variation
                
                gaussian = peak_intensity * np.exp(-((wavenumbers - peak_wn) / width)**2)
                absorbance += gaussian
        
        # Add random noise
        noise = np.random.normal(0, noise_level, num_points)
        absorbance += noise
        
        # Ensure non-negative values
        absorbance = np.maximum(absorbance, 0.001)
        
        return wavenumbers, absorbance
    
    def format_number(self, value: float, decimal_sep: str = '.') -> str:
        """Format number with specified decimal separator."""
        formatted = f"{value:.3f}"
        if decimal_sep != '.':
            formatted = formatted.replace('.', decimal_sep)
        return formatted
    
    def generate_standard_spectrum(self, name: str, spectrum_type: str = 'ftir') -> str:
        """Generate clean, standard format spectrum file.
        
        Args:
            name: Base filename (without extension)
            spectrum_type: Type of spectroscopy
            
        Returns:
            Path to generated file
        """
        wavenumbers, absorbance = self.generate_realistic_spectrum(spectrum_type)
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Wavenumber', 'Absorbance'])
            
            for wn, abs_val in zip(wavenumbers, absorbance):
                writer.writerow([f"{wn:.1f}", f"{abs_val:.6f}"])
        
        return filepath
    
    def generate_european_format(self, name: str) -> str:
        """Generate European format file (semicolon delimiter, comma decimal)."""
        wavenumbers, absorbance = self.generate_realistic_spectrum('ftir')
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # European format headers
            f.write("# FTIR Spektrometer Daten\n")
            f.write("# Datum: 2024-01-15\n")
            f.write("# Operator: Dr. Schmidt\n")
            f.write("# Instrument: Bruker FTIR Alpha\n")
            f.write("Wellenzahl (cm-1);Absorption\n")
            
            for wn, abs_val in zip(wavenumbers, absorbance):
                wn_str = self.format_number(wn, ',')
                abs_str = self.format_number(abs_val, ',')
                f.write(f"{wn_str};{abs_str}\n")
        
        return filepath
    
    def generate_tab_delimited(self, name: str) -> str:
        """Generate tab-delimited file with extra columns."""
        wavenumbers, absorbance = self.generate_realistic_spectrum('ftir')
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Comment headers
            f.write("// Instrument: Thermo Nicolet iS50\n")
            f.write("// Operator: Lab Technician\n")
            f.write("// Date: 2024-01-20\n")
            f.write("// Sample: Unknown polymer\n")
            
            # Tab-delimited headers
            f.write("Wavenumber\tAbsorbance\tSample_ID\tQuality_Flag\tTemperature\n")
            
            sample_id = "POLY_001"
            for i, (wn, abs_val) in enumerate(zip(wavenumbers, absorbance)):
                quality = random.choice(['Good', 'Fair', 'Excellent'])
                temp = 23.5 + random.uniform(-1, 1)
                f.write(f"{wn:.1f}\t{abs_val:.6f}\t{sample_id}\t{quality}\t{temp:.1f}\n")
        
        return filepath
    
    def generate_no_headers(self, name: str) -> str:
        """Generate file with no column headers."""
        wavenumbers, absorbance = self.generate_realistic_spectrum('ftir', num_points=800)
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            for wn, abs_val in zip(wavenumbers, absorbance):
                f.write(f"{wn:.1f},{abs_val:.6f}\n")
        
        return filepath
    
    def generate_wrong_order(self, name: str) -> str:
        """Generate file with ascending wavenumbers (wrong for FTIR)."""
        wavenumbers, absorbance = self.generate_realistic_spectrum('ftir')
        
        # Reverse to make ascending
        wavenumbers = wavenumbers[::-1]
        absorbance = absorbance[::-1]
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Wavenumber', 'Absorbance'])
            
            for wn, abs_val in zip(wavenumbers, absorbance):
                writer.writerow([f"{wn:.1f}", f"{abs_val:.6f}"])
        
        return filepath
    
    def generate_extra_columns(self, name: str) -> str:
        """Generate file with many extra metadata columns."""
        wavenumbers, absorbance = self.generate_realistic_spectrum('ftir')
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ['Wavenumber', 'Absorbance', 'Timestamp', 'Operator', 
                      'Quality_Flag', 'Temperature', 'Humidity', 'Pressure', 
                      'Scan_Number', 'Resolution', 'Apodization']
            writer.writerow(headers)
            
            base_time = datetime(2024, 1, 15, 10, 30, 0)
            
            for i, (wn, abs_val) in enumerate(zip(wavenumbers, absorbance)):
                timestamp = base_time + timedelta(seconds=i*0.1)
                row = [
                    f"{wn:.1f}",
                    f"{abs_val:.6f}",
                    timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Dr. Johnson",
                    random.choice(['GOOD', 'FAIR', 'EXCELLENT']),
                    f"{23.5 + random.uniform(-1, 1):.1f}",
                    f"{45 + random.uniform(-5, 5):.1f}",
                    f"{1013.25 + random.uniform(-2, 2):.2f}",
                    str(i + 1),
                    "4.0",
                    "Happ-Genzel"
                ]
                writer.writerow(row)
        
        return filepath
    
    def generate_mixed_issues(self, name: str) -> str:
        """Generate file with multiple format problems."""
        wavenumbers, absorbance = self.generate_realistic_spectrum('ftir')
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Mixed comment styles
            f.write("# FTIR Analysis - Sample XYZ\n")
            f.write("// Date: 15/01/2024\n")
            f.write("* Operator: Dr. Martinez\n")
            f.write("% Comments: High quality measurement\n")
            f.write("! Resolution: 4 cm-1\n")
            f.write("\n")  # Empty line
            
            # European-style headers with mixed delimiters
            f.write("Wave Number (cm-1);Transmittance %;Operator;Notes\n")
            
            # Convert absorbance to transmittance and use European format
            for wn, abs_val in zip(wavenumbers, absorbance):
                transmittance = 100 * (10 ** (-abs_val))  # Convert to %T
                wn_str = self.format_number(wn, ',')
                trans_str = self.format_number(transmittance, ',')
                notes = random.choice(['baseline', 'peak', 'good', 'noise', ''])
                f.write(f"{wn_str};{trans_str};Dr. Martinez;{notes}\n")
        
        return filepath
    
    def generate_scale_issues(self, name: str) -> str:
        """Generate file with wrong magnitude values."""
        wavenumbers, absorbance = self.generate_realistic_spectrum('ftir')
        
        # Scale issues: multiply by 1000 (common mistake)
        absorbance_scaled = absorbance * 1000
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Wavenumber', 'Absorbance_x1000'])
            
            for wn, abs_val in zip(wavenumbers, absorbance_scaled):
                writer.writerow([f"{wn:.1f}", f"{abs_val:.3f}"])
        
        return filepath
    
    def generate_raman_spectrum(self, name: str) -> str:
        """Generate Raman spectrum with different characteristics."""
        wavenumbers, intensity = self.generate_realistic_spectrum('raman', num_points=1200)
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write("# Raman Spectroscopy Data\n")
            f.write("# Instrument: Horiba LabRAM HR Evolution\n")
            f.write("# Laser: 532 nm\n")
            f.write("# Power: 1 mW\n")
            f.write("# Integration time: 10 s\n")
            
            writer = csv.writer(f)
            writer.writerow(['Raman_Shift_cm-1', 'Intensity_counts'])
            
            for wn, intens in zip(wavenumbers, intensity):
                # Raman intensities are typically much higher
                intens_scaled = intens * 10000
                writer.writerow([f"{wn:.1f}", f"{intens_scaled:.0f}"])
        
        return filepath
    
    def generate_noisy_data(self, name: str) -> str:
        """Generate spectrum with realistic noise and artifacts."""
        wavenumbers, absorbance = self.generate_realistic_spectrum(
            'ftir', noise_level=0.05, baseline_drift=0.2
        )
        
        # Add spikes (cosmic rays simulation)
        num_spikes = random.randint(3, 8)
        spike_indices = random.sample(range(len(absorbance)), num_spikes)
        for idx in spike_indices:
            absorbance[idx] += random.uniform(0.5, 2.0)
        
        # Add water vapor lines (atmospheric interference)
        water_lines = [3756, 3652, 1595, 1470]
        for line in water_lines:
            if wavenumbers.min() <= line <= wavenumbers.max():
                idx = np.argmin(np.abs(wavenumbers - line))
                absorbance[idx:idx+3] += 0.1
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write("# Noisy FTIR Data - Atmospheric Interference Present\n")
            f.write("# WARNING: Contains cosmic ray spikes\n")
            f.write("# NOTE: Water vapor contamination detected\n")
            
            writer = csv.writer(f)
            writer.writerow(['Wavenumber', 'Absorbance'])
            
            for wn, abs_val in zip(wavenumbers, absorbance):
                writer.writerow([f"{wn:.1f}", f"{abs_val:.6f}"])
        
        return filepath
    
    def generate_large_file(self, name: str) -> str:
        """Generate large file for performance testing."""
        wavenumbers, absorbance = self.generate_realistic_spectrum(
            'ftir', num_points=10000  # Much larger dataset
        )
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write("# Large Dataset - Performance Test\n")
            f.write("# High resolution: 0.5 cm-1\n")
            f.write("# Total points: 10000\n")
            
            writer = csv.writer(f)
            writer.writerow(['Wavenumber', 'Absorbance'])
            
            for wn, abs_val in zip(wavenumbers, absorbance):
                writer.writerow([f"{wn:.2f}", f"{abs_val:.8f}"])
        
        return filepath
    
    def generate_encoding_test(self, name: str) -> str:
        """Generate file with Latin-1 encoding and special characters."""
        wavenumbers, absorbance = self.generate_realistic_spectrum('ftir')
        
        filepath = os.path.join(self.output_dir, f"{name}.csv")
        
        with open(filepath, 'w', newline='', encoding='latin-1') as f:
            f.write("# Donnees FTIR - Echantillon special\n")
            f.write("# Operateur: Francois Muller\n")
            f.write("# Temperature: 25 degC\n")
            f.write("# Resolution: 4 cm-1\n")
            
            writer = csv.writer(f)
            writer.writerow(['Nombre_d_onde', 'Absorbance'])
            
            for wn, abs_val in zip(wavenumbers, absorbance):
                writer.writerow([f"{wn:.1f}", f"{abs_val:.6f}"])
        
        return filepath
    
    def create_complete_test_suite(self) -> Dict[str, str]:
        """Generate all test files and return mapping of names to paths."""
        print("Generating comprehensive test dataset...")
        
        files_created = {}
        
        # Baseline files (perfect formats)
        print("Creating baseline files...")
        files_created['baseline_perfect'] = self.generate_standard_spectrum('baseline_perfect', 'ftir')
        files_created['baseline_european'] = self.generate_european_format('baseline_european')
        files_created['baseline_raman'] = self.generate_standard_spectrum('baseline_raman', 'raman')
        
        # Sample files with various issues
        print("Creating problematic sample files...")
        files_created['sample_tab_delimited'] = self.generate_tab_delimited('sample_tab_delimited')
        files_created['sample_no_headers'] = self.generate_no_headers('sample_no_headers')
        files_created['sample_extra_columns'] = self.generate_extra_columns('sample_extra_columns')
        files_created['sample_wrong_order'] = self.generate_wrong_order('sample_wrong_order')
        files_created['sample_mixed_issues'] = self.generate_mixed_issues('sample_mixed_issues')
        files_created['sample_scale_issues'] = self.generate_scale_issues('sample_scale_issues')
        files_created['sample_raman_spectrum'] = self.generate_raman_spectrum('sample_raman_spectrum')
        files_created['sample_noisy_data'] = self.generate_noisy_data('sample_noisy_data')
        files_created['sample_encoding_test'] = self.generate_encoding_test('sample_encoding_test')
        
        # Performance test files
        print("Creating performance test files...")
        files_created['performance_large_file'] = self.generate_large_file('performance_large_file')
        
        print(f"Generated {len(files_created)} test files in {self.output_dir}/")
        return files_created


def main():
    """Main function to generate all test data."""
    generator = SpectralTestDataGenerator()
    files_created = generator.create_complete_test_suite()
    
    print("\nTest files created:")
    for name, path in files_created.items():
        print(f"  {name}: {path}")
    
    print(f"\nAll files saved to: {generator.output_dir}/")
    print("Test dataset generation complete!")


if __name__ == "__main__":
    main()
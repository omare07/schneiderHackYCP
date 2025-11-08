"""
Generate extremely challenging demo CSV files that require AI/LLM semantic understanding.

These files are designed to be impossible to fix with simple algorithmic parsing,
requiring domain knowledge, context interpretation, and semantic reasoning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random


def generate_realistic_spectrum(wavenumbers, spectrum_type='ftir'):
    """Generate realistic spectral data with peaks"""
    # Base spectrum
    absorbance = np.random.normal(0.01, 0.002, len(wavenumbers))
    
    if spectrum_type == 'ftir':
        # Add typical FTIR peaks for polymer
        peaks = [
            (2920, 0.8, 50),   # C-H stretch
            (2850, 0.6, 40),   # C-H stretch
            (1730, 0.9, 60),   # C=O stretch
            (1450, 0.5, 80),   # C-H bend
            (1370, 0.4, 50),   # C-H bend
            (1240, 0.7, 70),   # C-O stretch
            (1160, 0.6, 60),   # C-O stretch
            (700, 0.5, 100),   # C-H wag
        ]
        for center, height, width in peaks:
            peak = height * np.exp(-((wavenumbers - center) ** 2) / (2 * width ** 2))
            absorbance += peak
    
    return absorbance


def generate_german_headers(output_dir, wavenumbers):
    """
    Generate CSV with German headers and transmittance data.
    
    AI Must:
    - Recognize German: "Wellenzahl"=Wavenumber, "Durchlässigkeit"=Transmittance
    - Convert Transmittance % → Absorbance using A = -log10(T/100)
    - Remove irrelevant columns (Sample ID, Temperature)
    - Understand superscript notation (cm⁻¹)
    """
    absorbance = generate_realistic_spectrum(wavenumbers, 'ftir')
    
    # Convert absorbance to transmittance %
    transmittance = 100 * (10 ** (-absorbance))
    
    # Create dataframe with German headers
    df = pd.DataFrame({
        'Wellenzahl (cm⁻¹)': wavenumbers,
        'Durchlässigkeit (%)': transmittance,
        'Probennummer': ['GS-001'] * len(wavenumbers),
        'Temperatur (°C)': np.random.normal(22.5, 0.2, len(wavenumbers))
    })
    
    # Write with German comments
    output_file = output_dir / 'demo_german_headers.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# FTIR Spektrum Analyse\n')
        f.write('# Datum: 2024-01-15\n')
        f.write('# Messgerät: Bruker Tensor 27\n')
        df.to_csv(f, index=False)
    
    print(f"✓ Generated {output_file.name}")
    print(f"  Challenge: German headers, Transmittance→Absorbance conversion needed")


def generate_units_in_cells(output_dir, wavenumbers):
    """
    Generate CSV with units embedded in cell values.
    
    AI Must:
    - Extract numbers from strings with units
    - Normalize different unit notations (a.u., arb. units, AU)
    - Ignore text annotations
    - Understand "/" notation for wavenumber
    """
    absorbance = generate_realistic_spectrum(wavenumbers, 'ftir')
    
    # Various unit formats
    unit_formats = [
        'a.u.', 'arb. units', 'AU', 'arbitrary units', 'a.u', 'arb.u.'
    ]
    
    wavenumber_formats = [
        lambda x: f'"{x:.1f} cm-1"',
        lambda x: f'"{x:.1f} /cm"',
        lambda x: f'{x:.1f} cm⁻¹',
        lambda x: f'{x:.1f}cm-1',
    ]
    
    # Build rows with mixed formats
    rows = []
    for i, (wn, ab) in enumerate(zip(wavenumbers, absorbance)):
        wn_format = random.choice(wavenumber_formats)
        unit_format = random.choice(unit_formats)
        
        note = ''
        if ab > 0.5:
            note = f'peak at {wn:.0f}'
        elif ab > 0.3:
            note = 'shoulder'
        elif i % 100 == 0:
            note = 'baseline'
        
        rows.append({
            'Shift': wn_format(wn),
            'Intensity': f'"{ab:.3f} {unit_format}"',
            'Notes': note
        })
    
    df = pd.DataFrame(rows)
    
    output_file = output_dir / 'demo_units_in_cells.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# Raman Spectrum with inline units\n')
        f.write('# All intensity values in arbitrary units\n')
        df.to_csv(f, index=False)
    
    print(f"✓ Generated {output_file.name}")
    print(f"  Challenge: Units embedded in values, multiple notations")


def generate_inline_metadata(output_dir, wavenumbers):
    """
    Generate CSV with metadata scattered throughout data rows.
    
    AI Must:
    - Identify data vs metadata rows
    - Extract only spectral numerical data
    - Understand context that comments aren't column headers
    """
    absorbance = generate_realistic_spectrum(wavenumbers, 'ftir')
    
    output_file = output_dir / 'demo_inline_metadata.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Wavenumber,Absorbance\n')
        
        metadata_lines = [
            '# Instrument: Bruker FTIR',
            '# Resolution: 4cm-1',
            'Sample: Polystyrene Film',
            '# Scans averaged: 32',
            '# Detector: DTGS',
            'Operator: Dr. Sarah Johnson',
            '# Beam splitter: KBr',
            '# Date: 2024-01-15',
            'Notes: High quality scan',
            '# Atmosphere: Dry air purge',
        ]
        
        metadata_interval = len(wavenumbers) // len(metadata_lines)
        metadata_idx = 0
        
        for i, (wn, ab) in enumerate(zip(wavenumbers, absorbance)):
            # Insert metadata periodically
            if i > 0 and i % metadata_interval == 0 and metadata_idx < len(metadata_lines):
                f.write(f'{metadata_lines[metadata_idx]}\n')
                metadata_idx += 1
            
            f.write(f'{wn:.1f},{ab:.4f}\n')
    
    print(f"✓ Generated {output_file.name}")
    print(f"  Challenge: Metadata scattered in data, context understanding needed")


def generate_scientific_mixed(output_dir, wavenumbers):
    """
    Generate CSV with mixed scientific notation formats.
    
    AI Must:
    - Parse multiple scientific notation formats (E, e, *, 10^)
    - Recognize X=Wavenumber, Y=Absorbance from context
    - Remove extraneous columns
    - Normalize to standard decimal format
    """
    absorbance = generate_realistic_spectrum(wavenumbers, 'ftir')
    
    # Different scientific notation formats
    def format_scientific(value, style):
        if style == 0:
            return f"{value:.1E}"  # 4.0E+03
        elif style == 1:
            return f"{value:.3e}"  # 4.000e+03
        elif style == 2:
            # Convert to mantissa * 10^exponent format
            exp = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
            mantissa = value / (10 ** exp)
            return f"{mantissa:.3f}*10^{exp}"
        else:
            # Standard E notation with varying decimal places
            return f"{value:.2E}"
    
    rows = []
    for wn, ab in zip(wavenumbers, absorbance):
        wn_style = random.randint(0, 3)
        ab_style = random.randint(0, 3)
        
        rows.append({
            'X': format_scientific(wn, wn_style),
            'Y': format_scientific(ab, ab_style),
            'Sample_Type': 'polymer',
            'Operator': random.choice(['J. Smith', 'M. Chen', 'A. Patel'])
        })
    
    df = pd.DataFrame(rows)
    
    output_file = output_dir / 'demo_scientific_mixed.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# Spectral data export\n')
        f.write('# X axis: Wavenumber, Y axis: Absorbance\n')
        df.to_csv(f, index=False)
    
    print(f"✓ Generated {output_file.name}")
    print(f"  Challenge: Multiple scientific notations, cryptic column names")


def generate_cryptic_headers(output_dir, wavenumbers):
    """
    Generate CSV with cryptic abbreviations.
    
    AI Must:
    - Decode: WN=Wavenumber, Abs=Absorbance, Tx=Transmittance, Refl=Reflectance
    - Choose correct data columns (ignore SNR, BG, Comment)
    - Handle N/A values
    - Convert Transmittance to Absorbance when Absorbance is N/A
    - Understand domain knowledge of spectroscopy
    """
    absorbance = generate_realistic_spectrum(wavenumbers, 'ftir')
    transmittance = 100 * (10 ** (-absorbance))
    
    rows = []
    for i, (wn, ab, tx) in enumerate(zip(wavenumbers, absorbance, transmittance)):
        # Randomly use either Abs or Tx data, not both
        use_abs = random.random() > 0.3
        
        comment = ''
        if i == 0:
            comment = 'ref scan'
        elif ab > 0.7:
            comment = 'strong peak'
        elif i % 200 == 0:
            comment = 'checked baseline'
        
        rows.append({
            'WN': f"{wn:.1f}",
            'Abs.': f"{ab:.4f}" if use_abs else 'N/A',
            'Tx%': 'N/A' if use_abs else f"{tx:.2f}",
            'Refl': 'N/A',  # Not measured
            'SNR': f"{random.uniform(40, 50):.1f}",
            'BG': f"{random.uniform(0.0005, 0.002):.4f}",
            'Comment': comment
        })
    
    df = pd.DataFrame(rows)
    
    output_file = output_dir / 'demo_cryptic_headers.csv'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# Lab notebook export\n')
        f.write('# WN=Wavenumber(cm-1), Abs=Absorbance, Tx=Transmittance(%)\n')
        f.write('# Refl=Reflectance, SNR=Signal-to-Noise, BG=Background\n')
        df.to_csv(f, index=False)
    
    print(f"✓ Generated {output_file.name}")
    print(f"  Challenge: Cryptic abbreviations, N/A handling, Tx→Abs conversion")


def generate_all_demos():
    """Generate all extremely challenging demo files"""
    output_dir = Path(__file__).parent / "test_data"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Extreme Demo Data - AI Understanding Required")
    print("="*60 + "\n")
    
    # Generate 600 data points for realistic spectra
    wavenumbers = np.linspace(4000, 600, 600)
    
    # Generate all 5 challenging files
    generate_german_headers(output_dir, wavenumbers)
    generate_units_in_cells(output_dir, wavenumbers)
    generate_inline_metadata(output_dir, wavenumbers)
    generate_scientific_mixed(output_dir, wavenumbers)
    generate_cryptic_headers(output_dir, wavenumbers)
    
    print("\n" + "="*60)
    print("✅ All extreme demo files generated successfully!")
    print("="*60)
    print("\nThese files require AI to:")
    print("  • Understand foreign languages (German)")
    print("  • Parse units embedded in data")
    print("  • Distinguish metadata from data")
    print("  • Handle multiple scientific notations")
    print("  • Decode cryptic abbreviations")
    print("  • Apply domain knowledge (spectroscopy)")
    print("  • Perform semantic transformations (Tx→Abs)")
    print("\n❌ Cannot be solved with simple regex or parsing!\n")


if __name__ == '__main__':
    generate_all_demos()
# Spectral Analysis Test Dataset

This directory contains a comprehensive collection of test data files designed to demonstrate the robustness and AI normalization capabilities of the spectral analysis application. The dataset includes various CSV format challenges commonly encountered in real-world laboratory environments.

## Overview

The test dataset consists of **19 files** covering:
- **3 Baseline files** (perfect reference formats)
- **10 Sample files** (various format issues)
- **6 Legacy files** (existing test data)
- **1 Performance file** (large dataset testing)

## File Categories

### 🟢 Baseline Files (Perfect References)

#### [`baseline_perfect.csv`](baseline_perfect.csv)
- **Format**: Standard CSV with comma delimiter
- **Headers**: `Wavenumber,Absorbance`
- **Data Points**: 1,500
- **Range**: 4000-400 cm⁻¹ (descending)
- **Features**: Clean FTIR spectrum with realistic peaks at 3300, 2950, 1650, 1450, 1050, 800 cm⁻¹
- **Expected Normalization**: None required - perfect format
- **Use Case**: Control file for validation

#### [`baseline_european.csv`](baseline_european.csv)
- **Format**: European CSV (semicolon delimiter, comma decimal separator)
- **Headers**: German language (`Wellenzahl (cm-1);Absorption`)
- **Metadata**: Comment lines with instrument info
- **Expected Normalization**: 
  - Convert semicolon to comma delimiter
  - Convert comma decimal separator to period
  - Remove metadata headers
  - Translate headers to English

#### [`baseline_raman.csv`](baseline_raman.csv)
- **Format**: Standard CSV for Raman spectroscopy
- **Range**: 3500-200 cm⁻¹ (ascending)
- **Features**: Raman-specific peaks at 3000, 1600, 1000, 500 cm⁻¹
- **Expected Normalization**: None required for format, but different spectroscopy type

### 🔴 Sample Files (Format Issues)

#### [`sample_tab_delimited.csv`](sample_tab_delimited.csv)
- **Issue**: Tab-separated values instead of commas
- **Headers**: Comment lines with `//` prefix
- **Extra Columns**: `Sample_ID`, `Quality_Flag`, `Temperature`
- **Expected Normalization**:
  - Convert tabs to commas
  - Remove comment headers
  - Extract only Wavenumber and Absorbance columns

#### [`sample_no_headers.csv`](sample_no_headers.csv)
- **Issue**: No column headers - data starts immediately
- **Data Points**: 800
- **Expected Normalization**:
  - Add standard headers (`Wavenumber,Absorbance`)
  - Detect column types automatically

#### [`sample_extra_columns.csv`](sample_extra_columns.csv)
- **Issue**: Many extra metadata columns
- **Columns**: `Wavenumber`, `Absorbance`, `Timestamp`, `Operator`, `Quality_Flag`, `Temperature`, `Humidity`, `Pressure`, `Scan_Number`, `Resolution`, `Apodization`
- **Data Points**: 1,500
- **Expected Normalization**:
  - Extract only Wavenumber and Absorbance columns
  - Preserve metadata in separate structure if needed

#### [`sample_wrong_order.csv`](sample_wrong_order.csv)
- **Issue**: Ascending wavenumbers (wrong for FTIR)
- **Expected Normalization**:
  - Reverse data order to descending wavenumbers
  - Maintain data integrity during reversal

#### [`sample_mixed_issues.csv`](sample_mixed_issues.csv)
- **Issues**: Multiple problems in one file
  - Mixed comment styles (`#`, `//`, `*`, `%`, `!`)
  - European format (semicolon delimiter, comma decimal)
  - Transmittance instead of Absorbance
  - Extra columns
- **Expected Normalization**:
  - Remove all comment lines
  - Convert delimiters and decimal separators
  - Convert Transmittance to Absorbance: `A = -log10(T/100)`
  - Extract core spectral data

#### [`sample_scale_issues.csv`](sample_scale_issues.csv)
- **Issue**: Absorbance values scaled by 1000x
- **Header**: `Absorbance_x1000` (indicates scaling)
- **Expected Normalization**:
  - Detect scale issue from header or value ranges
  - Divide absorbance values by 1000
  - Update header to standard format

#### [`sample_raman_spectrum.csv`](sample_raman_spectrum.csv)
- **Format**: Raman spectroscopy data
- **Headers**: `Raman_Shift_cm-1`, `Intensity_counts`
- **Metadata**: Instrument parameters in comments
- **Range**: 3500-200 cm⁻¹
- **Expected Normalization**:
  - Standardize headers to `Wavenumber,Absorbance`
  - Convert intensity to absorbance-like units
  - Remove metadata comments

#### [`sample_noisy_data.csv`](sample_noisy_data.csv)
- **Issues**: Realistic noise and artifacts
  - Cosmic ray spikes (random high values)
  - Water vapor interference lines
  - High baseline noise
- **Expected Normalization**:
  - Detect and flag anomalous spikes
  - Identify atmospheric interference
  - Suggest data cleaning recommendations

#### [`sample_encoding_test.csv`](sample_encoding_test.csv)
- **Issue**: Latin-1 encoding with special characters
- **Language**: French headers and metadata
- **Expected Normalization**:
  - Detect encoding automatically
  - Convert to UTF-8
  - Translate headers to English

### 📊 Performance Testing

#### [`performance_large_file.csv`](performance_large_file.csv)
- **Purpose**: Performance and memory testing
- **Data Points**: 10,000 (high resolution: 0.5 cm⁻¹)
- **Size**: ~800KB
- **Expected Behavior**:
  - Fast parsing and processing
  - Efficient memory usage
  - Responsive UI during processing

### 📁 Legacy Files (Existing)

#### [`sample_spectral.csv`](sample_spectral.csv)
- Original test file with basic format

#### [`european_format.csv`](european_format.csv)
- European format with German headers

#### [`multi_column.csv`](multi_column.csv)
- Multiple sample columns

#### [`no_header.csv`](no_header.csv)
- No column headers

#### [`problematic_data.csv`](problematic_data.csv)
- Various data issues

#### [`tab_delimited.csv`](tab_delimited.csv)
- Tab-separated format

## Spectroscopic Features

### Realistic Peak Positions

All generated spectra include realistic spectroscopic features:

**FTIR Peaks:**
- **3300 cm⁻¹**: O-H stretch (alcohols, water)
- **2950 cm⁻¹**: C-H stretch (alkanes)
- **1650 cm⁻¹**: C=O stretch (carbonyls)
- **1450 cm⁻¹**: C-H bend (alkanes)
- **1050 cm⁻¹**: C-O stretch (alcohols, ethers)
- **800 cm⁻¹**: C-H out-of-plane bend

**Raman Peaks:**
- **3000 cm⁻¹**: C-H stretch
- **1600 cm⁻¹**: C=C stretch
- **1000 cm⁻¹**: C-C stretch
- **500 cm⁻¹**: C-C-C bend

### Data Quality Features

- **Realistic Noise**: Gaussian noise (σ = 0.02-0.05)
- **Baseline Drift**: Polynomial baseline variation
- **Peak Widths**: 10-30 cm⁻¹ FWHM
- **Intensity Variations**: ±30% natural variation
- **Atmospheric Interference**: Water vapor lines in noisy data

## AI Normalization Test Scenarios

### Format Detection Tests

1. **Delimiter Detection**:
   - Comma vs semicolon vs tab
   - Mixed delimiters in same file

2. **Decimal Separator Detection**:
   - Period vs comma decimal separators
   - Automatic conversion

3. **Header Detection**:
   - Presence/absence of headers
   - Language detection (English/German/French)
   - Column mapping

4. **Encoding Detection**:
   - UTF-8 vs Latin-1 vs other encodings
   - Special character handling

### Data Transformation Tests

1. **Unit Conversion**:
   - Transmittance ↔ Absorbance
   - Scale factor detection and correction
   - Wavenumber order correction

2. **Column Extraction**:
   - Identify spectral data columns
   - Ignore metadata columns
   - Handle multiple sample columns

3. **Data Cleaning**:
   - Spike detection and removal
   - Baseline correction suggestions
   - Noise level assessment

### Confidence Scoring

Expected AI confidence levels for normalization:

- **Perfect files**: 95-100% confidence
- **Simple issues**: 85-95% confidence
- **Complex issues**: 70-85% confidence
- **Ambiguous cases**: 50-70% confidence

## Usage Examples

### Basic Testing

```python
from spectral_analyzer.core.csv_parser import CSVParser
from spectral_analyzer.core.ai_normalizer import AINormalizer

# Test perfect format
parser = CSVParser()
data = parser.parse_file("test_data/baseline_perfect.csv")
print(f"Loaded {len(data)} data points")

# Test problematic format
normalizer = AINormalizer()
result = normalizer.normalize_file("test_data/sample_mixed_issues.csv")
print(f"Normalization confidence: {result.confidence}%")
```

### Batch Testing

```python
import os
from pathlib import Path

test_files = Path("test_data").glob("*.csv")
results = {}

for file_path in test_files:
    try:
        result = normalizer.normalize_file(str(file_path))
        results[file_path.name] = {
            'success': True,
            'confidence': result.confidence,
            'transformations': result.transformations
        }
    except Exception as e:
        results[file_path.name] = {
            'success': False,
            'error': str(e)
        }

# Analyze results
success_rate = sum(1 for r in results.values() if r['success']) / len(results)
print(f"Success rate: {success_rate:.1%}")
```

## Validation Criteria

### Data Quality Checks

1. **Wavenumber Range**: 200-12000 cm⁻¹
2. **Absorbance Range**: 0.001-3.0 AU
3. **Data Density**: 500-10000 points
4. **Monotonic Order**: Descending for IR, ascending for Raman
5. **No Missing Values**: Complete data series

### Format Validation

1. **Delimiter Consistency**: Single delimiter type per file
2. **Decimal Format**: Consistent decimal separator
3. **Header Presence**: Detectable column headers
4. **Encoding**: Valid character encoding

### AI Normalization Success Metrics

1. **Format Detection Accuracy**: >95%
2. **Data Extraction Accuracy**: >98%
3. **Transformation Correctness**: >90%
4. **Processing Speed**: <2 seconds per file
5. **Memory Efficiency**: <100MB peak usage

## Integration with Application

### Testing Workflow

1. **Load Test File**: Use file drop zone or file dialog
2. **AI Analysis**: Automatic format detection and normalization
3. **Preview Results**: Display normalized data preview
4. **Generate Graph**: Create spectral plot
5. **Validate Output**: Compare with expected results

### Error Handling Tests

Test files include scenarios for:
- **Parsing Errors**: Malformed CSV structure
- **Encoding Issues**: Character encoding problems
- **Data Validation**: Out-of-range values
- **Memory Limits**: Large file handling
- **Network Issues**: AI service unavailability

## Expected Normalization Outcomes

### High Confidence (>90%)

- `baseline_perfect.csv`: No changes needed
- `baseline_european.csv`: Delimiter and decimal conversion
- `sample_tab_delimited.csv`: Tab to comma conversion
- `sample_no_headers.csv`: Add standard headers

### Medium Confidence (70-90%)

- `sample_extra_columns.csv`: Column extraction
- `sample_wrong_order.csv`: Data reversal
- `sample_scale_issues.csv`: Scale correction
- `sample_raman_spectrum.csv`: Header standardization

### Lower Confidence (50-70%)

- `sample_mixed_issues.csv`: Multiple transformations
- `sample_noisy_data.csv`: Data quality assessment
- `sample_encoding_test.csv`: Encoding and translation

## File Statistics

| File | Size | Points | Issues | Confidence |
|------|------|--------|---------|------------|
| baseline_perfect.csv | ~120KB | 1,500 | None | 100% |
| baseline_european.csv | ~125KB | 1,500 | Delimiter, Decimal | 95% |
| baseline_raman.csv | ~95KB | 1,200 | Spectroscopy Type | 90% |
| sample_tab_delimited.csv | ~140KB | 1,500 | Tabs, Comments | 90% |
| sample_no_headers.csv | ~65KB | 800 | No Headers | 85% |
| sample_extra_columns.csv | ~350KB | 1,500 | Extra Columns | 85% |
| sample_wrong_order.csv | ~120KB | 1,500 | Wrong Order | 80% |
| sample_mixed_issues.csv | ~180KB | 1,500 | Multiple Issues | 70% |
| sample_scale_issues.csv | ~120KB | 1,500 | Scale Factor | 75% |
| sample_raman_spectrum.csv | ~110KB | 1,200 | Different Type | 80% |
| sample_noisy_data.csv | ~130KB | 1,500 | Noise, Spikes | 65% |
| sample_encoding_test.csv | ~120KB | 1,500 | Encoding | 75% |
| performance_large_file.csv | ~800KB | 10,000 | Size | 95% |

## Testing Recommendations

### Automated Testing

1. **Unit Tests**: Test each file individually
2. **Integration Tests**: End-to-end workflow testing
3. **Performance Tests**: Large file processing
4. **Stress Tests**: Multiple files simultaneously
5. **Error Tests**: Corrupted file handling

### Manual Testing

1. **UI Testing**: Drag-and-drop functionality
2. **Preview Testing**: Data preview accuracy
3. **Graph Testing**: Visualization quality
4. **Export Testing**: Normalized data export

### Regression Testing

1. **Version Comparison**: Compare AI normalization results across versions
2. **Performance Monitoring**: Track processing times
3. **Accuracy Tracking**: Monitor normalization success rates

## Maintenance

### Updating Test Data

To regenerate the test dataset:

```bash
cd spectral_analyzer/tests
python3 generate_test_data.py
```

### Adding New Test Cases

1. Modify [`generate_test_data.py`](../generate_test_data.py)
2. Add new generation methods
3. Update this documentation
4. Run validation scripts

### Quality Assurance

- Review generated data for scientific accuracy
- Validate format issues are representative
- Ensure comprehensive coverage of edge cases
- Update expected outcomes documentation

---

*Generated by SpectralTestDataGenerator v1.0*  
*Last updated: 2024-01-15*
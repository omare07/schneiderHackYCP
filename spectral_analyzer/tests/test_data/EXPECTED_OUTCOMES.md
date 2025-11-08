# Expected Normalization Outcomes

This document details the expected AI normalization transformations and outcomes for each test file in the dataset.

## Confidence Level Guidelines

- **95-100%**: Perfect or near-perfect format, minimal changes needed
- **85-95%**: Simple format issues, clear transformation path
- **70-85%**: Moderate complexity, multiple transformations required
- **50-70%**: High complexity, ambiguous cases requiring interpretation
- **<50%**: Problematic files requiring manual intervention

## Baseline Files (Reference Standards)

### [`baseline_perfect.csv`](baseline_perfect.csv)
**Expected Confidence**: 100%  
**Expected Transformations**: None  
**Expected Outcome**: File should pass through unchanged

```
Input Format:
Wavenumber,Absorbance
4000.0,0.036552
3997.6,0.037450
...

Output Format: Identical
Status: No changes required
```

**Validation Criteria**:
- ✅ Standard CSV format
- ✅ Correct headers
- ✅ Descending wavenumber order
- ✅ Realistic absorbance values (0.001-1.0)
- ✅ 1,500 data points

### [`baseline_european.csv`](baseline_european.csv)
**Expected Confidence**: 90-95%  
**Expected Transformations**: 3
1. Remove comment headers (`# FTIR Spektrometer Daten`, etc.)
2. Convert semicolon delimiter to comma
3. Convert comma decimal separator to period
4. Translate German headers to English

```
Input Format:
# FTIR Spektrometer Daten
# Datum: 2024-01-15
Wellenzahl (cm-1);Absorption
4000,000;0,036552

Expected Output:
Wavenumber,Absorbance
4000.0,0.036552
```

**Validation Criteria**:
- ✅ Comments removed
- ✅ Delimiter converted (`;` → `,`)
- ✅ Decimal separator converted (`,` → `.`)
- ✅ Headers translated (`Wellenzahl` → `Wavenumber`)

### [`baseline_raman.csv`](baseline_raman.csv)
**Expected Confidence**: 85-90%  
**Expected Transformations**: 1
1. Standardize headers (`Raman_Shift_cm-1` → `Wavenumber`)

```
Input Format:
Wavenumber,Absorbance
3500.0,0.025199
200.0,0.001000

Expected Output: Minimal changes
Status: Different spectroscopy type detected
```

## Sample Files (Format Issues)

### [`sample_tab_delimited.csv`](sample_tab_delimited.csv)
**Expected Confidence**: 85-90%  
**Expected Transformations**: 3
1. Remove comment lines (`// Instrument: Thermo Nicolet iS50`)
2. Convert tab delimiter to comma
3. Extract only Wavenumber and Absorbance columns

```
Input Format:
// Instrument: Thermo Nicolet iS50
Wavenumber	Absorbance	Sample_ID	Quality_Flag	Temperature
4000.0	0.001000	POLY_001	Good	23.5

Expected Output:
Wavenumber,Absorbance
4000.0,0.001000
```

**Validation Criteria**:
- ✅ Comments removed
- ✅ Tabs converted to commas
- ✅ Extra columns removed
- ✅ Core spectral data preserved

### [`sample_no_headers.csv`](sample_no_headers.csv)
**Expected Confidence**: 80-85%  
**Expected Transformations**: 1
1. Add standard column headers

```
Input Format:
4000.0,0.001000
3997.6,0.018537

Expected Output:
Wavenumber,Absorbance
4000.0,0.001000
3997.6,0.018537
```

**Validation Criteria**:
- ✅ Headers added automatically
- ✅ Data types correctly identified
- ✅ No data loss

### [`sample_extra_columns.csv`](sample_extra_columns.csv)
**Expected Confidence**: 80-85%  
**Expected Transformations**: 1
1. Extract Wavenumber and Absorbance columns only

```
Input Format:
Wavenumber,Absorbance,Timestamp,Operator,Quality_Flag,Temperature,Humidity,Pressure,Scan_Number,Resolution,Apodization
4000.0,0.001000,2024-01-15 10:30:00.000,Dr. Johnson,FAIR,22.7,47.1,1012.66,1,4.0,Happ-Genzel

Expected Output:
Wavenumber,Absorbance
4000.0,0.001000
```

**Validation Criteria**:
- ✅ Core columns identified correctly
- ✅ Metadata columns removed
- ✅ Data integrity maintained

### [`sample_wrong_order.csv`](sample_wrong_order.csv)
**Expected Confidence**: 75-80%  
**Expected Transformations**: 1
1. Reverse data order (ascending → descending wavenumbers)

```
Input Format: (Ascending wavenumbers)
Wavenumber,Absorbance
400.0,0.003319
402.4,0.006384

Expected Output: (Descending wavenumbers)
Wavenumber,Absorbance
4000.0,0.036552
3997.6,0.037450
```

**Validation Criteria**:
- ✅ Data order reversed
- ✅ Wavenumber-absorbance pairs maintained
- ✅ No data corruption

### [`sample_mixed_issues.csv`](sample_mixed_issues.csv)
**Expected Confidence**: 65-75%  
**Expected Transformations**: 5
1. Remove mixed comment styles (`#`, `//`, `*`, `%`, `!`)
2. Convert semicolon delimiter to comma
3. Convert comma decimal separator to period
4. Convert Transmittance to Absorbance: `A = -log10(T/100)`
5. Extract core spectral columns

```
Input Format:
# FTIR Analysis - Sample XYZ
// Date: 15/01/2024
* Operator: Dr. Martinez
Wave Number (cm-1);Transmittance %;Operator;Notes
4000,000;94,103;Dr. Martinez;baseline

Expected Output:
Wavenumber,Absorbance
4000.0,0.027  # Calculated from transmittance
```

**Validation Criteria**:
- ✅ All comment styles removed
- ✅ European format converted
- ✅ Transmittance converted to absorbance
- ✅ Extra columns removed

### [`sample_scale_issues.csv`](sample_scale_issues.csv)
**Expected Confidence**: 70-80%  
**Expected Transformations**: 2
1. Detect scale factor from header (`Absorbance_x1000`)
2. Divide absorbance values by 1000

```
Input Format:
Wavenumber,Absorbance_x1000
4000.0,36.552  # Scaled by 1000

Expected Output:
Wavenumber,Absorbance
4000.0,0.036552  # Corrected scale
```

**Validation Criteria**:
- ✅ Scale factor detected from header
- ✅ Values divided by 1000
- ✅ Header standardized

### [`sample_raman_spectrum.csv`](sample_raman_spectrum.csv)
**Expected Confidence**: 75-85%  
**Expected Transformations**: 3
1. Remove instrument metadata comments
2. Standardize headers (`Raman_Shift_cm-1` → `Wavenumber`)
3. Convert intensity counts to normalized units

```
Input Format:
# Raman Spectroscopy Data
# Instrument: Horiba LabRAM HR Evolution
Raman_Shift_cm-1,Intensity_counts
3500.0,2520  # High count values

Expected Output:
Wavenumber,Absorbance
3500.0,0.252  # Normalized intensity
```

**Validation Criteria**:
- ✅ Metadata removed
- ✅ Headers standardized
- ✅ Intensity values normalized

### [`sample_noisy_data.csv`](sample_noisy_data.csv)
**Expected Confidence**: 60-70%  
**Expected Transformations**: 2
1. Remove warning comments
2. Flag data quality issues (spikes, water vapor)

```
Input Format:
# Noisy FTIR Data - Atmospheric Interference Present
# WARNING: Contains cosmic ray spikes
Wavenumber,Absorbance
4000.0,0.036552
3756.0,2.156552  # Cosmic ray spike

Expected Output:
Wavenumber,Absorbance
4000.0,0.036552
3756.0,0.156552  # Spike flagged/corrected
```

**Validation Criteria**:
- ✅ Comments removed
- ✅ Spikes detected and flagged
- ✅ Water vapor lines identified
- ⚠️ Data quality warnings provided

### [`sample_encoding_test.csv`](sample_encoding_test.csv)
**Expected Confidence**: 70-80%  
**Expected Transformations**: 3
1. Detect Latin-1 encoding
2. Convert to UTF-8
3. Translate French headers to English

```
Input Format: (Latin-1 encoding)
# Donnees FTIR - Echantillon special
Nombre_d_onde,Absorbance

Expected Output: (UTF-8 encoding)
Wavenumber,Absorbance
```

**Validation Criteria**:
- ✅ Encoding converted
- ✅ Headers translated
- ✅ Special characters handled

## Performance File

### [`performance_large_file.csv`](performance_large_file.csv)
**Expected Confidence**: 95%  
**Expected Transformations**: 1 (remove comments only)  
**Performance Criteria**:
- ⚡ Parsing: <2 seconds
- ⚡ Normalization: <3 seconds  
- ⚡ Graph generation: <5 seconds
- 💾 Memory usage: <100MB peak

## Legacy Files (Existing)

### [`european_format.csv`](european_format.csv)
**Status**: ⚠️ Limited data (10 points)  
**Expected Confidence**: 85%  
**Note**: Small dataset for basic European format testing

### [`multi_column.csv`](multi_column.csv)
**Status**: ⚠️ Limited data (10 points)  
**Expected Confidence**: 80%  
**Expected Transformations**: Extract first absorbance column

### [`no_header.csv`](no_header.csv)
**Status**: ⚠️ Limited data (14 points)  
**Expected Confidence**: 75%  
**Expected Transformations**: Add headers

### [`tab_delimited.csv`](tab_delimited.csv)
**Status**: ⚠️ Limited data (10 points)  
**Expected Confidence**: 80%  
**Expected Transformations**: Convert tabs to commas

## AI Normalization Success Metrics

### Overall Success Criteria

1. **Format Detection Accuracy**: >95%
   - Correctly identify delimiter type
   - Detect decimal separator format
   - Recognize header presence/absence

2. **Data Extraction Accuracy**: >98%
   - Preserve all spectral data points
   - Maintain wavenumber-absorbance relationships
   - Handle missing values appropriately

3. **Transformation Correctness**: >90%
   - Accurate unit conversions
   - Proper data type handling
   - Correct column mapping

4. **Processing Performance**: 
   - Small files (<50KB): <1 second
   - Medium files (50-200KB): <3 seconds
   - Large files (>200KB): <10 seconds

5. **Memory Efficiency**: <100MB peak usage for any file

### Confidence Level Distribution

Expected distribution across test files:

- **High Confidence (>85%)**: 8 files (42%)
- **Medium Confidence (70-85%)**: 6 files (32%)
- **Lower Confidence (50-70%)**: 3 files (16%)
- **Problem Files (<50%)**: 2 files (10%)

### Common Transformation Patterns

1. **Delimiter Conversion**: `;` or `\t` → `,`
2. **Decimal Conversion**: `,` → `.`
3. **Header Standardization**: Various → `Wavenumber,Absorbance`
4. **Comment Removal**: `#`, `//`, `*`, `%`, `!` prefixed lines
5. **Column Extraction**: Multiple columns → 2 core columns
6. **Unit Conversion**: Transmittance → Absorbance, Scale corrections
7. **Data Ordering**: Ascending → Descending (for FTIR)
8. **Encoding Conversion**: Latin-1 → UTF-8

### Quality Assurance Checks

After normalization, all files should meet:

1. **Standard Format**:
   - CSV with comma delimiter
   - Period decimal separator
   - UTF-8 encoding
   - Standard headers: `Wavenumber,Absorbance`

2. **Data Quality**:
   - Wavenumber range: 200-12000 cm⁻¹
   - Absorbance range: 0.001-3.0 AU
   - Monotonic wavenumber order
   - No missing values
   - Realistic spectral features

3. **Metadata Preservation**:
   - Original filename referenced
   - Transformation log maintained
   - Confidence score recorded
   - Processing timestamp saved

## Testing Workflow

### Automated Testing

```python
# Example test workflow
def test_normalization_outcome(filename, expected_confidence, expected_transforms):
    # Load test file
    result = ai_normalizer.normalize_file(f"test_data/{filename}")
    
    # Validate confidence
    assert result.confidence >= expected_confidence - 10
    
    # Validate transformations
    assert len(result.transformations) == expected_transforms
    
    # Validate data quality
    assert validate_spectral_data(result.data)
    
    # Validate format
    assert result.data.columns.tolist() == ['Wavenumber', 'Absorbance']
```

### Manual Verification

1. **Visual Inspection**: Check graph output for realistic spectral features
2. **Data Comparison**: Compare input vs output data ranges
3. **Format Verification**: Confirm standard CSV format
4. **Performance Monitoring**: Track processing times and memory usage

## Regression Testing

### Version Comparison

When updating the AI normalization system:

1. **Baseline Comparison**: Compare results with previous version
2. **Confidence Tracking**: Monitor confidence score changes
3. **Performance Monitoring**: Track processing time changes
4. **Accuracy Assessment**: Validate transformation correctness

### Continuous Integration

Recommended CI pipeline:

```yaml
test_pipeline:
  - name: "Generate Test Data"
    run: python3 tests/generate_test_data.py
  
  - name: "Validate Data Quality"
    run: python3 tests/validate_test_data.py
  
  - name: "Run Integration Tests"
    run: python3 tests/test_integration_scenarios.py
  
  - name: "Check Expected Outcomes"
    run: python3 tests/verify_expected_outcomes.py
```

## Troubleshooting Guide

### Low Confidence Scores

If AI confidence is lower than expected:

1. **Check Input Data**: Verify file format and content
2. **Review Transformations**: Ensure transformations are appropriate
3. **Validate Output**: Confirm normalized data quality
4. **Update Expectations**: Adjust confidence thresholds if needed

### Failed Transformations

Common transformation failures:

1. **Encoding Issues**: Non-standard character encodings
2. **Delimiter Ambiguity**: Mixed or unusual delimiters
3. **Data Type Confusion**: Text in numeric columns
4. **Scale Detection**: Ambiguous magnitude scaling

### Performance Issues

If processing is slower than expected:

1. **File Size**: Check for unexpectedly large files
2. **Data Complexity**: Review number of columns and rows
3. **AI Service**: Verify API response times
4. **Memory Usage**: Monitor memory consumption

## Update Procedures

### Adding New Test Cases

1. **Generate Data**: Add new generation method to [`generate_test_data.py`](../generate_test_data.py)
2. **Document Expectations**: Add entry to this file
3. **Update Validation**: Modify validation criteria if needed
4. **Test Integration**: Verify with complete workflow

### Modifying Expectations

When AI normalization improves:

1. **Update Confidence Levels**: Raise expected confidence scores
2. **Refine Transformations**: Update expected transformation counts
3. **Enhance Validation**: Add new quality checks
4. **Document Changes**: Update this file with new expectations

---

*This document should be updated whenever the AI normalization system is modified or test data is regenerated.*

**Last Updated**: 2024-01-15  
**Version**: 1.0  
**Maintainer**: Spectral Analysis Team
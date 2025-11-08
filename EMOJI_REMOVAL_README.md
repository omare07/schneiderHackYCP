# Emoji Removal Script

## Overview

[`remove_emojis.py`](remove_emojis.py) is a comprehensive Python script that removes all emoji characters from the Schneider Prize project codebase.

## Features

✅ **Comprehensive Scanning**
- Python files (`.py`) in `spectral_analyzer/` and `spectral-analyzer-web/backend/`
- TypeScript/JavaScript files (`.ts`, `.tsx`, `.js`, `.jsx`) in `spectral-analyzer-web/frontend/src/`
- Markdown files (`.md`) in project root

✅ **Complete Emoji Detection**
- Emoticons (😀-🙏)
- Symbols & pictographs (🌀-🗿)
- Transport & map symbols (🚀-🛿)
- Flags (🇦-🇿)
- And many more Unicode ranges

✅ **Safety Features**
- Automatic `.backup` file creation before modification
- Dry-run mode to preview changes
- Restore functionality to undo changes
- UTF-8 encoding preservation
- Error handling for problematic files

✅ **Smart Filtering**
- Skips `node_modules`, `__pycache__`, `.git`, and other build directories
- Handles binary files gracefully
- Progress reporting during scan

## Usage

### Preview Changes (Dry Run)
```bash
python3 remove_emojis.py --dry-run
```

This will:
- Scan all relevant files
- Count emojis in each file
- Show which files contain emojis
- Display sample emojis found
- Generate a detailed report
- **NOT modify any files**

### Remove Emojis
```bash
python3 remove_emojis.py
```

This will:
- Scan all relevant files
- Create `.backup` files for modified files
- Remove all emojis from files
- Generate a detailed report

### Restore from Backups
```bash
python3 remove_emojis.py --restore
```

This will:
- Find all `.backup` files in the project
- Restore original content
- Delete backup files

## Example Output

```
🚀 Emoji Removal Script for Schneider Prize Project
📁 Working directory: /path/to/project

🔍 Scanning for files...
   Found 97 files to scan

🔎 DRY RUN MODE - No files will be modified

   Progress: 97/97 files...

======================================================================
📋 DRY RUN REPORT - NO CHANGES MADE
======================================================================

📊 Summary:
   • Files scanned: 97
   • Files with emojis: 29
   • Total emojis found: 513

📝 Files with emojis:
   • spectral_analyzer/demo_caching_system.py: 51 emoji(s)
   • spectral_analyzer/comprehensive_integration_test.py: 44 emoji(s)
   ...

======================================================================

💡 Run without --dry-run to actually remove emojis
```

## Current State

As of the last scan:
- **97 files** scanned across the codebase
- **29 files** contain emojis
- **513 total emojis** detected

Top files with emojis:
1. `spectral_analyzer/demo_caching_system.py` - 51 emojis
2. `spectral_analyzer/tests/run_comprehensive_tests.py` - 45 emojis
3. `spectral_analyzer/comprehensive_integration_test.py` - 44 emojis

## Technical Details

### Emoji Detection Pattern
The script uses a comprehensive Unicode regex pattern covering:
- U+1F600-U+1F64F (emoticons)
- U+1F300-U+1F5FF (symbols & pictographs)
- U+1F680-U+1F6FF (transport & map symbols)
- U+1F700-U+1F77F (alchemical symbols)
- U+1F780-U+1F7FF (geometric shapes extended)
- U+1F800-U+1F8FF (supplemental arrows-C)
- U+1F900-U+1F9FF (supplemental symbols and pictographs)
- U+1FA00-U+1FA6F (chess symbols)
- U+1FA70-U+1FAFF (symbols and pictographs extended-A)
- U+2600-U+27BF (miscellaneous symbols)
- U+1F1E0-U+1F1FF (flags)

### Excluded Directories
- `node_modules`
- `__pycache__`
- `.git`
- `.pytest_cache`
- `venv`, `env`, `.env`
- `dist`, `build`
- `.next`, `.cache`

## Workflow Recommendations

1. **Always run dry-run first** to see what will be changed:
   ```bash
   python3 remove_emojis.py --dry-run
   ```

2. **Review the report** to understand impact

3. **Run the actual removal** when ready:
   ```bash
   python3 remove_emojis.py
   ```

4. **Test your code** after removal to ensure functionality

5. **If needed, restore from backups**:
   ```bash
   python3 remove_emojis.py --restore
   ```

## Notes

- The script preserves file encoding (UTF-8)
- Backup files are created with `.backup` extension
- The script handles permission errors and encoding issues gracefully
- Progress is shown during scanning for better user experience
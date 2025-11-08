#!/usr/bin/env python3
"""
Emoji Removal Script for Schneider Prize Project
Removes all emoji characters from source code files
"""

import re
import sys
import os
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

# Comprehensive emoji regex pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000027BF"  # Miscellaneous symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE
)

# Directories to scan with their file extensions
SCAN_CONFIG = {
    'spectral_analyzer/': ['.py'],
    'spectral-analyzer-web/backend/': ['.py'],
    'spectral-analyzer-web/frontend/src/': ['.ts', '.tsx', '.js', '.jsx'],
    './': ['.md']  # Markdown files in root
}

# Directories to exclude
EXCLUDE_DIRS = {
    'node_modules',
    '__pycache__',
    '.git',
    '.pytest_cache',
    'venv',
    'env',
    '.env',
    'dist',
    'build',
    '.next',
    '.cache'
}


def find_files(directories: List[Tuple[str, List[str]]], base_path: Path) -> List[Path]:
    """
    Find all files with given extensions in directories
    
    Args:
        directories: List of tuples (directory_path, [extensions])
        base_path: Base path of the project
        
    Returns:
        List of Path objects for matching files
    """
    files_found = []
    
    for directory, extensions in directories:
        dir_path = base_path / directory
        
        if not dir_path.exists():
            print(f"⚠️  Warning: Directory not found: {directory}")
            continue
        
        # Handle markdown files in root specially
        if directory == './':
            for ext in extensions:
                files_found.extend(dir_path.glob(f'*{ext}'))
        else:
            # Recursively find files in subdirectories
            for ext in extensions:
                for file_path in dir_path.rglob(f'*{ext}'):
                    # Skip excluded directories
                    if any(excluded in file_path.parts for excluded in EXCLUDE_DIRS):
                        continue
                    if file_path.is_file():
                        files_found.append(file_path)
    
    return sorted(set(files_found))


def count_emojis(text: str) -> int:
    """Count the number of emojis in text"""
    return len(EMOJI_PATTERN.findall(text))


def remove_emojis_from_file(filepath: Path, dry_run: bool = False) -> Tuple[int, str, bool]:
    """
    Remove emojis from a single file
    
    Args:
        filepath: Path to the file
        dry_run: If True, only preview changes without modifying
        
    Returns:
        Tuple of (emoji_count, original_content, was_modified)
    """
    try:
        # Read file with UTF-8 encoding
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Count emojis before removal
        emoji_count = count_emojis(original_content)
        
        if emoji_count == 0:
            return 0, original_content, False
        
        # Remove emojis
        cleaned_content = EMOJI_PATTERN.sub('', original_content)
        
        if not dry_run:
            # Create backup
            backup_path = filepath.with_suffix(filepath.suffix + '.backup')
            shutil.copy2(filepath, backup_path)
            
            # Write cleaned content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
        
        return emoji_count, original_content, True
        
    except UnicodeDecodeError:
        print(f"⚠️  Warning: Could not decode {filepath} as UTF-8, skipping...")
        return 0, "", False
    except PermissionError:
        print(f"⚠️  Warning: Permission denied for {filepath}, skipping...")
        return 0, "", False
    except Exception as e:
        print(f"❌ Error processing {filepath}: {str(e)}")
        return 0, "", False


def restore_backups(base_path: Path) -> int:
    """
    Restore all .backup files
    
    Args:
        base_path: Base path of the project
        
    Returns:
        Number of files restored
    """
    restored = 0
    backup_files = list(base_path.rglob('*.backup'))
    
    if not backup_files:
        print("No backup files found.")
        return 0
    
    print(f"\n🔄 Found {len(backup_files)} backup files")
    
    for backup_path in backup_files:
        try:
            # Get original file path
            original_path = backup_path.with_suffix('')
            
            # Restore backup
            shutil.copy2(backup_path, original_path)
            
            # Remove backup file
            backup_path.unlink()
            
            print(f"✅ Restored: {original_path.relative_to(base_path)}")
            restored += 1
            
        except Exception as e:
            print(f"❌ Error restoring {backup_path}: {str(e)}")
    
    print(f"\n✨ Restored {restored} files from backups")
    return restored


def generate_report(files_scanned: int, files_modified: int, 
                   emoji_details: Dict[Path, int], dry_run: bool) -> None:
    """
    Generate and display report
    
    Args:
        files_scanned: Total number of files scanned
        files_modified: Number of files that were modified
        emoji_details: Dictionary mapping file paths to emoji counts
        dry_run: Whether this was a dry run
    """
    total_emojis = sum(emoji_details.values())
    
    print("\n" + "="*70)
    if dry_run:
        print("📋 DRY RUN REPORT - NO CHANGES MADE")
    else:
        print("📋 EMOJI REMOVAL REPORT")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"   • Files scanned: {files_scanned}")
    print(f"   • Files with emojis: {files_modified}")
    print(f"   • Total emojis found: {total_emojis}")
    
    if emoji_details:
        print(f"\n📝 Files with emojis:")
        for filepath, count in sorted(emoji_details.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {filepath}: {count} emoji(s)")
    else:
        print(f"\n✨ No emojis found in any files!")
    
    if not dry_run and files_modified > 0:
        print(f"\n💾 Backups created: {files_modified} files backed up with .backup extension")
        print(f"   To restore: python3 remove_emojis.py --restore")
    
    print("\n" + "="*70)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Remove emoji characters from codebase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 remove_emojis.py --dry-run    # Preview changes without modifying files
  python3 remove_emojis.py              # Remove emojis and create backups
  python3 remove_emojis.py --restore    # Restore from backups
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    
    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore files from .backup files'
    )
    
    args = parser.parse_args()
    
    # Get base path (current working directory)
    base_path = Path.cwd()
    
    print("🚀 Emoji Removal Script for Schneider Prize Project")
    print(f"📁 Working directory: {base_path}\n")
    
    # Handle restore mode
    if args.restore:
        restore_backups(base_path)
        return
    
    # Prepare scan configuration
    directories = [(dir_path, exts) for dir_path, exts in SCAN_CONFIG.items()]
    
    # Find all files
    print("🔍 Scanning for files...")
    files = find_files(directories, base_path)
    print(f"   Found {len(files)} files to scan\n")
    
    if not files:
        print("⚠️  No files found to scan!")
        return
    
    # Process files
    if args.dry_run:
        print("🔎 DRY RUN MODE - No files will be modified\n")
    else:
        print("✏️  PROCESSING MODE - Files will be modified and backed up\n")
    
    emoji_details = {}
    files_modified = 0
    
    for i, filepath in enumerate(files, 1):
        relative_path = filepath.relative_to(base_path)
        
        # Show progress
        if i % 10 == 0 or i == len(files):
            print(f"   Progress: {i}/{len(files)} files...", end='\r')
        
        emoji_count, original_content, was_modified = remove_emojis_from_file(filepath, args.dry_run)
        
        if was_modified:
            files_modified += 1
            emoji_details[relative_path] = emoji_count
            
            if args.dry_run:
                # Show preview for dry run
                print(f"\n📄 {relative_path}")
                print(f"   Found {emoji_count} emoji(s)")
                
                # Show sample of emojis found
                emojis_found = EMOJI_PATTERN.findall(original_content)
                if emojis_found:
                    sample = emojis_found[:5]
                    print(f"   Sample: {' '.join(sample)}")
                    if len(emojis_found) > 5:
                        print(f"   ... and {len(emojis_found) - 5} more")
    
    print()  # New line after progress
    
    # Generate report
    generate_report(len(files), files_modified, emoji_details, args.dry_run)
    
    # Exit message
    if not args.dry_run and files_modified > 0:
        print("\n✅ Emoji removal complete!")
        print("   Original files backed up with .backup extension")
    elif args.dry_run and files_modified > 0:
        print("\n💡 Run without --dry-run to actually remove emojis")
    else:
        print("\n✅ All files are emoji-free!")


if __name__ == "__main__":
    main()
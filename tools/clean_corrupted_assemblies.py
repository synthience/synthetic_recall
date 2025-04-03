#!/usr/bin/env python

"""
Utility script to clean up corrupted assembly files.

This script scans the assemblies directory for JSON files with invalid format
(missing required fields) and removes them to prevent loading issues on startup.
"""

import json
import os
import sys
import argparse
from pathlib import Path

def validate_assembly_file(file_path):
    """
    Check if an assembly file has the required fields.
    
    Args:
        file_path: Path to the assembly JSON file
        
    Returns:
        (bool, list): Tuple of (is_valid, missing_fields)
    """
    required_fields = [
        "assembly_id", "assembly_schema_version", "name", "creation_time",
        "memories", "memory_ids", # Check for either memories or memory_ids
        "composite_embedding"
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Check if either memories or memory_ids is present
        has_memories = "memories" in data or "memory_ids" in data
        
        missing = [field for field in required_fields 
                  if field not in data and not (field in ["memories", "memory_ids"] and has_memories)]
        
        return len(missing) == 0, missing
    except json.JSONDecodeError:
        return False, ["invalid_json"]
    except Exception as e:
        return False, [str(e)]

def main():
    parser = argparse.ArgumentParser(description="Clean up corrupted assembly files.")
    parser.add_argument("--storage-path", type=str, default="/app/memory/stored/synthians",
                        help="Base path to the memory storage directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't delete files, just report what would be deleted")
    args = parser.parse_args()
    
    # Construct path to assemblies directory
    assemblies_dir = Path(args.storage_path) / "assemblies"
    
    if not assemblies_dir.exists():
        print(f"Error: Assemblies directory {assemblies_dir} does not exist.")
        return 1
    
    # Scan for corrupted files
    corrupted_files = []
    total_files = 0
    
    for file_path in assemblies_dir.glob("*.json"):
        total_files += 1
        is_valid, missing_fields = validate_assembly_file(file_path)
        
        if not is_valid:
            corrupted_files.append((file_path, missing_fields))
    
    # Report findings
    print(f"Found {len(corrupted_files)} corrupted files out of {total_files} total assembly files.")
    
    if corrupted_files:
        print("\nCorrupted Files:")
        for file_path, missing_fields in corrupted_files:
            print(f"  - {file_path.name} (Missing: {', '.join(missing_fields)})")
        
        # Delete corrupted files if not in dry run mode
        if not args.dry_run:
            print("\nDeleting corrupted files...")
            for file_path, _ in corrupted_files:
                try:
                    file_path.unlink()
                    print(f"  - Deleted {file_path.name}")
                except Exception as e:
                    print(f"  - Failed to delete {file_path.name}: {e}")
        else:
            print("\nDry run mode - no files were deleted.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

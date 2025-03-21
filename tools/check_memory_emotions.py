#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check emotion metadata in memory JSON files
"""

import os
import sys
import json
from pathlib import Path
import argparse

def check_memory_files(directory, verbose=False, limit=5):
    """Check memory JSON files for emotion metadata"""
    files = list(Path(directory).glob('*.json'))
    print(f"Found {len(files)} memory files in {directory}")
    
    if limit > 0 and len(files) > limit:
        print(f"Limiting analysis to {limit} files")
        files = files[:limit]
    
    files_with_emotion = 0
    files_without_emotion = 0
    
    for i, file_path in enumerate(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            if verbose:
                print(f"\nFile {i+1}: {file_path.name}")
            
            # Check for emotions in metadata
            metadata = memory_data.get('metadata', {})
            has_emotions = ('emotions' in metadata and metadata['emotions']) or \
                        ('dominant_emotion' in metadata and metadata['dominant_emotion'] != 'unknown')
            
            if has_emotions:
                files_with_emotion += 1
                if verbose:
                    print(f"u2713 Has emotion data: {metadata.get('dominant_emotion', 'N/A')}")
                    print(f"  Emotions: {metadata.get('emotions', {})}")
            else:
                files_without_emotion += 1
                if verbose:
                    print(f"u2717 No emotion data in metadata")
                    print(f"  Metadata keys: {list(metadata.keys())}")
                    if 'emotions' in metadata:
                        print(f"  Empty emotions dict: {metadata.get('emotions', {})}")
                    if 'dominant_emotion' in metadata:
                        print(f"  Dominant emotion: {metadata.get('dominant_emotion', 'N/A')}")
        
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            files_without_emotion += 1
    
    print(f"\nSummary:")
    print(f"Files with emotion data: {files_with_emotion} ({files_with_emotion / len(files) * 100:.2f}%)")
    print(f"Files without emotion data: {files_without_emotion} ({files_without_emotion / len(files) * 100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Check memory JSON files for emotion metadata')
    parser.add_argument('directory', type=str, help='Directory containing memory JSON files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information for each file')
    parser.add_argument('-l', '--limit', type=int, default=5, help='Limit the number of files to check (0 for no limit)')
    
    args = parser.parse_args()
    check_memory_files(args.directory, args.verbose, args.limit)

if __name__ == '__main__':
    main()

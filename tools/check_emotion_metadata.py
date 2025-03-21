#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check emotion metadata in .npz files
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import argparse

def check_npz_files(directory, verbose=False, limit=5):
    """Check .npz files for emotion metadata"""
    files = list(Path(directory).glob('*.npz'))
    print(f"Found {len(files)} .npz files in {directory}")
    
    if limit > 0 and len(files) > limit:
        print(f"Limiting analysis to {limit} files")
        files = files[:limit]
    
    files_with_emotion = 0
    files_without_emotion = 0
    
    for i, file_path in enumerate(files):
        try:
            data = np.load(file_path, allow_pickle=True)
            
            if verbose:
                print(f"\nFile {i+1}: {file_path.name}")
                print(f"Keys in .npz file: {list(data.keys())}")
            
            if 'metadata' in data:
                # Load metadata from the string
                try:
                    metadata_str = str(data['metadata'])
                    metadata = json.loads(metadata_str)
                    
                    # Check for emotions
                    has_emotions = ('emotions' in metadata and metadata['emotions']) or \
                                ('dominant_emotion' in metadata and metadata['dominant_emotion'] != 'unknown')
                    
                    if has_emotions:
                        files_with_emotion += 1
                        if verbose:
                            print(f"✓ Has emotion data: {metadata.get('dominant_emotion', 'N/A')}")
                            print(f"  Emotions: {metadata.get('emotions', {})}")
                    else:
                        files_without_emotion += 1
                        if verbose:
                            print(f"✗ No emotion data in metadata")
                            if 'emotions' in metadata:
                                print(f"  Empty emotions dict: {metadata.get('emotions', {})}")
                            if 'dominant_emotion' in metadata:
                                print(f"  Dominant emotion: {metadata.get('dominant_emotion', 'N/A')}")
                
                except json.JSONDecodeError:
                    print(f"Error decoding JSON metadata in {file_path.name}")
                    if verbose:
                        print(f"Raw metadata: {metadata_str[:200]}...")
                    files_without_emotion += 1
            else:
                files_without_emotion += 1
                if verbose:
                    print(f"✗ No metadata key in {file_path.name}")
        
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            files_without_emotion += 1
    
    print(f"\nSummary:")
    print(f"Files with emotion data: {files_with_emotion} ({files_with_emotion / len(files) * 100:.2f}%)")
    print(f"Files without emotion data: {files_without_emotion} ({files_without_emotion / len(files) * 100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Check .npz files for emotion metadata')
    parser.add_argument('directory', type=str, help='Directory containing .npz files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information for each file')
    parser.add_argument('-l', '--limit', type=int, default=5, help='Limit the number of files to check (0 for no limit)')
    
    args = parser.parse_args()
    check_npz_files(args.directory, args.verbose, args.limit)

if __name__ == '__main__':
    main()

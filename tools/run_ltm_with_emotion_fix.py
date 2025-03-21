#!/usr/bin/env python3
# run_ltm_with_emotion_fix.py - Fix emotion data and then run LTM converter

import os
import sys
import json
import numpy as np
from pathlib import Path
import logging
import argparse
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ltm_emotion_fix')

def load_npz_emotion_data():
    """Load emotion data from NPZ files"""
    # Define directories to search for NPZ files
    npz_dirs = [
        Path('memory/npz_files'),
        Path('memory/test_emotion_data'),
        Path('memory'),  # Search in the root memory directory
        Path()  # Also search in the current directory
    ]
    
    emotion_data = {}
    npz_files = []
    
    # Find all NPZ files
    for npz_dir in npz_dirs:
        if npz_dir.exists():
            npz_files.extend(list(npz_dir.glob('**/*.npz')))
    
    logger.info(f"Found {len(npz_files)} NPZ files to process")
    
    # Extract emotion data from NPZ files
    for npz_file in npz_files:
        try:
            npz_data = np.load(npz_file)
            if 'metadata' in npz_data:
                metadata_json = npz_data['metadata'].item()
                
                # Handle potential binary string format
                if isinstance(metadata_json, bytes):
                    metadata_json = metadata_json.decode('utf-8')
                if isinstance(metadata_json, str) and metadata_json.startswith("b'") and metadata_json.endswith("'"):
                    metadata_json = metadata_json[2:-1].replace('\\', '\\')
                
                # Parse the metadata JSON
                try:
                    stored_metadata = json.loads(metadata_json)
                    
                    # Extract text to use as key
                    text = stored_metadata.get('text', '')
                    if text and ('emotions' in stored_metadata or 'dominant_emotion' in stored_metadata):
                        # Store emotion data keyed by text
                        emotion_data[text] = {
                            'emotions': stored_metadata.get('emotions', {}),
                            'dominant_emotion': stored_metadata.get('dominant_emotion', None)
                        }
                        logger.debug(f"Found emotion data for: {text[:50]}...")
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing metadata JSON from {npz_file.name}: {e}")
                except KeyError as e:
                    logger.warning(f"Missing key in metadata from {npz_file.name}: {e}")
                except Exception as e:
                    logger.warning(f"An error occurred while parsing metadata from {npz_file.name}: {e}")
        except FileNotFoundError:
            logger.warning(f"NPZ file {npz_file.name} not found")
        except EOFError as e:
            logger.warning(f"EOF error in NPZ file {npz_file.name}: {e}")
        except ValueError as e:
            logger.warning(f"Value error loading NPZ file {npz_file.name}: {e}")
        except Exception as e:
            logger.warning(f"An error occurred while loading NPZ file {npz_file.name}: {e}")
    
    logger.info(f"Extracted emotion data for {len(emotion_data)} text entries")
    return emotion_data

def fix_memory_files(emotion_data, memory_dir=None):
    """Fix memory files by adding missing emotion data"""
    # Define memory directories
    memory_dirs = []
    if memory_dir:
        memory_dirs.append(Path(memory_dir))
    else:
        memory_dirs.extend([
            Path('memory/stored'),
            Path('memory/test_memory_output'),
            Path('memory/debug_output')
        ])
    
    memory_files = []
    for memory_dir in memory_dirs:
        if memory_dir.exists():
            memory_files.extend(list(memory_dir.glob('*.json')))
    
    logger.info(f"Found {len(memory_files)} memory files to process")
    
    fixed_count = 0
    already_has_emotion = 0
    no_match_found = 0
    
    for memory_file in memory_files:
        try:
            with open(memory_file, 'r') as f:
                memory_data = json.load(f)
            
            # Check if already has emotion data
            metadata = memory_data.get('metadata', {})
            if metadata and 'emotions' in metadata and 'dominant_emotion' in metadata:
                logger.debug(f"Memory {memory_file.name} already has emotion data")
                already_has_emotion += 1
                continue
            
            # Look for matching text
            text = memory_data.get('text', '')
            if not text:
                logger.debug(f"Memory {memory_file.name} has no text content")
                continue
            
            # First try exact match
            if text in emotion_data:
                # Add emotion data
                if 'metadata' not in memory_data:
                    memory_data['metadata'] = {}
                
                memory_data['metadata']['emotions'] = emotion_data[text]['emotions']
                memory_data['metadata']['dominant_emotion'] = emotion_data[text]['dominant_emotion']
                memory_data['metadata']['emotion_source'] = 'exact_match'
                
                # Save updated file
                with open(memory_file, 'w') as f:
                    json.dump(memory_data, f)
                
                fixed_count += 1
                logger.info(f"Fixed memory file with exact match: {memory_file.name}")
            # Then try partial match
            else:
                matched = False
                for stored_text in emotion_data.keys():
                    # If one text contains the other (in either direction)
                    if text in stored_text or stored_text in text:
                        # Add emotion data from partial match
                        if 'metadata' not in memory_data:
                            memory_data['metadata'] = {}
                        
                        memory_data['metadata']['emotions'] = emotion_data[stored_text]['emotions']
                        memory_data['metadata']['dominant_emotion'] = emotion_data[stored_text]['dominant_emotion']
                        memory_data['metadata']['emotion_source'] = 'partial_match'
                        
                        # Save updated file
                        with open(memory_file, 'w') as f:
                            json.dump(memory_data, f)
                        
                        fixed_count += 1
                        logger.info(f"Fixed memory file with partial match: {memory_file.name}")
                        matched = True
                        break
                
                if not matched:
                    # Add inferred emotion data based on text content analysis
                    if 'nf1' in text.lower() or 'pregnant' in text.lower() or 'pregnancy' in text.lower() or 'neurofibromatosis' in text.lower():
                        # For NF1 and pregnancy related content, likely emotions would be concern/worry
                        if 'metadata' not in memory_data:
                            memory_data['metadata'] = {}
                        
                        memory_data['metadata']['emotions'] = {
                            'concern': 0.55,
                            'worry': 0.25,
                            'hope': 0.20
                        }
                        memory_data['metadata']['dominant_emotion'] = 'concern'
                        memory_data['metadata']['emotion_source'] = 'inferred_medical'
                        
                        # Save updated file
                        with open(memory_file, 'w') as f:
                            json.dump(memory_data, f)
                        
                        fixed_count += 1
                        logger.info(f"Fixed memory file with inferred medical emotion: {memory_file.name}")
                    else:
                        logger.debug(f"No emotion data match found for: {memory_file.name}")
                        no_match_found += 1
        except Exception as e:
            logger.warning(f"Error processing memory file {memory_file.name}: {e}")
    
    logger.info(f"Processing complete:")
    logger.info(f"  - {fixed_count} memory files fixed with emotion data")
    logger.info(f"  - {already_has_emotion} memory files already had emotion data")
    logger.info(f"  - {no_match_found} memory files had no matching emotion data")
    return fixed_count

def run_ltm_converter(ltm_path):
    """Run the LTM converter with the specified path"""
    logger.info("Running LTM converter")
    cmd = ["python", "tools/run_ltm_converter.py", "--ltm-path", ltm_path]
    
    try:
        subprocess.check_call(cmd)
        logger.info("LTM converter completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"LTM converter failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.info("LTM converter was interrupted")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fix emotion data and run LTM converter')
    parser.add_argument('--ltm-path', default='memory/stored/ltm', help='Path to store LTM memories')
    parser.add_argument('--skip-fix', action='store_true', help='Skip fixing emotion data')
    args = parser.parse_args()
    
    # Fix emotion data
    if not args.skip_fix:
        logger.info("Step 1: Fixing emotion data in memory files")
        emotion_data = load_npz_emotion_data()
        if emotion_data:
            memory_dir = Path(args.ltm_path).parent if 'ltm' in args.ltm_path else args.ltm_path
            fix_memory_files(emotion_data, memory_dir)
        else:
            logger.warning("No emotion data found in NPZ files")
    else:
        logger.info("Skipping emotion data fix as requested")
    
    # Run LTM converter
    logger.info("Step 2: Running LTM converter")
    success = run_ltm_converter(args.ltm_path)
    
    if success:
        logger.info("All steps completed successfully")
        return 0
    else:
        logger.error("Process completed with errors")
        return 1

if __name__ == '__main__':
    sys.exit(main())

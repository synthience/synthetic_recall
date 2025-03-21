#!/usr/bin/env python3
# index_embeddings.py - Index the converted embeddings for use with the memory system

import os
import sys
import logging
import time
import json
import uuid
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import memory system components
from memory.lucidia_memory_system.core.long_term_memory import LongTermMemory
from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager
from server.qr_calculator import UnifiedQuickRecallCalculator
from server.memory_system import MemorySystem
from server.memory_index import MemoryIndex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('index_embeddings')

class EmbeddingIndexer:
    """Tool for indexing converted embeddings into the memory system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'embedding_dim': 768,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'source_path': os.path.join('memory', 'indexed', 'embeddings'),
            'target_path': os.path.join('memory', 'stored'),
            'batch_size': 100,
            'min_quickrecal_score': 0.5,
            **(config or {})
        }
        
        logger.info(f"Initializing Embedding Indexer with config: {self.config}")
        
        # Create target directory if it doesn't exist
        os.makedirs(self.config['target_path'], exist_ok=True)
        
        # Initialize memory system
        self.memory_system = MemorySystem({
            'storage_path': Path(self.config['target_path']),
            'embedding_dim': self.config['embedding_dim'],
            'rebuild_threshold': 50  # More frequent rebuilds for better performance
        })
        
        # Initialize QR calculator for assigning scores
        self.qr_calculator = UnifiedQuickRecallCalculator({
            'embedding_dim': self.config['embedding_dim']
        })
        
        # Initialize HPCQR flow manager
        self.flow_manager = HPCQRFlowManager({
            'embedding_dim': self.config['embedding_dim'],
            'calculator': self.qr_calculator
        })
        
        logger.info("Embedding Indexer initialized successfully")
    
    async def process_embedding_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single embedding file."""
        try:
            # Load the .npz file
            data = np.load(file_path)
            
            # Extract embedding and metadata
            embedding = None
            metadata = {}
            content = ""
            
            # Most .npz files have 'embedding' as the key
            if 'embedding' in data:
                embedding = data['embedding']
            elif 'arr_0' in data:  # Some files might use the default array name
                embedding = data['arr_0']
            else:
                # Try to find any array that could be an embedding
                for key in data.keys():
                    if isinstance(data[key], np.ndarray) and len(data[key].shape) == 1:
                        embedding = data[key]
                        break
            
            if embedding is None:
                logger.warning(f"No valid embedding found in {file_path}")
                return {}
            
            # Convert embedding to tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            
            # Extract metadata from the .npz file if it exists
            if 'metadata' in data:
                try:
                    # Get the metadata JSON string
                    metadata_json = str(data['metadata'])
                    
                    # Handle potential binary string format (b'...')
                    if metadata_json.startswith("b'") and metadata_json.endswith("'"):
                        metadata_json = metadata_json[2:-1].replace('\\', '\\')
                    
                    # Parse the metadata JSON
                    stored_metadata = json.loads(metadata_json)
                    
                    # Extract emotional data if present - with extensive logging
                    if 'emotions' in stored_metadata:
                        metadata['emotions'] = stored_metadata['emotions']
                        logger.info(f"Found emotions in {file_path}: {stored_metadata['emotions']}")
                        logger.debug(f"Adding emotions to metadata: {metadata['emotions']}")
                    
                    if 'dominant_emotion' in stored_metadata:
                        metadata['dominant_emotion'] = stored_metadata['dominant_emotion']
                        logger.info(f"Found dominant emotion in {file_path}: {stored_metadata['dominant_emotion']}")
                        logger.debug(f"Adding dominant_emotion to metadata: {metadata['dominant_emotion']}")
                    
                    # Extract other useful metadata
                    if 'text' in stored_metadata:
                        content = stored_metadata['text']
                    
                    if 'source' in stored_metadata:
                        metadata['original_source'] = stored_metadata['source']
                    
                    if 'timestamp' in stored_metadata:
                        metadata['original_timestamp'] = stored_metadata['timestamp']
                    
                    if 'role' in stored_metadata:
                        metadata['role'] = stored_metadata['role']
                    
                    # Log if we found emotion data
                    if 'emotions' in metadata or 'dominant_emotion' in metadata:
                        logger.info(f"Emotion data found for {file_path}: {metadata.get('dominant_emotion', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Error extracting metadata from {file_path}: {e}")
                    # Continue processing with what we have
            
            # Try to extract content from a corresponding JSON file if we don't already have it
            if not content:
                base_name_with_ext = os.path.basename(file_path)
                base_name = os.path.splitext(base_name_with_ext)[0]  # Remove .npz extension
                
                # Extract the conversation number from the filename (e.g., "conversations_11_10")
                # Pattern typically: conversations_XX_YY.cleaned_for_lora_ZZ.npz
                conversation_match = re.match(r'(conversations_\d+_\d+)', base_name)
                conversation_id = conversation_match.group(1) if conversation_match else None
                
                json_filename = f"{conversation_id}.cleaned.json" if conversation_id else f"{base_name}.json"
                
                # Check in potential locations for the JSON file
                potential_paths = [
                    os.path.join('memory', 'stored', 'ltm', 'Frameworks', json_filename),
                    os.path.join('memory', 'stored', 'ltm', 'Frameworks', f"{base_name}.json"),
                    os.path.join('memory', 'stored', 'ltm', json_filename),
                    os.path.join('memory', 'indexed', 'text', json_filename),
                    # Add more potential paths if needed
                ]
                
                json_content = ""
                json_path = None
                
                # Try to find and read the JSON file
                for potential_path in potential_paths:
                    if os.path.exists(potential_path):
                        json_path = potential_path
                        try:
                            with open(potential_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                                
                                # Extract content from JSON file (adapt this based on your JSON structure)
                                if isinstance(json_data, list) and len(json_data) > 0:
                                    # Handle list format
                                    for item in json_data:
                                        if isinstance(item, dict):
                                            # Look for mapping property which contains the conversation
                                            if 'mapping' in item:
                                                # Extract text from messages
                                                for msg_id, msg_data in item['mapping'].items():
                                                    if 'message' in msg_data and msg_data['message']:
                                                        if 'content' in msg_data['message'] and msg_data['message']['content']:
                                                            content_data = msg_data['message']['content']
                                                            if 'parts' in content_data and len(content_data['parts']) > 0:
                                                                # Add message content to the overall content
                                                                msg_content = content_data['parts'][0]
                                                                if isinstance(msg_content, str) and len(msg_content) > 0:
                                                                    json_content += msg_content + "\n\n"
                                            
                                            # If title exists, use it as part of the content
                                            if 'title' in item:
                                                json_content = f"Title: {item['title']}\n\n" + json_content
                                        
                                elif isinstance(json_data, dict):
                                    # Handle dictionary format
                                    title = json_data.get('title', '')
                                    if title:
                                        json_content = f"Title: {title}\n\n"
                                    
                                    # Extract text content - adapt this based on your JSON structure
                                    if 'content' in json_data:
                                        if isinstance(json_data['content'], str):
                                            json_content += json_data['content']
                                        elif isinstance(json_data['content'], dict):
                                            if 'text' in json_data['content']:
                                                json_content += json_data['content']['text']
                                            if 'user_editable_context' in json_data['content'].get('content_type', ''):
                                                # Special handling for user_editable_context
                                                if 'user_profile' in json_data['content']:
                                                    json_content += f"User Profile: {json_data['content']['user_profile']}\n\n"
                                                if 'user_instructions' in json_data['content']:
                                                    json_content += f"User Instructions: {json_data['content']['user_instructions']}\n\n"
                                                    
                                    # Extract from mapping property if available (complex conversation structure)
                                    if 'mapping' in json_data:
                                        json_content += self._extract_content_from_mapping(json_data['mapping'])
                        except Exception as e:
                            logger.error(f"Error reading JSON file {potential_path}: {e}")
                        break  # Stop after finding the first matching JSON file
                
                # If no JSON content was found, fall back to using the filename
                if not json_content:
                    logger.warning(f"No corresponding JSON content found for {file_path}, using filename as content")
                    content = base_name.replace('_', ' ')
                else:
                    content = json_content
                    # Limit content length if it's too long
                    if len(content) > 10000:  # Adjust this threshold as needed
                        content = content[:10000] + "...[truncated]"
            
            # Update metadata with file-related information
            metadata.update({
                'source': file_path,
                'json_source': json_path if 'json_path' in locals() and json_path else None,
                'category': self._extract_category(file_path),
                'timestamp': time.time(),
                'filename': base_name_with_ext if 'base_name_with_ext' in locals() else os.path.basename(file_path)
            })
            
            # Process through HPC-QR flow to get score
            processed_result = await self.flow_manager.process_embedding_batch([embedding_tensor])
            
            # Extract processed embedding and QR score
            if processed_result and len(processed_result) > 0:
                processed_embedding, qr_score = processed_result[0]
            else:
                # Use original embedding and default score if processing fails
                processed_embedding = embedding_tensor
                qr_score = self.config['min_quickrecal_score']
            
            memory_id = str(uuid.uuid4())
            
            # Create memory record
            memory = {
                'id': memory_id,
                'text': content,
                'embedding': processed_embedding.cpu().numpy().tolist(),
                'timestamp': time.time(),
                'quickrecal_score': float(qr_score),
                'metadata': metadata
            }
            
            # Only index if the QR score is above threshold
            if qr_score >= self.config['min_quickrecal_score']:
                # Log the metadata being passed to memory system to ensure it contains emotion data
                if 'emotions' in metadata or 'dominant_emotion' in metadata:
                    logger.info(f"Adding memory with emotion data: {metadata.get('dominant_emotion', 'unknown')}")
                    logger.debug(f"Full metadata being passed to memory system: {metadata}")
                else:
                    logger.warning(f"No emotion data found for {file_path} when adding to memory system")
                
                # Store in memory system
                await self.memory_system.add_memory(
                    text=content,
                    embedding=processed_embedding.cpu(),
                    quickrecal_score=float(qr_score),
                    metadata=metadata  # Now includes emotion data if available
                )
                return memory
            else:
                logger.debug(f"Skipping {file_path} due to low QR score: {qr_score}")
                return {}
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {}
    
    def _extract_category(self, file_path: str) -> str:
        """Extract category from file path."""
        parts = file_path.split(os.path.sep)
        if len(parts) >= 2:
            # Try to extract a meaningful category
            filename = os.path.basename(file_path)
            prefix = filename.split('_')[0] if '_' in filename else ''
            if prefix:
                return prefix
        return "general"
    
    def _extract_content_from_mapping(self, mapping: Dict[str, Any]) -> str:
        """Extract content from a mapping structure."""
        content = ""
        for msg_id, msg_data in mapping.items():
            if 'message' in msg_data and msg_data['message']:
                if 'content' in msg_data['message'] and msg_data['message']['content']:
                    content_data = msg_data['message']['content']
                    if 'parts' in content_data and len(content_data['parts']) > 0:
                        # Add message content to the overall content
                        msg_content = content_data['parts'][0]
                        if isinstance(msg_content, str) and len(msg_content) > 0:
                            content += msg_content + "\n\n"
        return content
    
    async def index_all_embeddings(self) -> Dict[str, Any]:
        """Index all embeddings in the source directory."""
        source_path = self.config['source_path']
        if not os.path.exists(source_path):
            logger.error(f"Source path does not exist: {source_path}")
            return {'indexed': 0, 'failed': 0, 'total': 0}
        
        # Get all .npz files
        embedding_files = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith('.npz'):
                    embedding_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(embedding_files)} embedding files to index")
        
        # Process in batches
        stats = {'indexed': 0, 'failed': 0, 'total': len(embedding_files), 'with_emotion': 0}
        
        for i in tqdm(range(0, len(embedding_files), self.config['batch_size']), desc="Indexing embeddings"):
            batch = embedding_files[i:i+self.config['batch_size']]
            batch_results = []
            
            for file_path in batch:
                result = await self.process_embedding_file(file_path)
                if result:
                    # Check if this record has emotion data
                    if 'metadata' in result and ('emotions' in result['metadata'] or 'dominant_emotion' in result['metadata']):
                        stats['with_emotion'] += 1
                    
                    batch_results.append(result)
                    stats['indexed'] += 1
                else:
                    stats['failed'] += 1
            
            # Update memory index after each batch
            if batch_results:
                logger.info(f"Indexed batch of {len(batch_results)} embeddings")
                
        # Write summary
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_path': self.config['source_path'],
            'target_path': self.config['target_path'],
            'indexed': stats['indexed'],
            'failed': stats['failed'],
            'total': stats['total'],
            'with_emotion': stats['with_emotion'],
            'embedding_dim': self.config['embedding_dim']
        }
        
        summary_path = os.path.join(self.config['target_path'], f"indexing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Indexing complete. Successfully indexed {stats['indexed']} of {stats['total']} embeddings.")
        logger.info(f"Found emotion data in {stats['with_emotion']} of {stats['indexed']} indexed entries.")
        logger.info(f"Summary written to {summary_path}")
        
        return stats

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Index converted embeddings for the memory system')
    parser.add_argument('--source', type=str, default=os.path.join('memory', 'indexed', 'embeddings'), 
                       help='Source directory containing embeddings (.npz files)')
    parser.add_argument('--target', type=str, default=os.path.join('memory', 'stored'),
                       help='Target directory for storing indexed memories')
    parser.add_argument('--embedding-dim', type=int, default=768,
                       help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--min-qr', type=float, default=0.5,
                       help='Minimum QuickRecal score threshold')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger('index_embeddings').setLevel(logging.DEBUG)
    
    indexer = EmbeddingIndexer({
        'source_path': args.source,
        'target_path': args.target,
        'embedding_dim': args.embedding_dim,
        'batch_size': args.batch_size,
        'min_quickrecal_score': args.min_qr
    })
    
    stats = await indexer.index_all_embeddings()
    print(f"\nIndexing Summary:")
    print(f"  - Total embedding files: {stats['total']}")
    print(f"  - Successfully indexed: {stats['indexed']}")
    print(f"  - With emotion data: {stats.get('with_emotion', 0)}")
    print(f"  - Failed to index: {stats['failed']}")
    print(f"  - Success rate: {stats['indexed']/max(stats['total'], 1)*100:.2f}%")
    print(f"  - Emotion data rate: {stats.get('with_emotion', 0)/max(stats['indexed'], 1)*100:.2f}%")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
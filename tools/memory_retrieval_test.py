#!/usr/bin/env python3
# memory_retrieval_test.py - Test memory retrieval with HPC-QR integration

import argparse
import logging
import os
import sys
import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import memory system components
from memory.lucidia_memory_system.core.long_term_memory import LongTermMemory
from memory.lucidia_memory_system.core.integration.hpc_qr_flow_manager import HPCQRFlowManager
from server.qr_calculator import UnifiedQuickRecallCalculator
from server.tensor_server import TensorServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('memory_retrieval_test')

class MemoryRetrievalTester:
    """Test memory retrieval with dimension mismatch handling."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'embedding_dim': 768,
            'use_emotion': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'memory_path': os.path.join('memory', 'stored'),
            'tensor_server_endpoint': 'http://localhost:5010/embeddings',
            **(config or {})
        }
        
        logger.info(f"Initializing Memory Retrieval Tester with config: {self.config}")
        
        # Initialize tensor server for embeddings
        self.tensor_server = TensorServer()
        
        # Initialize QR calculator
        self.qr_calculator = UnifiedQuickRecallCalculator({
            'embedding_dim': self.config['embedding_dim'],
            'use_emotion': self.config['use_emotion']
        })
        
        # Initialize HPCQR flow manager
        self.flow_manager = HPCQRFlowManager({
            'embedding_dim': self.config['embedding_dim'],
            'calculator': self.qr_calculator
        })
        
        # Initialize long-term memory
        self.ltm = LongTermMemory({
            'storage_path': self.config['memory_path'],
            'embedding_dim': self.config['embedding_dim']
        })
        
        logger.info("Memory Retrieval Tester initialized successfully")
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for query text."""
        try:
            embedding = await self.tensor_server.get_embedding(text)
            logger.info(f"Generated embedding with shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a random embedding as fallback
            random_emb = np.random.random(self.config['embedding_dim'])
            return random_emb / np.linalg.norm(random_emb)
    
    async def retrieve_similar_memories(self, query: str, top_k: int = 5, min_quickrecal: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve similar memories for a query."""
        start_time = time.time()
        
        # Generate embedding for query
        query_embedding = await self.get_embedding(query)
        
        # Process through HPC-QR flow
        try:
            processed_result = await self.flow_manager.process_text_and_embedding(
                text=query,
                embedding=query_embedding
            )
            
            # Extract the embedding and QR score from the result
            processed_embedding = processed_result['embedding']
            qr_score = processed_result.get('qr_score', 0.0)
            
            logger.info(f"Query processed with QR score: {qr_score:.4f}")
        except Exception as e:
            logger.error(f"Error processing query through HPC-QR flow: {e}")
            processed_embedding = query_embedding
            qr_score = 0.5  # Default fallback QR score
            
        # Retrieve similar memories
        try:
            similar_memories = await self.ltm.search_memory(
                query=query,
                limit=top_k,
                min_quickrecal_score=min_quickrecal  # Use the proper parameter name
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Retrieved {len(similar_memories)} memories in {elapsed:.2f} seconds")
            
            return similar_memories
        except TypeError as e:
            if "unexpected keyword argument 'min_quickrecal_score'" in str(e):
                logger.warning("LongTermMemory.search_memory doesn't support min_quickrecal_score, trying with min_quickrecal")
                similar_memories = await self.ltm.search_memory(
                    query=query,
                    limit=top_k,
                    min_quickrecal=min_quickrecal
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Retrieved {len(similar_memories)} memories in {elapsed:.2f} seconds")
                
                return similar_memories
            else:
                logger.error(f"Error retrieving memories: {e}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def extract_emotion_data(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotion data from memory metadata safely."""
        emotion_data = {
            'dominant_emotion': 'neutral',
            'emotions': {}
        }
        
        try:
            metadata = memory.get('metadata', {})
            
            # Try to extract emotion data from different possible locations
            if 'emotions' in metadata:
                emotion_data['emotions'] = metadata['emotions']
            elif 'emotional_data' in metadata:
                emotion_data['emotions'] = metadata['emotional_data']
            elif 'emotional_context' in metadata and isinstance(metadata['emotional_context'], dict):
                if 'emotions' in metadata['emotional_context']:
                    emotion_data['emotions'] = metadata['emotional_context']['emotions']
            
            # Try to extract dominant emotion
            if 'dominant_emotion' in metadata:
                emotion_data['dominant_emotion'] = metadata['dominant_emotion']
            elif 'emotional_context' in metadata and isinstance(metadata['emotional_context'], dict):
                if 'dominant_emotion' in metadata['emotional_context']:
                    emotion_data['dominant_emotion'] = metadata['emotional_context']['dominant_emotion']
                elif 'emotional_state' in metadata['emotional_context']:
                    emotion_data['dominant_emotion'] = metadata['emotional_context']['emotional_state']
        except Exception as e:
            logger.warning(f"Error extracting emotion data: {e}")
        
        return emotion_data
    
    def print_memory_results(self, query: str, memories: List[Dict[str, Any]]):
        """Print memory retrieval results."""
        print("\n" + "=" * 80)
        print(f"QUERY: '{query}'")
        print("=" * 80)
        
        if not memories:
            print("\nNo memories found matching the query.")
            return
        
        for idx, memory in enumerate(memories):
            print(f"\n[{idx+1}] Score: {memory.get('score', 0.0):.4f}")
            print(f"Source: {memory.get('source', 'Unknown')}")
            print(f"Timestamp: {memory.get('timestamp', 'Unknown')}")
            
            # Get emotion data safely
            emotion_data = self.extract_emotion_data(memory)
            if emotion_data['emotions']:
                print(f"Dominant Emotion: {emotion_data['dominant_emotion']}")
                top_emotions = sorted(emotion_data['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"Top Emotions: {top_emotions}")
            
            # Get first 150 chars of content with ellipsis
            content = memory.get('content', '')
            if len(content) > 150:
                content = content[:147] + '...'
            print(f"Content: {content}")
        
        print("\n" + "=" * 80)

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test memory retrieval with QR integration')
    parser.add_argument('--query', type=str, default='What is HPC-QR?', help='Query string')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--embedding-dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--emotion', action='store_true', help='Use emotion in retrieval')
    parser.add_argument('--min-quickrecal', type=float, default=0.0, help='Minimum QuickRecal score threshold')
    
    args = parser.parse_args()
    
    # Initialize and run tester
    tester = MemoryRetrievalTester({
        'embedding_dim': args.embedding_dim,
        'use_emotion': args.emotion
    })
    
    # Get memories similar to query
    similar_memories = await tester.retrieve_similar_memories(
        query=args.query,
        top_k=args.top_k,
        min_quickrecal=args.min_quickrecal
    )
    
    # Print results
    tester.print_memory_results(args.query, similar_memories)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

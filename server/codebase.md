# __init__.py

```py


```

# chat_processor.py

```py
import torch
import time
import logging
from typing import Dict, Any, List
from memory_index import MemoryIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatProcessor:
    def __init__(self, memory_index: MemoryIndex, config: Dict[str, Any] = None):
        """Initialize chat processor with memory integration."""
        self.config = {
            'max_memories': 5,
            'min_similarity': 0.7,
            'time_decay': 0.01,
            'significance_threshold': 0.5
        }
        if config:
            self.config.update(config)
            
        self.memory_index = memory_index

    async def retrieve_context(self, query_embedding: torch.Tensor) -> List[Dict]:
        """Retrieve relevant memories with significance and time weighting."""
        # Normalize query embedding
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.clone().detach()
            query_norm = torch.norm(query_embedding, p=2)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
        
        memories = self.memory_index.search(
            query_embedding,
            k=self.config['max_memories']
        )
        
        logger.info(f"Retrieved {len(memories)} memories before filtering")
        for i, m in enumerate(memories):
            logger.info(f"Memory {i}: similarity={m['similarity']:.3f}, content={m['memory'].get('content', 'None')}")
        
        # Filter by similarity threshold
        filtered_memories = [
            m for m in memories 
            if m['similarity'] >= self.config['min_similarity'] 
            and m['memory'].get('content')
        ]
        
        logger.info(f"Filtered to {len(filtered_memories)} memories")
        
        # Sort by significance and similarity
        filtered_memories.sort(
            key=lambda x: (x['memory']['significance'], x['similarity']),
            reverse=True
        )
        
        return filtered_memories

    async def process_chat(self, user_input: str, embedding: torch.Tensor) -> Dict[str, Any]:
        """Process chat with memory integration."""
        context = await self.retrieve_context(embedding)
        messages = []
        
        logger.info(f"Processing chat with {len(context)} context memories")
        
        # Add memory context if available
        if context:
            memory_texts = []
            for memory in context:
                if memory['memory'].get('content'):
                    memory_texts.append(
                        f"Previous Memory (Significance: {memory['memory']['significance']:.2f}, "
                        f"Similarity: {memory['similarity']:.2f}):\n"
                        f"{memory['memory']['content']}"
                    )
            
            if memory_texts:
                logger.info(f"Adding {len(memory_texts)} memories to context")
                messages.append({
                    "role": "system",
                    "content": "Relevant context:\n" + "\n\n".join(memory_texts)
                })
            else:
                logger.warning("No memory texts generated despite having context")
        else:
            logger.warning("No context memories retrieved")
        
        # Add user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Prepare response
        response = {
            "messages": messages,
            "model": "qwen2.5-7b-instruct",
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        
        return response
```

# hpc_flow_manager.py

```py
"""
LUCID RECALL PROJECT
Agent: LucidAurora 1.1
Date: 2/13/25
Time: 4:41 PM EST

HPC Flow Manager: Handles hypersphere processing chain integration with memory system
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCFlowManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'chunk_size': 512,
            'embedding_dim': 768,  # Match memory dimension
            'batch_size': 32,
            'momentum': 0.9,
            'diversity_threshold': 0.7,
            'surprise_threshold': 0.8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        self.momentum_buffer = None
        self.current_batch = []
        self.batch_timestamps = []
        
        logger.info(f"Initialized HPCFlowManager with config: {self.config}")
        
    async def process_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Process a single embedding through the HPC pipeline
        Returns: (processed_embedding, significance_score)
        """
        with torch.no_grad():
            # Move to correct device
            embedding = embedding.to(self.config['device'])
            
            # Project to unit hypersphere
            norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
            normalized = embedding / (norm + 1e-8)
            
            # Calculate surprise if we have momentum
            surprise_score = 0.0
            if self.momentum_buffer is not None:
                surprise_score = self._compute_surprise(normalized)
                
                # Apply shock absorber if surprise is high
                if surprise_score > self.config['surprise_threshold']:
                    normalized = self._apply_shock_absorber(normalized)
            
            # Update momentum buffer
            self._update_momentum(normalized)
            
            # Calculate significance score
            significance = self._calculate_significance(normalized, surprise_score)
            
            return normalized, significance
            
    def _compute_surprise(self, embedding: torch.Tensor) -> float:
        """Calculate surprise score based on momentum buffer"""
        if self.momentum_buffer is None:
            return 0.0
            
        similarity = torch.matmul(embedding, self.momentum_buffer.T)
        return 1.0 - torch.mean(similarity).item()
        
    def _apply_shock_absorber(self, embedding: torch.Tensor) -> torch.Tensor:
        """Smooth out high-surprise embeddings"""
        if self.momentum_buffer is None:
            return embedding
            
        alpha = 1.0 - self.config['momentum']
        absorbed = alpha * embedding + (1 - alpha) * self.momentum_buffer[-1:]
        
        # Re-normalize
        norm = torch.norm(absorbed, p=2, dim=-1, keepdim=True)
        return absorbed / (norm + 1e-8)
        
    def _update_momentum(self, embedding: torch.Tensor):
        """Update momentum buffer with new embedding"""
        if self.momentum_buffer is None:
            self.momentum_buffer = embedding
        else:
            combined = torch.cat([self.momentum_buffer, embedding])
            self.momentum_buffer = combined[-self.config['chunk_size']:]
            
    def _calculate_significance(self, embedding: torch.Tensor, surprise: float) -> float:
        """Calculate significance score for memory storage"""
        # Base significance on combination of:
        # 1. Surprise value (novel information)
        # 2. Embedding magnitude (information content)
        # 3. Diversity from momentum buffer (uniqueness)
        
        magnitude = torch.norm(embedding).item()
        
        if self.momentum_buffer is not None:
            diversity = 1.0 - torch.max(torch.matmul(embedding, self.momentum_buffer.T)).item()
        else:
            diversity = 1.0
            
        # Combine factors (weights can be tuned)
        significance = (
            0.4 * surprise +
            0.3 * magnitude +
            0.3 * diversity
        )
        
        return significance
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current state statistics"""
        return {
            'has_momentum': self.momentum_buffer is not None,
            'momentum_size': len(self.momentum_buffer) if self.momentum_buffer is not None else 0,
            'device': self.config['device']
        }
```

# hpc_server.py

```py
import asyncio
import websockets
import json
import logging
import torch

# Example HPC manager (you'll see the real code below in hpc_sig_flow_manager.py)
from hpc_sig_flow_manager import HPCSIGFlowManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCServer:
    """
    HPCServer listens on ws://0.0.0.0:5005
    Expects JSON messages:
      {
        "type": "process",
        "embeddings": [...]
      }
    or
      {
        "type": "stats"
      }
    """
    def __init__(self, host='0.0.0.0', port=5005):
        self.host = host
        self.port = port
        # HPC manager that does hypothetical processing
        self.hpc_sig_manager = HPCSIGFlowManager({
            'embedding_dim': 384
        })
        logger.info("Initialized HPCServer with HPC-SIG manager")

    def get_stats(self):
        # Return HPC state
        return { 
            'type': 'stats',
            **self.hpc_sig_manager.get_stats()
        }

    async def handle_websocket(self, websocket):
        logger.info(f"New connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data}")

                    if data['type'] == 'process':
                        # Perform HPC pipeline
                        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
                        processed_embedding, significance = await self.hpc_sig_manager.process_embedding(embeddings)

                        # Example response
                        response = {
                            'type': 'processed',
                            'embeddings': processed_embedding.tolist(),
                            'significance': significance
                        }
                        logger.info(f"Sending HPC response: {response}")
                        await websocket.send(json.dumps(response))

                    elif data['type'] == 'stats':
                        stats = self.get_stats()
                        await websocket.send(json.dumps(stats))

                    else:
                        # Unknown message type
                        error_msg = {
                            'type': 'error',
                            'error': f"Unknown message type: {data['type']}"
                        }
                        logger.warning(f"Unknown message type: {data['type']}")
                        await websocket.send(json.dumps(error_msg))

                except Exception as e:
                    err = {'type': 'error', 'error': str(e)}
                    logger.error(f"Error handling HPC message: {str(e)}")
                    await websocket.send(json.dumps(err))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")

        except Exception as e:
            logger.error(f"Unexpected HPC server error: {str(e)}")

    async def start(self):
        logger.info(f"Starting HPC server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_websocket, self.host, self.port):
            logger.info(f"HPC Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # keep running

if __name__ == '__main__':
    server = HPCServer()
    asyncio.run(server.start())

```

# hpc_sig_flow_manager.py

```py
"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 12:08 AM EST

HPC-SIG Flow Manager: Handles hypersphere processing chain and significance calculation
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCSIGFlowManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'chunk_size': 384,  # Match embedding dimension
            'embedding_dim': 768,
            'batch_size': 32,
            'momentum': 0.9,
            'diversity_threshold': 0.7,
            'surprise_threshold': 0.8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        self.momentum_buffer = None
        self.current_batch = []
        self.batch_timestamps = []
        
        logger.info(f"Initialized HPCSIGFlowManager with config: {self.config}")
        
    async def process_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Process a single embedding through the HPC pipeline"""
        with torch.no_grad():
            # Log input shape
            logger.info(f"Input embedding shape: {embedding.shape}")
            
            # Move to correct device
            embedding = embedding.to(self.config['device'])
            
            # Ensure correct shape
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()[:384]
                logger.info(f"Flattened embedding shape: {embedding.shape}")
            
            # Project to unit hypersphere
            norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
            normalized = embedding / (norm + 1e-8)
            logger.info(f"Normalized embedding shape: {normalized.shape}")
            
            # Calculate surprise if we have momentum
            surprise_score = 0.0
            if self.momentum_buffer is not None:
                surprise_score = self._compute_surprise(normalized)
                logger.info(f"Calculated surprise score: {surprise_score}")
                
                # Apply shock absorber if surprise is high
                if surprise_score > self.config['surprise_threshold']:
                    normalized = self._apply_shock_absorber(normalized)
                    logger.info("Applied shock absorber")
            
            # Update momentum buffer
            self._update_momentum(normalized)
            
            # Calculate significance score
            significance = self._calculate_significance(normalized, surprise_score)
            logger.info(f"Calculated significance score: {significance}")
            
            return normalized, significance
            
    def _compute_surprise(self, embedding: torch.Tensor) -> float:
        """Calculate surprise score based on momentum buffer"""
        if self.momentum_buffer is None:
            return 0.0
            
        similarity = torch.matmul(embedding, self.momentum_buffer.T)
        return 1.0 - torch.mean(similarity).item()
        
    def _apply_shock_absorber(self, embedding: torch.Tensor) -> torch.Tensor:
        """Smooth out high-surprise embeddings"""
        if self.momentum_buffer is None:
            return embedding
            
        alpha = 1.0 - self.config['momentum']
        absorbed = alpha * embedding + (1 - alpha) * self.momentum_buffer[-1:]
        
        # Re-normalize
        norm = torch.norm(absorbed, p=2, dim=-1, keepdim=True)
        return absorbed / (norm + 1e-8)
        
    def _update_momentum(self, embedding: torch.Tensor):
        """Update momentum buffer with new embedding"""
        if self.momentum_buffer is None:
            self.momentum_buffer = embedding
        else:
            combined = torch.cat([self.momentum_buffer, embedding])
            self.momentum_buffer = combined[-self.config['chunk_size']:]
            
    def _calculate_significance(self, embedding: torch.Tensor, surprise: float) -> float:
        """Calculate significance score for memory storage"""
        magnitude = torch.norm(embedding).item()
        
        if self.momentum_buffer is not None:
            diversity = 1.0 - torch.max(torch.matmul(embedding, self.momentum_buffer.T)).item()
        else:
            diversity = 1.0
            
        # Combine factors (weights can be tuned)
        significance = (
            0.4 * surprise +
            0.3 * magnitude +
            0.3 * diversity
        )
        
        return significance
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current state statistics"""
        return {
            'has_momentum': self.momentum_buffer is not None,
            'momentum_size': len(self.momentum_buffer) if self.momentum_buffer is not None else 0,
            'device': self.config['device']
        }

```

# memory_core.py

```py
"""
LUCID RECALL PROJECT
Agent: LucidAurora 1.1
Date: 2/13/25
Time: 4:43 PM EST

MemoryCore: Core memory system with HPC integration
"""

import torch
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from ..server.hpc_flow_manager import HPCFlowManager
from .memory_types import MemoryTypes, MemoryEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryCore:
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'dimension': 768,
            'max_size': 10000,
            'batch_size': 32,
            'cleanup_threshold': 0.7,
            'memory_path': Path('/workspace/memory/stored'),
            **(config or {})
        }
        
        # Initialize memory storage
        self.memories = defaultdict(list)
        self.total_memories = 0
        
        # Initialize HPC Manager
        self.hpc_manager = HPCFlowManager(config)
        
        # Performance tracking
        self.last_cleanup_time = time.time()
        self.stats = {
            'processed': 0,
            'stored': 0,
            'cleaned': 0
        }
        
        logger.info(f"Initialized MemoryCore with config: {self.config}")
        
    async def process_and_store(self, embedding: torch.Tensor, memory_type: MemoryTypes) -> bool:
        """Process embedding through HPC pipeline and store if significant"""
        try:
            # Process through HPC pipeline
            processed_embedding, significance = await self.hpc_manager.process_embedding(embedding)
            
            self.stats['processed'] += 1
            
            # Store if significant enough
            if significance > self.config['cleanup_threshold']:
                success = self._store_memory(MemoryEntry(
                    embedding=processed_embedding,
                    memory_type=memory_type,
                    significance=significance,
                    timestamp=time.time()
                ))
                
                if success:
                    self.stats['stored'] += 1
                    
                # Run cleanup if needed
                await self._maybe_cleanup()
                
                return success
                
            return False
            
        except Exception as e:
            logger.error(f"Error in process_and_store: {str(e)}")
            return False
            
    def _store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry in the appropriate type bucket"""
        try:
            # Check if we have room
            if self.total_memories >= self.config['max_size']:
                return False
                
            # Add to appropriate bucket
            self.memories[memory.memory_type].append(memory)
            self.total_memories += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return False
            
    async def _maybe_cleanup(self):
        """Run cleanup if memory usage is high"""
        current_time = time.time()
        
        # Only clean up periodically
        if (current_time - self.last_cleanup_time < 3600 and  # 1 hour
            self.total_memories < self.config['max_size'] * 0.9):  # 90% full
            return
            
        await self._cleanup()
        
    async def _cleanup(self):
        """Remove least significant memories when storage is full"""
        try:
            logger.info("Starting memory cleanup...")
            
            # Sort all memories by significance
            all_memories = []
            for type_memories in self.memories.values():
                all_memories.extend(type_memories)
                
            all_memories.sort(key=lambda x: x.significance)
            
            # Remove bottom 20%
            num_to_remove = len(all_memories) // 5
            memories_to_keep = all_memories[num_to_remove:]
            
            # Reset storage
            self.memories = defaultdict(list)
            self.total_memories = 0
            
            # Re-add memories to keep
            for memory in memories_to_keep:
                self._store_memory(memory)
                
            self.stats['cleaned'] += num_to_remove
            self.last_cleanup_time = time.time()
            
            logger.info(f"Cleanup complete. Removed {num_to_remove} memories")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    def get_recent_memories(self, count: int = 5, memory_type: Optional[MemoryTypes] = None) -> List[MemoryEntry]:
        """Get most recent memories, optionally filtered by type"""
        try:
            if memory_type:
                memories = self.memories[memory_type]
            else:
                memories = []
                for type_memories in self.memories.values():
                    memories.extend(type_memories)
                    
            # Sort by timestamp descending
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            
            return memories[:count]
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {str(e)}")
            return []
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current memory system statistics"""
        return {
            'total_memories': self.total_memories,
            'memory_types': {k: len(v) for k, v in self.memories.items()},
            'processed': self.stats['processed'],
            'stored': self.stats['stored'],
            'cleaned': self.stats['cleaned'],
            'last_cleanup': self.last_cleanup_time,
            'hpc_stats': self.hpc_manager.get_stats()
        }
```

# memory_index.py

```py
import torch
import numpy as np
import time

class MemoryIndex:
    def __init__(self, embedding_dim=384, rebuild_threshold=100, time_decay=0.01, min_similarity=0.7):
        """Initialize memory index with configurable parameters."""
        self.embedding_dim = embedding_dim
        self.rebuild_threshold = rebuild_threshold
        self.time_decay = time_decay
        self.min_similarity = min_similarity
        self.memories = []
        self.index = None

    async def add_memory(self, memory_id, embedding, timestamp, significance=1.0, content=None):
        """Add a memory with an embedding, timestamp, significance score, and content."""
        # Ensure embedding is normalized
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.clone().detach()
        else:
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Normalize embedding
        norm = torch.norm(embedding, p=2)
        if norm > 0:
            embedding = embedding / norm

        memory = {
            'id': memory_id,
            'embedding': embedding,
            'timestamp': timestamp,
            'significance': significance,
            'content': content or ""  # Ensure content is never None
        }
        self.memories.append(memory)

        if len(self.memories) % self.rebuild_threshold == 0:
            self.build_index()
        
        return memory

    def build_index(self):
        """Build the search index from stored memories."""
        if not self.memories:
            return
        
        # Stack and normalize embeddings
        embeddings = torch.stack([m['embedding'] for m in self.memories])
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        self.index = embeddings / (norms + 1e-8)  # Add epsilon to prevent division by zero
        print(f"🔹 Built index with {len(self.memories)} memories")

    def search(self, query_embedding, k=5):
        """Search for top-k similar memories with time decay and significance weighting."""
        if self.index is None:
            self.build_index()
            
        if not self.memories:
            return []

        # Normalize query embedding
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.clone().detach()
        else:
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        
        query_norm = torch.norm(query_embedding, p=2)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding

        # Compute cosine similarities
        similarities = torch.matmul(self.index, query_normalized)

        # Apply significance weighting
        significance_scores = torch.tensor([m['significance'] for m in self.memories])
        weighted_similarities = similarities * significance_scores

        # Apply time decay (newer memories get a boost)
        timestamps = torch.tensor([m['timestamp'] for m in self.memories], dtype=torch.float32)
        max_timestamp = torch.max(timestamps)
        time_decay_weights = torch.exp(-self.time_decay * (max_timestamp - timestamps))
        final_scores = weighted_similarities * time_decay_weights

        # Get top k results
        k = min(k, len(self.memories))
        values, indices = torch.topk(final_scores, k)

        results = []
        for val, idx in zip(values, indices):
            results.append({
                'memory': self.memories[idx],
                'similarity': similarities[idx].item()  # Use raw similarity for threshold checks
            })

        return results
```

# memory_system.py

```py
import torch
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'storage_path': Path('../memory/stored'),
            'embedding_dim': 384,
            'rebuild_threshold': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        self.memories = []
        self.storage_path = Path(self.config['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing memories
        self._load_memories()
        logger.info(f"Initialized MemorySystem with {len(self.memories)} memories")

    def _load_memories(self):
        """Load all memories from disk."""
        self.memories = []
        try:
            for file_path in self.storage_path.glob('*.json'):
                with open(file_path, 'r') as f:
                    memory = json.load(f)
                    if isinstance(memory, dict) and 'timestamp' in memory:
                        self.memories.append(memory)
            
            # Sort by timestamp if memories exist
            if self.memories:
                self.memories.sort(key=lambda x: x.get('timestamp', 0))
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            self.memories = []

    async def add_memory(self, text: str, embedding: torch.Tensor, 
                        significance: float = None) -> Dict[str, Any]:
        """Add memory with persistence."""
        # Normalize embedding
        embedding = self._normalize_embedding(embedding)
        
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        
        memory = {
            'id': memory_id,
            'text': text,
            'embedding': embedding.tolist(),
            'timestamp': timestamp,
            'significance': significance
        }
        
        # Add to memory list
        self.memories.append(memory)
        
        # Save to disk
        self._save_memory(memory)
        
        logger.info(f"Stored memory {memory_id} with significance {significance}")
        return memory

    async def search_memories(self, query_embedding: torch.Tensor, 
                            limit: int = 5) -> List[Dict]:
        """Search for similar memories."""
        if not self.memories:
            return []
            
        # Normalize query
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Calculate similarities
        similarities = []
        for memory in self.memories:
            memory_embedding = torch.tensor(memory['embedding'], 
                                         device=self.config['device'])
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                memory_embedding.unsqueeze(0)
            )
            similarities.append({
                'memory': memory,
                'similarity': similarity.item()
            })
        
        # Sort by similarity and significance
        sorted_memories = sorted(
            similarities,
            key=lambda x: (x['similarity'] * 0.7 + 
                          (x['memory']['significance'] or 0) * 0.3),
            reverse=True
        )
        
        return sorted_memories[:limit]

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Normalize embedding vector."""
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, device=self.config['device'])
        embedding = embedding.to(self.config['device'])
        norm = torch.norm(embedding, p=2)
        return embedding / norm if norm > 0 else embedding

    def _save_memory(self, memory: Dict[str, Any]):
        """Save memory to disk."""
        try:
            file_path = self.storage_path / f"{memory['id']}.json"
            with open(file_path, 'w') as f:
                json.dump(memory, f)
        except Exception as e:
            logger.error(f"Error saving memory {memory['id']}: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            latest_timestamp = max([m.get('timestamp', 0) for m in self.memories]) if self.memories else 0
        except Exception as e:
            logger.error(f"Error calculating latest timestamp: {str(e)}")
            latest_timestamp = 0

        return {
            'memory_count': len(self.memories),
            'device': self.config['device'],
            'storage_path': str(self.storage_path),
            'latest_timestamp': latest_timestamp
        }
```

# memory_types.py

```py
"""
LUCID RECALL PROJECT
Agent: LucidAurora 1.1
Date: 2/13/25
Time: 4:42 PM EST

Memory Types: Definitions for memory system types and structures
"""

from enum import Enum
from dataclasses import dataclass
import torch
from typing import Optional
import time

class MemoryTypes(Enum):
    """Types of memories that can be stored"""
    EPISODIC = "episodic"      # Event/experience memories
    SEMANTIC = "semantic"       # Factual/conceptual memories
    PROCEDURAL = "procedural"   # Skill/procedure memories
    WORKING = "working"         # Temporary processing memories
    
@dataclass
class MemoryEntry:
    """Container for a single memory entry"""
    embedding: torch.Tensor
    memory_type: MemoryTypes
    significance: float = 0.0
    timestamp: float = time.time()
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Validate memory entry on creation"""
        if not isinstance(self.embedding, torch.Tensor):
            raise ValueError("Embedding must be a torch.Tensor")
            
        if not isinstance(self.memory_type, MemoryTypes):
            raise ValueError("Invalid memory type")
            
        if self.significance < 0.0 or self.significance > 1.0:
            raise ValueError("Significance must be between 0 and 1")
```

# run_voice_server.py

```py
import asyncio
import logging
from server.websocket_server import WebSocketServer, WebSocketMessage
from voice_core.voice_handler import VoiceHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create voice handler instance and make it globally available
voice_handler = VoiceHandler()

async def ensure_session(client_id: str):
    """Ensure a session exists for the client"""
    if client_id not in voice_handler.sessions:
        await voice_handler.initialize_session(client_id)
    return voice_handler.sessions[client_id]

async def handle_voice_input(message: WebSocketMessage) -> dict:
    """Handle voice input messages"""
    session = await ensure_session(message.client_id)
    return await voice_handler.handle_voice_input(message.data, message.client_id)

async def handle_session_control(message: WebSocketMessage) -> dict:
    """Handle session control messages"""
    session = await ensure_session(message.client_id)
    return await voice_handler.handle_session_control(message.data, message.client_id)

async def handle_start_listening(message: WebSocketMessage) -> dict:
    """Handle start listening messages"""
    session = await ensure_session(message.client_id)
    await voice_handler.handle_start_listening(session)
    return {"status": "started"}

async def handle_stop_listening(message: WebSocketMessage) -> dict:
    """Handle stop listening messages"""
    session = await ensure_session(message.client_id)
    await voice_handler.handle_stop_listening(session)
    return {"status": "stopped"}

async def main():
    try:
        # Create and configure WebSocket server
        server = WebSocketServer(host="127.0.0.1", port=5410)
        
        # Register specific handlers for each message type
        server.register_handler("voice_input", handle_voice_input)
        server.register_handler("session_control", handle_session_control)
        server.register_handler("start_listening", handle_start_listening)
        server.register_handler("stop_listening", handle_stop_listening)
        
        # Start server
        logger.info("Starting voice WebSocket server...")
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await voice_handler.cleanup()
        await server.stop()
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        await voice_handler.cleanup()
        await server.stop()
        raise

if __name__ == "__main__":
    asyncio.run(main())

```

# significance_calculator.py

```py
"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 5:50 AM EST

Significance Calculator: Memory Importance Evaluation
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SignificanceCalculator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'time_weight': 0.4,    # aT weight
            'info_weight': 0.3,    # bI weight
            'pattern_weight': 0.3,  # cP weight
            'decay_rate': 0.1,
            'history_window': 1000,
            'batch_size': 32,
            **(config or {})
        }
        
        self.significance_history = []
        self.start_time = time.time()
        
        logger.info(f"Initialized SignificanceCalculator with config: {self.config}")

    def calculate(self, embedding: torch.Tensor, context: List[torch.Tensor]) -> float:
        """
        Calculate significance using the equation:
        Significance = ∫(aT + bI + cP) × f(t) × g(past) × ψ(x,t) dt
        """
        try:
            # Calculate components
            temporal = self._temporal_component()  # T
            info = self._information_component(embedding, context)  # I
            pattern = self._predictive_component(embedding, context)  # P
            
            # Combine with weights (a, b, c)
            weighted_sum = (
                self.config['time_weight'] * temporal +
                self.config['info_weight'] * info +
                self.config['pattern_weight'] * pattern
            )
            
            # Apply time evolution f(t)
            time_evolution = self._time_evolution()
            
            # Apply historical context g(past)
            historical = self._historical_context()
            
            # Apply state function ψ(x,t)
            state = self._state_function(embedding)
            
            # Calculate final significance
            significance = weighted_sum * time_evolution * historical * state
            
            # Track history
            self.significance_history.append(significance.item())
            if len(self.significance_history) > self.config['history_window']:
                self.significance_history = self.significance_history[-self.config['history_window']:]
            
            return significance.item()
            
        except Exception as e:
            logger.error(f"Error calculating significance: {str(e)}")
            return 0.0

    def _information_component(self, embedding: torch.Tensor, context: List[torch.Tensor]) -> torch.Tensor:
        """Calculate information density/uniqueness"""
        if not context:
            return torch.tensor(1.0, device=embedding.device)
            
        with torch.no_grad():
            context_tensor = torch.stack(context).to(embedding.device)
            similarities = torch.matmul(embedding, context_tensor.T)
            uniqueness = 1.0 - torch.mean(similarities)
            return torch.clamp(uniqueness, 0.0, 1.0)

    def _predictive_component(self, embedding: torch.Tensor, context: List[torch.Tensor]) -> torch.Tensor:
        """Calculate predictive value"""
        if not context:
            return torch.tensor(0.5, device=embedding.device)
            
        with torch.no_grad():
            recent = context[-10:]  # Use last 10 memories
            context_tensor = torch.stack(recent).to(embedding.device)
            pred_value = torch.matmul(embedding, context_tensor.T)
            return torch.clamp(torch.sigmoid(pred_value.mean()), 0.0, 1.0)

    def _temporal_component(self) -> torch.Tensor:
        """Calculate temporal relevance"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        return torch.tensor(np.exp(-self.config['decay_rate'] * elapsed))

    def _time_evolution(self) -> float:
        """Time evolution function f(t)"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        return np.sin(2 * np.pi * elapsed / (24 * 3600)) * 0.5 + 0.5  # Daily cycle

    def _historical_context(self) -> float:
        """Historical context function g(past)"""
        if not self.significance_history:
            return 1.0
        recent_significance = np.mean(self.significance_history[-10:])
        return np.clip(recent_significance, 0.1, 1.0)

    def _state_function(self, embedding: torch.Tensor) -> float:
        """State function ψ(x,t)"""
        # Use embedding magnitude as state indicator
        with torch.no_grad():
            magnitude = torch.norm(embedding)
            return torch.clamp(torch.sigmoid(magnitude), 0.1, 1.0).item()

    def get_stats(self) -> Dict[str, Any]:
        """Get calculator statistics"""
        stats = {
            'history_size': len(self.significance_history),
            'uptime': time.time() - self.start_time
        }
        
        if self.significance_history:
            stats.update({
                'mean': np.mean(self.significance_history),
                'std': np.std(self.significance_history),
                'min': np.min(self.significance_history),
                'max': np.max(self.significance_history)
            })
            
        return stats

```

# tensor_server.py

```py
"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 1:19 AM EST

Tensor Server: Memory & Embedding Operations with Unified Memory System
"""

import asyncio
import websockets
import json
import logging
import torch
import time
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
from server.hpc_sig_flow_manager import HPCSIGFlowManager
from server.memory_system import MemorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5001):
        self.host = host
        self.port = port
        self.setup_gpu()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        if torch.cuda.is_available():
            self.model.to('cuda')
        logger.info(f"Model loaded on {self.model.device}")
        
        # Initialize HPC manager
        self.hpc_manager = HPCSIGFlowManager({
            'embedding_dim': 384,
            'device': self.device
        })
        
        # Initialize unified memory system
        self.memory_system = MemorySystem({
            'device': self.device,
            'embedding_dim': 384
        })
        logger.info("Initialized unified memory system")

    def setup_gpu(self) -> None:
        if torch.cuda.is_available():
            self.device = 'cuda'
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
            logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            logger.warning("GPU not available, using CPU")

    async def add_memory(self, text: str, embedding: torch.Tensor) -> Dict[str, Any]:
        """Add memory with embedding and return metadata."""
        # Process through HPC
        processed_embedding, significance = await self.hpc_manager.process_embedding(embedding)
        
        # Store in unified memory system
        memory = await self.memory_system.add_memory(
            text=text,
            embedding=processed_embedding,
            significance=significance
        )
        
        logger.info(f"Stored memory {memory['id']} with significance {significance}")
        return {
            'id': memory['id'],
            'significance': significance,
            'timestamp': memory['timestamp']
        }

    async def search_memories(self, query_embedding: torch.Tensor, limit: int = 5) -> List[Dict]:
        """Search for similar memories."""
        # Get processed query embedding
        processed_query, _ = await self.hpc_manager.process_embedding(query_embedding)
        
        # Search using unified memory system
        results = await self.memory_system.search_memories(
            query_embedding=processed_query,
            limit=limit
        )
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        memory_stats = self.memory_system.get_stats()
        stats = {
            'type': 'stats',
            'gpu_memory': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            'gpu_cached': torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            'device': self.device,
            'hpc_status': self.hpc_manager.get_stats(),
            'memory_count': memory_stats['memory_count'],
            'storage_path': memory_stats['storage_path']
        }
        logger.info(f"Stats requested: {stats}")
        return stats

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections and messages."""
        try:
            logger.info(f"New connection from {websocket.remote_address}")
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data}")

                    if data['type'] == 'embed':
                        # Generate embedding
                        embeddings = self.model.encode(data['text'])
                        
                        # Store memory
                        metadata = await self.add_memory(
                            data['text'], 
                            torch.tensor(embeddings)
                        )
                        
                        response = {
                            'type': 'embeddings',
                            'embeddings': embeddings.tolist(),
                            **metadata
                        }
                        
                    elif data['type'] == 'search':
                        # Generate query embedding
                        query_embedding = self.model.encode(data['text'])
                        
                        # Search memories
                        results = await self.search_memories(
                            torch.tensor(query_embedding),
                            limit=data.get('limit', 5)
                        )
                        
                        response = {
                            'type': 'search_results',
                            'results': [{
                                'id': r['memory']['id'],
                                'text': r['memory']['text'],
                                'similarity': r['similarity'],
                                'significance': r['memory']['significance']
                            } for r in results]
                        }
                        
                    elif data['type'] == 'stats':
                        response = self.get_stats()
                        
                    else:
                        response = {
                            'type': 'error',
                            'error': f"Unknown message type: {data['type']}"
                        }
                        logger.warning(f"Unknown message type received: {data['type']}")

                    await websocket.send(json.dumps(response))
                    logger.info(f"Sent response: {response['type']}")

                except Exception as e:
                    error_msg = {
                        'type': 'error',
                        'error': str(e)
                    }
                    logger.error(f"Error processing message: {str(e)}")
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    async def start(self):
        """Start the WebSocket server."""
        async with websockets.serve(self.handle_websocket, self.host, self.port):
            logger.info(f"Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()

if __name__ == '__main__':
    server = TensorServer()
    asyncio.run(server.start())

```

# websocket_server.py

```py
import asyncio
import json
import websockets
from typing import Dict, Any, Callable, Awaitable, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WebSocketMessage:
    type: str
    data: Dict[str, Any]
    client_id: str

class WebSocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.handlers: Dict[str, Callable[[WebSocketMessage], Awaitable[Dict[str, Any]]]] = {}
        self.server: Optional[websockets.WebSocketServer] = None

    def register_handler(self, message_type: str, handler: Callable[[WebSocketMessage], Awaitable[Dict[str, Any]]]):
        """Register a handler for a specific message type"""
        self.handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connections"""
        client_id = str(id(websocket))
        self.clients[client_id] = websocket
        logger.info(f"New client connected. ID: {client_id}")

        try:
            # Send initial connection success message
            await websocket.send(json.dumps({
                "type": "connection_status",
                "status": "connected",
                "client_id": client_id
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type in self.handlers:
                        # Create message object
                        ws_message = WebSocketMessage(
                            type=msg_type,
                            data=data,
                            client_id=client_id
                        )
                        
                        try:
                            # Call appropriate handler
                            response = await self.handlers[msg_type](ws_message)
                            
                            # Ensure response is JSON serializable
                            json.dumps(response)  # Test serialization
                            
                            # Send response back to client
                            await websocket.send(json.dumps(response))
                        except Exception as e:
                            logger.error(f"Handler error: {str(e)}")
                            await websocket.send(json.dumps({
                                "type": "error",
                                "error": f"Handler error: {str(e)}"
                            }))
                    else:
                        logger.warning(f"No handler for message type: {msg_type}")
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": f"Unsupported message type: {msg_type}"
                        }))
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON format"
                    }))
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Client connection closed: {client_id}")
                    break
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    try:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": str(e)
                        }))
                    except:
                        pass

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected. ID: {client_id}")
        except Exception as e:
            logger.error(f"Unexpected error in client handler: {str(e)}")
        finally:
            # Clean up client connection
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Cleaned up client: {client_id}")
                
                # Call cleanup on voice handler if available
                try:
                    from voice_core.voice_handler import voice_handler
                    await voice_handler.cleanup_session(client_id)
                except Exception as e:
                    logger.error(f"Error cleaning up voice session: {str(e)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        disconnected_clients = []
        
        for client_id, websocket in self.clients.items():
            try:
                await websocket.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.clients:
                del self.clients[client_id]

    async def start(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10
            )
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            await self.server.wait_closed()
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            raise

    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            # Close all client connections
            close_tasks = []
            for websocket in self.clients.values():
                try:
                    close_tasks.append(websocket.close())
                except:
                    pass
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

# Example handlers for voice and memory operations
async def handle_voice_input(message: WebSocketMessage) -> Dict[str, Any]:
    """Handle voice input messages"""
    text = message.data.get('text', '')
    logger.info(f"Received voice input from client {message.client_id}: {text}")
    # Process voice input here
    return {
        "type": "voice_response",
        "text": f"Processed voice input: {text}"
    }

async def handle_memory_operation(message: WebSocketMessage) -> Dict[str, Any]:
    """Handle memory operation messages"""
    operation = message.data.get('operation')
    content = message.data.get('content', '')
    logger.info(f"Received memory operation from client {message.client_id}: {operation}")
    # Process memory operation here
    return {
        "type": "memory_response",
        "operation": operation,
        "status": "success"
    }

# Example usage
if __name__ == "__main__":
    server = WebSocketServer()
    
    # Register handlers
    server.register_handler("voice_input", handle_voice_input)
    server.register_handler("memory_operation", handle_memory_operation)
    
    # Run the server
    asyncio.run(server.start())

```


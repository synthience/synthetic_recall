import torch
import numpy as np
import time

class MemoryIndex:
    def __init__(self, embedding_dim=384, rebuild_threshold=100, time_decay=0.01, min_similarity=0.7):
        """Initialize memory index with HPC-QR-friendly parameters."""
        self.embedding_dim = embedding_dim
        self.rebuild_threshold = rebuild_threshold
        self.time_decay = time_decay
        self.min_similarity = min_similarity
        self.memories = []
        self.index = None

    async def add_memory(self, memory_id, embedding, timestamp, significance=1.0, content=None):
        """
        Add a memory with HPC-QR 'quickrecal_score' (renamed from significance).
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.clone().detach()
        else:
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        norm = torch.norm(embedding, p=2)
        if norm > 0:
            embedding = embedding / norm

        memory = {
            'id': memory_id,
            'embedding': embedding,
            'timestamp': timestamp,
            'quickrecal_score': significance,  # rename significance => quickrecal_score
            'content': content or ""
        }
        self.memories.append(memory)

        if len(self.memories) % self.rebuild_threshold == 0:
            self.build_index()
        
        return memory

    def add_memory_sync(self, memory_id, embedding, timestamp, significance=1.0, content=None, quickrecal_score=None):
        """Synchronous version of add_memory."""
        # Use quickrecal_score if provided, otherwise use significance
        if quickrecal_score is not None:
            significance = quickrecal_score
            
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.clone().detach()
        else:
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        norm = torch.norm(embedding, p=2)
        if norm > 0:
            embedding = embedding / norm

        memory = {
            'id': memory_id,
            'embedding': embedding,
            'timestamp': timestamp,
            'quickrecal_score': significance,
            'content': content or ""
        }
        self.memories.append(memory)

        if len(self.memories) % self.rebuild_threshold == 0:
            self.build_index()
        
        return memory

    def build_index(self):
        if not self.memories:
            return
        
        embeddings = torch.stack([m['embedding'] for m in self.memories])
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        self.index = embeddings / (norms + 1e-8)
        print(f" Built index with {len(self.memories)} memories")

    def search(self, query_embedding, k=5):
        if self.index is None:
            self.build_index()
            
        if not self.memories:
            return []

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.clone().detach()
        else:
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        
        query_norm = torch.norm(query_embedding, p=2)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding

        similarities = torch.matmul(self.index, query_normalized)
        
        # Weighted by quickrecal_score
        quickrecal_scores = torch.tensor([m['quickrecal_score'] for m in self.memories])
        weighted_similarities = similarities * quickrecal_scores
        
        timestamps = torch.tensor([m['timestamp'] for m in self.memories], dtype=torch.float32)
        max_timestamp = torch.max(timestamps)
        time_decay_weights = torch.exp(-self.time_decay * (max_timestamp - timestamps))
        final_scores = weighted_similarities * time_decay_weights

        k = min(k, len(self.memories))
        values, indices = torch.topk(final_scores, k)

        results = []
        for val, idx in zip(values, indices):
            results.append({
                'memory': self.memories[idx],
                'similarity': similarities[idx].item()  # raw similarity
            })

        return results

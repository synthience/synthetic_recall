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
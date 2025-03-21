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
        
        # Score fusion parameters
        self.fusion_weights = {
            'similarity_weight': 0.6,  # Increased from 0.5
            'quickrecal_weight': 0.4   # Decreased from 0.5
        }

    async def add_memory(self, memory_id, embedding, timestamp, significance=1.0, content=None, quickrecal_score=None):
        """
        Add a memory with HPC-QR 'quickrecal_score' (renamed from significance).
        """
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

    def search(self, query_embedding, k=5, min_quickrecal_score=0.0, threshold_boost=0.0):
        """Search for similar memories, with optional filtering by quickrecal score.
        
        Args:
            query_embedding: The embedding to search for
            k: Number of results to return
            min_quickrecal_score: Minimum quickrecal score threshold
            threshold_boost: Additional boost to apply to the threshold
                dynamically based on score volatility (0-1)
                
        Returns:
            List of {memory, similarity, combined_score} dicts sorted by combined_score
        """
        if self.index is None or len(self.memories) == 0:
            return []
            
        # Calculate the effective threshold with the dynamic boost
        effective_threshold = min_quickrecal_score
        if threshold_boost > 0.0:
            # Apply a sigmoid-based boost to make it more aggressive at higher values
            # This makes high volatility periods require higher scores
            boost_factor = 1.0 / (1.0 + np.exp(-10 * (threshold_boost - 0.5)))
            # Scale the boost to ensure we don't exceed 1.0
            max_boost = 1.0 - min_quickrecal_score
            applied_boost = boost_factor * max_boost * 0.5  # 50% of available headroom
            effective_threshold = min_quickrecal_score + applied_boost

        # Convert to torch tensor if it's a numpy array
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
            
        # Normalize query
        query_norm = torch.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding

        # Get embedding similarities
        similarities = torch.matmul(self.index, query_normalized)
        
        # Get QR scores for all memories
        qr_scores = torch.tensor([memory['quickrecal_score'] for memory in self.memories])
        
        # Stage 2.1: Weighted Score Fusion - combine similarity and quickrecal scores
        # Use logarithmic fusion to prevent high scores from dominating
        log_similarities = torch.log1p(similarities * 9 + 1) / torch.log(torch.tensor(11.0))  # log scaling to [0,1]
        log_qr_scores = torch.log1p(qr_scores * 9 + 1) / torch.log(torch.tensor(11.0))  # log scaling to [0,1]
        
        # Create a combined score using the weighted fusion approach with log scaling
        combined_scores = (
            self.fusion_weights['similarity_weight'] * log_similarities + 
            self.fusion_weights['quickrecal_weight'] * log_qr_scores
        )
        
        # If we still want to apply a minimum threshold to QR scores (softer filtering)
        if effective_threshold > 0:
            # Instead of hard threshold, use sigmoid weighting to smoothly dampen low QR scores
            qr_weight = 1.0 / (1.0 + torch.exp(-12 * (qr_scores - effective_threshold)))  # sigmoid centered at threshold
            combined_scores = combined_scores * qr_weight
            
        # Get top-k results
        k = min(k, len(self.memories))
        values, indices = torch.topk(combined_scores, k)
        
        results = []
        for val, idx in zip(values, indices):
            idx_item = idx.item()
            results.append({
                'memory': self.memories[idx_item],
                'similarity': similarities[idx_item].item(),
                'qr_score': qr_scores[idx_item].item(),
                'combined_score': val.item()
            })
        
        return results

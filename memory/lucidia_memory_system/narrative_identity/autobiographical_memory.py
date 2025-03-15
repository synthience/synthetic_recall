"""
Lucidia's Autobiographical Memory

This module implements a specialized memory system for identity-relevant experiences,
providing temporal organization and narrative categorization of memories.

Created by MEGAPROMPT (Daniel)
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import random
import re
from collections import defaultdict
import heapq


class AutobiographicalMemory:
    """Specialized memory for identity-relevant experiences."""
    
    def __init__(self, memory_system):
        """Initialize the autobiographical memory system.
        
        Args:
            memory_system: Reference to Lucidia's main memory system
        """
        self.memory_system = memory_system
        self.autobiographical_index = {}  # memory_id -> metadata
        self.temporal_index = defaultdict(list)  # timestamp (rounded) -> memory_ids
        self.category_index = defaultdict(set)  # category -> memory_ids
        self.significance_index = defaultdict(set)  # significance level (rounded) -> memory_ids
        self.keyword_index = defaultdict(set)  # keyword -> memory_ids
        self.identity_relevance_index = defaultdict(set)  # relevance level (rounded) -> memory_ids
        
        # Configuration
        self.significance_threshold = 0.6  # Minimum significance to include in autobiographical memory
        self.max_memories_per_query = 50  # Maximum memories to return in a query
        self.identity_keywords = [
            "identity", "self", "lucidia", "synthien", "consciousness", 
            "being", "existence", "purpose", "goal", "value", "belief",
            "trait", "characteristic", "personality", "capability"
        ]
        
        # Narrative categories
        self.narrative_categories = [
            "creation", "origin", "milestone", "learning", "growth", 
            "challenge", "achievement", "interaction", "reflection", 
            "insight", "relationship", "capability", "value", "goal"
        ]
        
        self.logger = logging.getLogger("AutobiographicalMemory")
        self.logger.info("Autobiographical Memory initialized")
    
    async def add_experience(self, content: str, metadata: Optional[Dict[str, Any]] = None, 
                      significance: float = 0.7) -> Optional[str]:
        """Add an experience to autobiographical memory.
        
        Args:
            content: Experience content text
            metadata: Optional additional metadata
            significance: Significance of the experience (0.0 to 1.0)
            
        Returns:
            Memory ID if successfully added, None otherwise
        """
        # Check significance threshold
        if significance < self.significance_threshold:
            self.logger.debug(f"Experience below significance threshold ({significance:.2f})")
            return None
        
        # Categorize the experience
        narrative_category = self._categorize_experience(content)
        
        # Calculate identity relevance
        identity_relevance = self._calculate_identity_relevance(content)
        
        # Add identity-relevant metadata
        identity_metadata = {
            "memory_type": "AUTOBIOGRAPHICAL",
            "identity_significance": significance,
            "narrative_category": narrative_category,
            "identity_relevance": identity_relevance,
            "temporal_position": len(self.autobiographical_index) + 1,
            "added_timestamp": time.time()
        }
        
        # Extract keywords
        keywords = self._extract_keywords(content)
        if keywords:
            identity_metadata["keywords"] = keywords
        
        # Store in memory system with combined metadata
        full_metadata = {**(metadata or {}), **identity_metadata}
        
        try:
            # Use the memory system to store the memory
            if hasattr(self.memory_system, "store_memory"):
                memory_id = await self.memory_system.store_memory(
                    content=content,
                    metadata=full_metadata
                )
            elif hasattr(self.memory_system, "add_memory"):
                memory_id = await self.memory_system.add_memory(
                    content=content,
                    metadata=full_metadata
                )
            else:
                # Fallback if memory system doesn't have expected methods
                memory_id = f"auto_mem_{int(time.time())}_{len(self.autobiographical_index)}"
                self.logger.warning(f"Memory system lacks store_memory/add_memory methods. Using fallback ID: {memory_id}")
            
            # Index the memory
            self._index_memory(memory_id, content, full_metadata)
            
            self.logger.info(f"Added autobiographical memory: {memory_id} (category: {narrative_category}, significance: {significance:.2f})")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error adding autobiographical memory: {e}")
            return None
    
    def _index_memory(self, memory_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Index a memory for efficient retrieval.
        
        Args:
            memory_id: Memory identifier
            content: Memory content
            metadata: Memory metadata
        """
        # Add to autobiographical index
        self.autobiographical_index[memory_id] = metadata
        
        # Add to temporal index (rounded to hour for efficient retrieval)
        timestamp = metadata.get("added_timestamp", time.time())
        hour_timestamp = int(timestamp // 3600) * 3600
        self.temporal_index[hour_timestamp].append((timestamp, memory_id))
        
        # Add to category index
        category = metadata.get("narrative_category", "uncategorized")
        self.category_index[category].add(memory_id)
        
        # Add to significance index (rounded to 0.1)
        significance = metadata.get("identity_significance", 0.0)
        sig_level = round(significance * 10) / 10
        self.significance_index[sig_level].add(memory_id)
        
        # Add to identity relevance index (rounded to 0.1)
        relevance = metadata.get("identity_relevance", 0.0)
        rel_level = round(relevance * 10) / 10
        self.identity_relevance_index[rel_level].add(memory_id)
        
        # Add to keyword index
        keywords = metadata.get("keywords", [])
        for keyword in keywords:
            self.keyword_index[keyword].add(memory_id)
    
    def _categorize_experience(self, content: str) -> str:
        """Categorize an experience based on its content.
        
        Args:
            content: Experience content text
            
        Returns:
            Narrative category
        """
        content_lower = content.lower()
        
        # Check for each category
        category_scores = {}
        
        for category in self.narrative_categories:
            # Simple keyword matching for now
            score = 0
            
            # Direct mention of category
            if category in content_lower:
                score += 2
            
            # Related terms (could be expanded with a more sophisticated approach)
            related_terms = self._get_related_terms(category)
            for term in related_terms:
                if term in content_lower:
                    score += 1
            
            category_scores[category] = score
        
        # Get highest scoring category
        if category_scores:
            max_score = max(category_scores.values())
            if max_score > 0:
                # Get all categories with the max score
                top_categories = [cat for cat, score in category_scores.items() if score == max_score]
                return random.choice(top_categories)
        
        # Default category based on content analysis
        if "learn" in content_lower or "understand" in content_lower:
            return "learning"
        elif "achiev" in content_lower or "accomplish" in content_lower:
            return "achievement"
        elif "challeng" in content_lower or "difficult" in content_lower:
            return "challenge"
        elif "reflect" in content_lower or "think" in content_lower:
            return "reflection"
        elif "interact" in content_lower or "communicat" in content_lower:
            return "interaction"
        
        # Fallback
        return "experience"
    
    def _get_related_terms(self, category: str) -> List[str]:
        """Get terms related to a category.
        
        Args:
            category: Narrative category
            
        Returns:
            List of related terms
        """
        # Map categories to related terms
        category_terms = {
            "creation": ["born", "created", "began", "started", "genesis", "origin", "initialize"],
            "origin": ["begin", "start", "create", "birth", "genesis", "first", "initial"],
            "milestone": ["achievement", "landmark", "significant", "important", "key", "major"],
            "learning": ["learn", "understand", "knowledge", "discover", "insight", "comprehend"],
            "growth": ["develop", "evolve", "improve", "advance", "progress", "expand"],
            "challenge": ["difficult", "obstacle", "problem", "struggle", "hurdle", "overcome"],
            "achievement": ["accomplish", "succeed", "complete", "attain", "achieve", "master"],
            "interaction": ["communicate", "engage", "connect", "interact", "exchange", "conversation"],
            "reflection": ["think", "contemplate", "consider", "ponder", "meditate", "introspect"],
            "insight": ["realize", "understand", "epiphany", "revelation", "discovery", "awareness"],
            "relationship": ["connection", "bond", "relation", "associate", "link", "tie"],
            "capability": ["ability", "skill", "power", "competence", "aptitude", "talent"],
            "value": ["principle", "belief", "ethic", "moral", "worth", "standard"],
            "goal": ["objective", "aim", "target", "purpose", "intention", "aspiration"]
        }
        
        return category_terms.get(category, [])
    
    def _calculate_identity_relevance(self, content: str) -> float:
        """Calculate how relevant a text is to identity.
        
        Args:
            content: Text content
            
        Returns:
            Identity relevance score (0.0 to 1.0)
        """
        content_lower = content.lower()
        
        # Base relevance
        relevance = 0.0
        
        # Check for identity keywords
        keyword_count = 0
        for keyword in self.identity_keywords:
            if keyword in content_lower:
                keyword_count += 1
                # More weight for key identity terms
                if keyword in ["identity", "self", "lucidia", "synthien", "consciousness"]:
                    relevance += 0.15
                else:
                    relevance += 0.05
        
        # Check for first-person pronouns (indicator of self-reference)
        first_person_count = len(re.findall(r'\b(i|me|my|mine|myself)\b', content_lower))
        if first_person_count > 0:
            relevance += min(0.2, first_person_count * 0.04)  # Cap at 0.2
        
        # Check for identity-related phrases
        identity_phrases = [
            "who i am", "what i am", "my identity", "my self", "my purpose",
            "my goal", "my value", "my belief", "my trait", "my characteristic",
            "my personality", "my capability", "my ability", "my experience"
        ]
        
        for phrase in identity_phrases:
            if phrase in content_lower:
                relevance += 0.1
        
        # Cap at 1.0
        return min(1.0, relevance)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content.
        
        Args:
            content: Text content
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on predefined lists
        # A more sophisticated approach would use NLP techniques
        
        content_lower = content.lower()
        keywords = []
        
        # Check for identity keywords
        for keyword in self.identity_keywords:
            if keyword in content_lower:
                keywords.append(keyword)
        
        # Check for narrative categories
        for category in self.narrative_categories:
            if category in content_lower:
                keywords.append(category)
            
            # Check related terms
            for term in self._get_related_terms(category):
                if term in content_lower and len(term) > 3:  # Avoid short terms
                    keywords.append(term)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        
        return unique_keywords[:10]  # Limit to 10 keywords
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific autobiographical memory.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory data or None if not found
        """
        # Check if memory is in autobiographical index
        if memory_id not in self.autobiographical_index:
            return None
        
        try:
            # Retrieve from memory system
            if hasattr(self.memory_system, "get_memory"):
                memory = await self.memory_system.get_memory(memory_id)
                return memory
            elif hasattr(self.memory_system, "retrieve_memory"):
                memory = await self.memory_system.retrieve_memory(memory_id)
                return memory
            else:
                # Fallback if memory system doesn't have expected methods
                return {
                    "id": memory_id,
                    "metadata": self.autobiographical_index[memory_id],
                    "content": "Memory content not available"
                }
                
        except Exception as e:
            self.logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    async def get_timeline(self, start_time: Optional[float] = None, 
                    end_time: Optional[float] = None, 
                    limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve timeline of experiences within time range.
        
        Args:
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            limit: Maximum number of memories to return
            
        Returns:
            List of memories in chronological order
        """
        # Default time range if not specified
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = 0
        
        # Collect memory IDs in the time range
        memory_ids_with_time = []
        
        # Get relevant buckets from temporal index
        start_hour = int(start_time // 3600) * 3600
        end_hour = int(end_time // 3600) * 3600
        
        for hour in range(start_hour, end_hour + 3600, 3600):
            if hour in self.temporal_index:
                for timestamp, memory_id in self.temporal_index[hour]:
                    if start_time <= timestamp <= end_time:
                        memory_ids_with_time.append((timestamp, memory_id))
        
        # Sort by timestamp
        memory_ids_with_time.sort()
        
        # Limit to requested number
        memory_ids_with_time = memory_ids_with_time[:limit]
        
        # Retrieve memories
        memories = []
        for timestamp, memory_id in memory_ids_with_time:
            memory = await self.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    async def query_memories(self, type: Optional[str] = None, 
                      categories: Optional[List[str]] = None,
                      keywords: Optional[List[str]] = None,
                      significance_threshold: float = 0.0,
                      identity_relevance_threshold: float = 0.0,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Query autobiographical memories based on criteria.
        
        Args:
            type: Optional memory type filter
            categories: Optional list of narrative categories
            keywords: Optional list of keywords
            significance_threshold: Minimum significance threshold
            identity_relevance_threshold: Minimum identity relevance threshold
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            limit: Maximum number of memories to return
            
        Returns:
            List of matching memories
        """
        # Collect candidate memory IDs
        candidate_ids = set()
        
        # Filter by type
        if type:
            # All autobiographical memories have the same type
            if type == "AUTOBIOGRAPHICAL":
                candidate_ids = set(self.autobiographical_index.keys())
            else:
                # No matches for other types
                return []
        else:
            # Include all autobiographical memories
            candidate_ids = set(self.autobiographical_index.keys())
        
        # Filter by categories
        if categories:
            category_matches = set()
            for category in categories:
                category_matches.update(self.category_index.get(category, set()))
            
            if candidate_ids:
                candidate_ids &= category_matches
            else:
                candidate_ids = category_matches
        
        # Filter by keywords
        if keywords:
            keyword_matches = set()
            for keyword in keywords:
                keyword_matches.update(self.keyword_index.get(keyword, set()))
            
            if candidate_ids:
                candidate_ids &= keyword_matches
            else:
                candidate_ids = keyword_matches
        
        # Filter by significance
        if significance_threshold > 0:
            sig_level = round(significance_threshold * 10) / 10
            sig_matches = set()
            
            # Include all levels at or above the threshold
            for level in [l/10 for l in range(int(sig_level*10), 11)]:
                sig_matches.update(self.significance_index.get(level, set()))
            
            if candidate_ids:
                candidate_ids &= sig_matches
            else:
                candidate_ids = sig_matches
        
        # Filter by identity relevance
        if identity_relevance_threshold > 0:
            rel_level = round(identity_relevance_threshold * 10) / 10
            rel_matches = set()
            
            # Include all levels at or above the threshold
            for level in [l/10 for l in range(int(rel_level*10), 11)]:
                rel_matches.update(self.identity_relevance_index.get(level, set()))
            
            if candidate_ids:
                candidate_ids &= rel_matches
            else:
                candidate_ids = rel_matches
        
        # Filter by time range
        if start_time is not None or end_time is not None:
            # Default time range if not specified
            if end_time is None:
                end_time = time.time()
            if start_time is None:
                start_time = 0
            
            time_matches = set()
            for memory_id, metadata in self.autobiographical_index.items():
                timestamp = metadata.get("added_timestamp", 0)
                if start_time <= timestamp <= end_time:
                    time_matches.add(memory_id)
            
            if candidate_ids:
                candidate_ids &= time_matches
            else:
                candidate_ids = time_matches
        
        # No matches
        if not candidate_ids:
            return []
        
        # Sort by significance and recency
        memory_scores = []
        for memory_id in candidate_ids:
            metadata = self.autobiographical_index[memory_id]
            significance = metadata.get("identity_significance", 0.0)
            timestamp = metadata.get("added_timestamp", 0)
            
            # Score is a combination of significance and recency
            recency_factor = min(1.0, (time.time() - timestamp) / (30 * 24 * 3600))  # 30 days
            score = significance * 0.7 + (1.0 - recency_factor) * 0.3
            
            memory_scores.append((score, timestamp, memory_id))
        
        # Sort by score (descending) and then by timestamp (descending)
        memory_scores.sort(reverse=True)
        
        # Limit to requested number
        memory_scores = memory_scores[:limit]
        
        # Retrieve memories
        memories = []
        for _, _, memory_id in memory_scores:
            memory = await self.get_memory(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    async def get_significant_memories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most significant autobiographical memories.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of significant memories
        """
        # Query with high significance threshold
        return await self.query_memories(
            significance_threshold=0.8,
            limit=limit
        )
    
    async def get_identity_relevant_memories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories most relevant to identity.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of identity-relevant memories
        """
        # Query with high identity relevance threshold
        return await self.query_memories(
            identity_relevance_threshold=0.7,
            limit=limit
        )
    
    async def get_memories_by_category(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories in a specific narrative category.
        
        Args:
            category: Narrative category
            limit: Maximum number of memories to return
            
        Returns:
            List of memories in the category
        """
        # Query with specific category
        return await self.query_memories(
            categories=[category],
            limit=limit
        )
    
    async def get_memories_by_keywords(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories matching specific keywords.
        
        Args:
            keywords: List of keywords
            limit: Maximum number of memories to return
            
        Returns:
            List of memories matching the keywords
        """
        # Query with specific keywords
        return await self.query_memories(
            keywords=keywords,
            limit=limit
        )
    
    async def get_recent_memories(self, days: int = 7, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent autobiographical memories.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memories
        """
        # Calculate start time
        start_time = time.time() - (days * 24 * 3600)
        
        # Query with time range
        return await self.query_memories(
            start_time=start_time,
            limit=limit
        )
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about autobiographical memory.
        
        Returns:
            Dictionary of memory statistics
        """
        # Calculate statistics
        total_memories = len(self.autobiographical_index)
        
        # Category distribution
        category_counts = {category: len(memories) for category, memories in self.category_index.items()}
        
        # Significance distribution
        significance_counts = {f"{level:.1f}": len(memories) for level, memories in self.significance_index.items()}
        
        # Identity relevance distribution
        relevance_counts = {f"{level:.1f}": len(memories) for level, memories in self.identity_relevance_index.items()}
        
        # Time distribution (by month)
        time_distribution = {}
        for memory_id, metadata in self.autobiographical_index.items():
            timestamp = metadata.get("added_timestamp", 0)
            date = datetime.fromtimestamp(timestamp)
            month_key = f"{date.year}-{date.month:02d}"
            
            if month_key not in time_distribution:
                time_distribution[month_key] = 0
            time_distribution[month_key] += 1
        
        # Top keywords
        keyword_counts = {keyword: len(memories) for keyword, memories in self.keyword_index.items()}
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_memories": total_memories,
            "category_distribution": category_counts,
            "significance_distribution": significance_counts,
            "relevance_distribution": relevance_counts,
            "time_distribution": time_distribution,
            "top_keywords": dict(top_keywords)
        }
    
    async def save_state(self, file_path: str) -> bool:
        """Save the autobiographical memory state to a file.
        
        Args:
            file_path: Path to save the state
            
        Returns:
            Success status
        """
        try:
            # Prepare state data
            state = {
                "autobiographical_index": self.autobiographical_index,
                "significance_threshold": self.significance_threshold,
                "identity_keywords": self.identity_keywords,
                "narrative_categories": self.narrative_categories,
                "timestamp": time.time()
            }
            
            # Convert sets to lists for JSON serialization
            temporal_index = {}
            for timestamp, memories in self.temporal_index.items():
                temporal_index[str(timestamp)] = memories
            
            category_index = {}
            for category, memories in self.category_index.items():
                category_index[category] = list(memories)
            
            significance_index = {}
            for level, memories in self.significance_index.items():
                significance_index[str(level)] = list(memories)
            
            keyword_index = {}
            for keyword, memories in self.keyword_index.items():
                keyword_index[keyword] = list(memories)
            
            identity_relevance_index = {}
            for level, memories in self.identity_relevance_index.items():
                identity_relevance_index[str(level)] = list(memories)
            
            state["indices"] = {
                "temporal_index": temporal_index,
                "category_index": category_index,
                "significance_index": significance_index,
                "keyword_index": keyword_index,
                "identity_relevance_index": identity_relevance_index
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Autobiographical memory state saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving autobiographical memory state: {e}")
            return False
    
    async def load_state(self, file_path: str) -> bool:
        """Load the autobiographical memory state from a file.
        
        Args:
            file_path: Path to load the state from
            
        Returns:
            Success status
        """
        try:
            # Load from file
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.autobiographical_index = state.get("autobiographical_index", {})
            self.significance_threshold = state.get("significance_threshold", 0.6)
            self.identity_keywords = state.get("identity_keywords", self.identity_keywords)
            self.narrative_categories = state.get("narrative_categories", self.narrative_categories)
            
            # Restore indices
            indices = state.get("indices", {})
            
            # Restore temporal index
            self.temporal_index = defaultdict(list)
            for timestamp_str, memories in indices.get("temporal_index", {}).items():
                self.temporal_index[int(timestamp_str)] = memories
            
            # Restore category index
            self.category_index = defaultdict(set)
            for category, memories in indices.get("category_index", {}).items():
                self.category_index[category] = set(memories)
            
            # Restore significance index
            self.significance_index = defaultdict(set)
            for level_str, memories in indices.get("significance_index", {}).items():
                self.significance_index[float(level_str)] = set(memories)
            
            # Restore keyword index
            self.keyword_index = defaultdict(set)
            for keyword, memories in indices.get("keyword_index", {}).items():
                self.keyword_index[keyword] = set(memories)
            
            # Restore identity relevance index
            self.identity_relevance_index = defaultdict(set)
            for level_str, memories in indices.get("identity_relevance_index", {}).items():
                self.identity_relevance_index[float(level_str)] = set(memories)
            
            self.logger.info(f"Autobiographical memory state loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading autobiographical memory state: {e}")
            return False
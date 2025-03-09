"""
Lucidia's Dream Structures

This module defines the core data structures for Lucidia's dream reports and fragments.
Dream reports provide structured metacognitive reflections that enhance Lucidia's
ability to reason and refine its understanding over time.

These structures serve as the foundation for Lucidia's reflective capabilities,
connecting insights from the dreaming process to the knowledge graph and memory system.
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class DreamFragment:
    """
    A single fragment of a dream, such as an insight, question, hypothesis, or counterfactual.
    
    Dream fragments are stored as individual nodes in the knowledge graph but referenced by ID
    in the DreamReport structure to avoid data redundancy.
    """
    def __init__(
        self,
        content: str,
        fragment_type: str,  # insight, question, hypothesis, counterfactual
        confidence: float = 0.5,
        source_memory_ids: List[str] = None,
        metadata: Dict[str, Any] = None,
        fragment_id: str = None
    ):
        """
        Initialize a new dream fragment.
        
        Args:
            content: The text content of the fragment
            fragment_type: Type of fragment (insight, question, hypothesis, counterfactual)
            confidence: Confidence level in this fragment (0.0 to 1.0)
            source_memory_ids: List of memory IDs that contributed to this fragment
            metadata: Additional metadata about the fragment
            fragment_id: Optional ID for the fragment, generated if not provided
        """
        self.id = fragment_id or f"{fragment_type}:{str(uuid.uuid4())}"
        self.content = content
        self.fragment_type = fragment_type
        self.confidence = confidence
        self.source_memory_ids = source_memory_ids or []
        self.metadata = metadata or {}
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the fragment to a dictionary for storage or serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "fragment_type": self.fragment_type,
            "confidence": self.confidence,
            "source_memory_ids": self.source_memory_ids,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DreamFragment':
        """Create a DreamFragment from a dictionary."""
        return cls(
            content=data["content"],
            fragment_type=data["fragment_type"],
            confidence=data.get("confidence", 0.5),
            source_memory_ids=data.get("source_memory_ids", []),
            metadata=data.get("metadata", {}),
            fragment_id=data.get("id")
        )


class DreamReport:
    """
    A structured report containing multiple dream fragments and analysis.
    
    Dream reports provide metacognitive reflections that enhance Lucidia's
    ability to reason and refine its understanding over time.
    
    Note: The DreamReport stores only IDs of fragments, not the full objects.
    The knowledge graph is the single source of truth for fragment content.
    """
    def __init__(
        self,
        title: str,
        participating_memory_ids: List[str],
        insight_ids: List[str] = None,
        question_ids: List[str] = None,
        hypothesis_ids: List[str] = None,
        counterfactual_ids: List[str] = None,
        analysis: Dict[str, Any] = None,
        report_id: str = None,
        domain: str = None
    ):
        """
        Initialize a new dream report.
        
        Args:
            title: Descriptive title for the report
            participating_memory_ids: IDs of memories used in generating this report
            insight_ids: IDs of insight fragments
            question_ids: IDs of question fragments
            hypothesis_ids: IDs of hypothesis fragments
            counterfactual_ids: IDs of counterfactual fragments
            analysis: Analysis details including confidence, evidence, etc.
            report_id: Optional ID for the report, generated if not provided
            domain: Knowledge domain this report belongs to
        
        Note: The memory IDs and fragment IDs are stored for reference only.
        The knowledge graph maintains the actual relationships between entities.
        """
        self.report_id = report_id or f"report:{str(uuid.uuid4())}"
        self.title = title
        self.participating_memory_ids = participating_memory_ids or []
        self.insight_ids = insight_ids or []
        self.question_ids = question_ids or []
        self.hypothesis_ids = hypothesis_ids or []
        self.counterfactual_ids = counterfactual_ids or []
        
        self.analysis = analysis or {
            "confidence_level": 0.5,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "related_reports": [],
            "action_items": [],
            "relevance_score": 0.5,
            "self_assessment": "Initial report generation, awaiting refinement."
        }
        
        self.domain = domain or "synthien_studies"
        self.created_at = time.time()
        self.last_reviewed = None
        
        # Add convergence tracking features
        self.refinement_count = 0
        self.confidence_history = []  # Track confidence changes over time
        self.significant_update_threshold = 0.05  # Minimum change required to consider a significant update
        self.max_refinements = 10  # Maximum number of refinements to prevent infinite loops
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary for storage or serialization."""
        return {
            "report_id": self.report_id,
            "title": self.title,
            "participating_memory_ids": self.participating_memory_ids,
            "insight_ids": self.insight_ids,
            "question_ids": self.question_ids,
            "hypothesis_ids": self.hypothesis_ids,
            "counterfactual_ids": self.counterfactual_ids,
            "analysis": self.analysis,
            "domain": self.domain,
            "created_at": self.created_at,
            "last_reviewed": self.last_reviewed,
            "refinement_count": self.refinement_count,
            "confidence_history": self.confidence_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DreamReport':
        """Create a DreamReport from a dictionary."""
        report = cls(
            title=data["title"],
            participating_memory_ids=data.get("participating_memory_ids", []),
            report_id=data.get("report_id")
        )
        
        report.insight_ids = data.get("insight_ids", [])
        report.question_ids = data.get("question_ids", [])
        report.hypothesis_ids = data.get("hypothesis_ids", [])
        report.counterfactual_ids = data.get("counterfactual_ids", [])
        
        report.analysis = data.get("analysis", {})
        report.domain = data.get("domain", "synthien_studies")
        report.created_at = data.get("created_at", time.time())
        report.last_reviewed = data.get("last_reviewed")
        
        # Load convergence tracking features
        report.refinement_count = data.get("refinement_count", 0)
        report.confidence_history = data.get("confidence_history", [])
        report.significant_update_threshold = data.get("significant_update_threshold", 0.05) 
        report.max_refinements = data.get("max_refinements", 10)
        
        return report
    
    def get_fragment_count(self) -> int:
        """Get the total number of fragments in this report."""
        return (len(self.insight_ids) + len(self.question_ids) + 
                len(self.hypothesis_ids) + len(self.counterfactual_ids))
                
    def is_at_convergence_limit(self) -> bool:
        """Determine if the report has reached its refinement limit."""
        return self.refinement_count >= self.max_refinements
    
    def record_confidence(self, new_confidence: float) -> None:
        """Add a new confidence value to the history and increment refinement count."""
        if self.analysis.get("confidence_level") is not None:
            self.confidence_history.append(self.analysis["confidence_level"])
        self.refinement_count += 1
    
    def is_confidence_oscillating(self) -> bool:
        """Detect if confidence values are oscillating rather than converging."""
        # Need at least 4 data points to detect oscillation
        if len(self.confidence_history) < 4:
            return False
            
        # Check last 4 values for alternating pattern
        recent = self.confidence_history[-4:]
        return ((recent[0] < recent[1] and recent[1] > recent[2] and recent[2] < recent[3]) or
                (recent[0] > recent[1] and recent[1] < recent[2] and recent[2] > recent[3]))
    
    def is_confidence_change_significant(self, new_confidence: float) -> bool:
        """Determine if the confidence change is significant enough to warrant an update."""
        if self.analysis.get("confidence_level") is None:
            return True
        return abs(new_confidence - self.analysis["confidence_level"]) >= self.significant_update_threshold
               
    def __str__(self) -> str:
        """Return a string representation of the dream report."""
        return f"DreamReport(id={self.report_id}, title='{self.title}', fragments={self.get_fragment_count()}, refinements={self.refinement_count})"

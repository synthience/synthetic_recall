#!/usr/bin/env python

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from .titans_variants import TitansVariantType

logger = logging.getLogger(__name__)

class VariantSelector:
    """
    Selects the optimal Titans variant (MAC, MAG, MAL, NONE) based on context.
    
    This selector uses a rule-based approach to determine which variant is most
    appropriate for a given context, considering factors such as:
    - LLM guidance (highest priority)
    - Metadata about the task and content
    - Neural Memory performance metrics (surprise level)
    - Query content keywords
    - Default fallback rules
    """
    
    def __init__(self, high_surprise_threshold=0.5, low_surprise_threshold=0.1):
        """
        Initialize the variant selector with configurable thresholds.
        
        Args:
            high_surprise_threshold: Threshold above which surprise is considered high
            low_surprise_threshold: Threshold below which surprise is considered low
        """
        self.high_surprise_threshold = high_surprise_threshold
        self.low_surprise_threshold = low_surprise_threshold
        logger.info(f"VariantSelector initialized with thresholds: High={high_surprise_threshold}, Low={low_surprise_threshold}")

    def select_variant(
        self,
        query: Optional[str],
        metadata: Dict[str, Any],
        nm_performance: Dict[str, Any],
        llm_variant_hint: Optional[str] = None
    ) -> Tuple[TitansVariantType, str, List[str]]:
        """
        Selects the best variant based on context, performance, and LLM hints.
        
        Args:
            query: The user query or input text
            metadata: Dictionary containing metadata about the task/query
            nm_performance: Dictionary with Neural Memory performance metrics
                            Expected keys: avg_loss, avg_grad_norm, sample_count, 
                            trend_increasing (optional), trend_decreasing (optional)
            llm_variant_hint: Optional variant suggestion from an LLM
            
        Returns:
            Tuple of (selected_variant_type, reason, decision_trace)
        """
        decision_trace = []
        selected_variant = TitansVariantType.MAC  # Default to MAC
        decision_reason = "Default"
        
        # Validate inputs
        if not isinstance(metadata, dict):
            metadata = {}
        if not isinstance(nm_performance, dict):
            nm_performance = {}
            
        # Store incoming state in trace
        perf_str = f"Loss: {nm_performance.get('avg_loss', 'N/A')}, GradNorm: {nm_performance.get('avg_grad_norm', 'N/A')}"
        sample_count = nm_performance.get('sample_count', 0)
        if sample_count > 0:
            perf_str += f", Samples: {sample_count}"
        decision_trace.append(f"Input metrics: {perf_str}")
            
        # 1. Check LLM Hint (Highest Priority)
        if llm_variant_hint:
            decision_trace.append(f"LLM provided variant hint: {llm_variant_hint}")
            try:
                # Try to match the hint to a valid enum value
                hinted_variant = TitansVariantType(llm_variant_hint.upper())
                logger.info(f"Using LLM variant hint: {hinted_variant.value}")
                decision_trace.append(f"Using LLM hint: {hinted_variant.value}")
                return hinted_variant, f"LLM Hint ({hinted_variant.value})", decision_trace
            except ValueError:
                logger.warning(f"Invalid LLM variant hint received: '{llm_variant_hint}'. Ignoring.")
                decision_trace.append(f"Invalid LLM hint ignored: {llm_variant_hint}")

        # 2. Check Metadata Hints (Task Type)
        task_type = metadata.get("task_type", "").lower()
        decision_trace.append(f"Task type: {task_type or 'not specified'}")
        
        if task_type == "summarize":
            decision_trace.append(f"Task type 'summarize' matches MAC variant")
            return TitansVariantType.MAC, "Task Type (Summarize -> MAC)", decision_trace
        if task_type in ["causal_reasoning", "explanation"]:
            decision_trace.append(f"Task type '{task_type}' matches MAL variant")
            return TitansVariantType.MAL, f"Task Type ({task_type} -> MAL)", decision_trace
        if task_type in ["background", "low_priority"]:
            decision_trace.append(f"Task type '{task_type}' matches NONE variant")
            return TitansVariantType.NONE, f"Task Type ({task_type} -> NONE)", decision_trace

        # 3. Enhanced Performance Metrics Analysis
        surprise_metric = None
        avg_loss = nm_performance.get("avg_loss")
        avg_grad = nm_performance.get("avg_grad_norm")
        sample_count = nm_performance.get("sample_count", 0)
        trend_increasing = nm_performance.get("trend_increasing", False)
        trend_decreasing = nm_performance.get("trend_decreasing", False)
        
        # Only consider performance metrics if we have enough samples
        if isinstance(avg_loss, (int, float)) and isinstance(avg_grad, (int, float)) and sample_count >= 3:
            # Use weighted combination of loss and normalized gradient
            # Loss is typically smaller (0-2 range) while grad can be 1-50+
            # Normalize gradient to a similar scale as loss for better comparison
            norm_factor = 10.0  # Empirically determined scaling factor
            surprise_metric = (avg_loss + min(avg_grad / norm_factor, 2.0)) / 2.0  # Cap normalized grad contribution
            decision_trace.append(f"Calculated surprise metric: {surprise_metric:.3f} from {sample_count} samples")
            
            # 3.1 Trend analysis - increasing surprise suggests switching to MAG for adaptive learning
            # Make the selector more proactive - respond to any significant increasing trend
            # regardless of how close the surprise is to the high threshold
            if trend_increasing and nm_performance.get("trend_slope", 0.0) > 0.05:  # Use explicit threshold
                decision_trace.append(f"Increasing surprise trend detected (slope={nm_performance.get('trend_slope', 0.0):.4f})")
                decision_trace.append(f"Increasing surprise suggests MAG variant for adaptive learning")
                return TitansVariantType.MAG, f"Performance (Increasing Surprise Trend -> MAG)", decision_trace
                
            # 3.2 Consistently high surprise
            if surprise_metric > self.high_surprise_threshold:
                # High surprise -> Adapt learning parameters more aggressively
                decision_trace.append(f"High surprise ({surprise_metric:.3f} > {self.high_surprise_threshold}) suggests MAG variant")
                return TitansVariantType.MAG, f"Performance (High Surprise {surprise_metric:.3f} -> MAG)", decision_trace
                
            # 3.3 Trend analysis - decreasing surprise with moderate values suggests MAL for refinement
            # Ensure we're checking the actual trend slope matches the flag
            if trend_decreasing and nm_performance.get("trend_slope", 0.0) < -0.05 and \
               self.low_surprise_threshold < surprise_metric < self.high_surprise_threshold:
                decision_trace.append(f"Decreasing surprise trend with moderate values detected (slope={nm_performance.get('trend_slope', 0.0):.4f})")
                decision_trace.append(f"Moderate decreasing surprise suggests MAL variant for knowledge refinement")
                return TitansVariantType.MAL, f"Performance (Decreasing Moderate Surprise -> MAL)", decision_trace
        else:
            # Handle the case where metrics are missing or invalid
            if sample_count < 3:
                logger.info(f"Insufficient performance samples ({sample_count}) to make data-driven decision")
                decision_trace.append(f"Skipping performance check due to insufficient samples ({sample_count} < 3)")
            else:
                logger.warning(f"Could not calculate surprise metric due to invalid performance data: Loss={avg_loss}, Grad={avg_grad}")
                decision_trace.append(f"Skipping performance check due to invalid data (Loss: {avg_loss}, Grad: {avg_grad})")

        # 4. Check Query Keywords (Examples)
        if query:
            query_lower = query.lower()
            decision_trace.append(f"Analyzing query keywords: '{query_lower[:50]}...'")
            
            if any(phrase in query_lower for phrase in ["explain why", "cause of", "reason for", "because"]):
                decision_trace.append(f"Detected causal reasoning keywords -> MAL")
                return TitansVariantType.MAL, "Query Keyword (Causal reasoning -> MAL)", decision_trace
                
            if any(phrase in query_lower for phrase in ["remember when", "recall events", "sequence", "timeline", "history of"]):
                decision_trace.append(f"Detected recall/sequence keywords -> MAC")
                return TitansVariantType.MAC, "Query Keyword (Recall/Sequence -> MAC)", decision_trace
                
            if any(phrase in query_lower for phrase in ["adapt", "learn", "adjust to", "handle new"]):
                decision_trace.append(f"Detected adaptive keywords -> MAG")
                return TitansVariantType.MAG, "Query Keyword (Adaptation -> MAG)", decision_trace

        # 5. Default Logic based on Surprise
        if surprise_metric is not None:
            if surprise_metric < self.low_surprise_threshold:
                # Low surprise -> be efficient
                decision_trace.append(f"Low surprise ({surprise_metric:.3f} < {self.low_surprise_threshold}) suggests NONE variant")
                return TitansVariantType.NONE, f"Performance (Low Surprise {surprise_metric:.3f} -> NONE)", decision_trace
            else:
                # Moderate surprise or default case
                decision_trace.append(f"Moderate surprise ({surprise_metric:.3f}) suggests MAC variant")
                return TitansVariantType.MAC, f"Default (Moderate Surprise {surprise_metric:.3f} -> MAC)", decision_trace
        else:
            # No valid surprise metric available
            decision_trace.append("No valid surprise metric available, using fallback decision")
            decision_trace.append("Using final fallback to MAC variant")
            return TitansVariantType.MAC, "Final Fallback -> MAC", decision_trace

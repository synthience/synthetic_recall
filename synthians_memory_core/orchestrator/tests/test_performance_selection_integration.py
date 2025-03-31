#!/usr/bin/env python

import pytest
import sys
import os
import json
import numpy as np

# Add the necessary path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import directly (not using relative imports)
from synthians_memory_core.orchestrator.variant_selector import VariantSelector
from synthians_memory_core.orchestrator.titans_variants import TitansVariantType

# Test data
HIGH_SURPRISE_METRICS = {
    "avg_loss": 0.85,      # High value
    "avg_grad_norm": 5.5,  # High value
    "sample_count": 5,
    "trend_increasing": False,
    "trend_decreasing": False,
    "trend_slope": 0.02    # Small slope, not significant
}

LOW_SURPRISE_METRICS = {
    "avg_loss": 0.05,      # Low value
    "avg_grad_norm": 0.08, # Low value
    "sample_count": 5,
    "trend_increasing": False,
    "trend_decreasing": False,
    "trend_slope": 0.01    # Small slope, not significant
}

INCREASING_TREND_METRICS = {
    "avg_loss": 0.4,       # Moderate value
    "avg_grad_norm": 2.0,  # Moderate value
    "sample_count": 5,
    "trend_increasing": True,
    "trend_decreasing": False,
    "trend_slope": 0.1     # Positive slope
}

DECREASING_TREND_METRICS = {
    "avg_loss": 0.3,       # Moderate value
    "avg_grad_norm": 1.5,  # Moderate value
    "sample_count": 5,
    "trend_increasing": False,
    "trend_decreasing": True,
    "trend_slope": -0.1    # Negative slope
}

# Test fixtures
@pytest.fixture
def variant_selector():
    return VariantSelector(high_surprise_threshold=0.5, low_surprise_threshold=0.1)

@pytest.fixture
def basic_metadata():
    return {
        "type": "memory",
        "tags": [],
        "complexity": 0.5
    }

# Test cases
def test_variant_selector_high_surprise(variant_selector, basic_metadata):
    """Test that VariantSelector selects MAG for high surprise metrics"""
    query = "Test high surprise"
    
    # Call selector
    selected_variant, reason, decision_trace = variant_selector.select_variant(
        query=query,
        metadata=basic_metadata,
        nm_performance=HIGH_SURPRISE_METRICS,
        llm_variant_hint=None
    )
    
    # Assertions
    assert selected_variant == TitansVariantType.MAG, \
        f"Expected MAG, got {selected_variant}"
    assert "High Surprise" in reason, \
        f"Expected reason to mention high surprise, got: {reason}"
    
    print(f"Selected: {selected_variant}, Reason: {reason}")
    print(f"Decision Trace: {json.dumps(decision_trace, indent=2)}")

def test_variant_selector_low_surprise(variant_selector, basic_metadata):
    """Test that VariantSelector selects NONE for low surprise metrics"""
    query = "Test low surprise"
    
    # Call selector
    selected_variant, reason, decision_trace = variant_selector.select_variant(
        query=query,
        metadata=basic_metadata,
        nm_performance=LOW_SURPRISE_METRICS,
        llm_variant_hint=None
    )
    
    # Assertions
    assert selected_variant == TitansVariantType.NONE, \
        f"Expected NONE, got {selected_variant}"
    assert "Low Surprise" in reason, \
        f"Expected reason to mention low surprise, got: {reason}"
    
    print(f"Selected: {selected_variant}, Reason: {reason}")
    print(f"Decision Trace: {json.dumps(decision_trace, indent=2)}")

def test_variant_selector_increasing_trend(variant_selector, basic_metadata):
    """Test that VariantSelector selects MAG for increasing surprise trend"""
    query = "Test increasing trend"
    
    # Call selector
    selected_variant, reason, decision_trace = variant_selector.select_variant(
        query=query,
        metadata=basic_metadata,
        nm_performance=INCREASING_TREND_METRICS,
        llm_variant_hint=None
    )
    
    # Assertions
    assert selected_variant == TitansVariantType.MAG, \
        f"Expected MAG, got {selected_variant}"
    assert "Increasing" in reason, \
        f"Expected reason to mention increasing trend, got: {reason}"
    
    print(f"Selected: {selected_variant}, Reason: {reason}")
    print(f"Decision Trace: {json.dumps(decision_trace, indent=2)}")

def test_variant_selector_decreasing_trend(variant_selector, basic_metadata):
    """Test that VariantSelector selects MAL for decreasing surprise trend"""
    query = "Test decreasing trend"
    
    # Call selector
    selected_variant, reason, decision_trace = variant_selector.select_variant(
        query=query,
        metadata=basic_metadata,
        nm_performance=DECREASING_TREND_METRICS,
        llm_variant_hint=None
    )
    
    # Assertions
    assert selected_variant == TitansVariantType.MAL, \
        f"Expected MAL, got {selected_variant}"
    assert "Decreasing" in reason, \
        f"Expected reason to mention decreasing trend, got: {reason}"
    
    print(f"Selected: {selected_variant}, Reason: {reason}")
    print(f"Decision Trace: {json.dumps(decision_trace, indent=2)}")

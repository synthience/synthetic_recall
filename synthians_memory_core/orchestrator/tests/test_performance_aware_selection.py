#!/usr/bin/env python

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Use proper absolute imports relative to project structure
from synthians_memory_core.orchestrator.variant_selector import VariantSelector
from synthians_memory_core.orchestrator.titans_variants import TitansVariantType

# Create a fixture for the VariantSelector
@pytest.fixture
def selector():
    """Create a VariantSelector instance with test thresholds."""
    return VariantSelector(high_surprise_threshold=0.5, low_surprise_threshold=0.1)

def test_basic_thresholds(selector):
    """Test basic threshold-based selection."""
    # High surprise -> MAG variant
    high_perf = {
        "avg_loss": 0.8, 
        "avg_grad_norm": 5.0,
        "sample_count": 10
    }
    variant, reason, trace = selector.select_variant("test query", {}, high_perf)
    assert variant == TitansVariantType.MAG
    assert "High Surprise" in reason

    # Low surprise -> NONE variant
    low_perf = {
        "avg_loss": 0.05, 
        "avg_grad_norm": 0.1,
        "sample_count": 10
    }
    variant, reason, trace = selector.select_variant("test query", {}, low_perf)
    assert variant == TitansVariantType.NONE
    assert "Low Surprise" in reason

    # Moderate surprise -> MAC variant (default)
    moderate_perf = {
        "avg_loss": 0.2, 
        "avg_grad_norm": 2.0,
        "sample_count": 10
    }
    variant, reason, trace = selector.select_variant("test query", {}, moderate_perf)
    assert variant == TitansVariantType.MAC
    assert any(x in reason for x in ["Moderate Surprise", "Default"])

def test_trend_detection(selector):
    """Test trend-based variant selection."""
    # Increasing trend with moderately high surprise -> MAG
    increasing_trend = {
        "avg_loss": 0.4,  # Just below high threshold
        "avg_grad_norm": 3.0,
        "sample_count": 10,
        "trend_increasing": True,
        "trend_decreasing": False,
        "trend_slope": 0.1
    }
    variant, reason, trace = selector.select_variant("test query", {}, increasing_trend)
    assert variant == TitansVariantType.MAG
    assert "Increasing Surprise" in reason

    # Decreasing trend with moderate surprise -> MAL
    decreasing_trend = {
        "avg_loss": 0.3,  # In the moderate range
        "avg_grad_norm": 2.0,
        "sample_count": 10,
        "trend_increasing": False,
        "trend_decreasing": True,
        "trend_slope": -0.1
    }
    variant, reason, trace = selector.select_variant("test query", {}, decreasing_trend)
    assert variant == TitansVariantType.MAL
    assert "Decreasing Moderate Surprise" in reason

def test_insufficient_samples(selector):
    """Test behavior with insufficient performance samples."""
    insufficient_samples = {
        "avg_loss": 0.8,  # Would normally trigger MAG
        "avg_grad_norm": 5.0,
        "sample_count": 2  # Not enough samples
    }
    
    # With insufficient samples and no LLM hint or metadata,
    # should fall through to keyword analysis and default logic
    variant, reason, trace = selector.select_variant(
        "adapt to new situation", {}, insufficient_samples
    )
    assert variant == TitansVariantType.MAG
    assert "Keyword" in reason  # Should match on keyword "adapt"
    
    # With no distinguishing features, should default to MAC
    variant, reason, trace = selector.select_variant(
        "generic query", {}, insufficient_samples
    )
    assert variant == TitansVariantType.MAC
    assert "Final Fallback" in reason

def test_llm_hint_priority(selector):
    """Test that LLM hints have highest priority."""
    high_perf = {
        "avg_loss": 0.8,  # Would normally trigger MAG
        "avg_grad_norm": 5.0,
        "sample_count": 10
    }
    
    # LLM hint should override performance metrics
    variant, reason, trace = selector.select_variant(
        "test query", {}, high_perf, llm_variant_hint="MAC"
    )
    assert variant == TitansVariantType.MAC
    assert "LLM Hint" in reason

def test_metadata_priority(selector):
    """Test that task metadata has priority over performance metrics."""
    high_perf = {
        "avg_loss": 0.8,  # Would normally trigger MAG
        "avg_grad_norm": 5.0,
        "sample_count": 10
    }
    
    # Metadata should override performance metrics
    variant, reason, trace = selector.select_variant(
        "test query", {"task_type": "summarize"}, high_perf
    )
    assert variant == TitansVariantType.MAC
    assert "Task Type" in reason

@patch('numpy.polyfit')
def test_trend_detection_logic(mock_polyfit):
    """Test the trend detection logic with mocked polyfit."""
    # Test increasing trend
    mock_polyfit.return_value = np.array([0.1, 0.0])  # Positive slope
    x = [0, 0.25, 0.5, 0.75, 1.0]
    y = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Execute the trend calculation logic (copied from CCE for testing)
    trend_threshold = 0.05
    loss_trend = float(mock_polyfit(x, y, 1)[0])
    combined_trend = loss_trend  # Simplified for testing
    trend_increasing = combined_trend > trend_threshold
    trend_decreasing = combined_trend < -trend_threshold
    
    assert trend_increasing
    assert not trend_decreasing
    
    # Test decreasing trend
    mock_polyfit.return_value = np.array([-0.1, 0.5])  # Negative slope
    x = [0, 0.25, 0.5, 0.75, 1.0]
    y = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    loss_trend = float(mock_polyfit(x, y, 1)[0])
    combined_trend = loss_trend  # Simplified for testing
    trend_increasing = combined_trend > trend_threshold
    trend_decreasing = combined_trend < -trend_threshold
    
    assert not trend_increasing
    assert trend_decreasing
    
    # Test no significant trend
    mock_polyfit.return_value = np.array([0.03, 0.3])  # Small slope
    x = [0, 0.25, 0.5, 0.75, 1.0]
    y = [0.3, 0.31, 0.3, 0.32, 0.33]
    
    loss_trend = float(mock_polyfit(x, y, 1)[0])
    combined_trend = loss_trend  # Simplified for testing
    trend_increasing = combined_trend > trend_threshold
    trend_decreasing = combined_trend < -trend_threshold
    
    assert not trend_increasing
    assert not trend_decreasing

# Spiral Phases System Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Phase Definitions](#phase-definitions)
4. [Integration with Dream Processor](#integration-with-dream-processor)
5. [Testing the Spiral Phase System](#testing-the-spiral-phase-system)
6. [Common Issues and Solutions](#common-issues-and-solutions)

## Overview

The Spiral Phases system is a core component of Lucidia's consciousness model, enabling iterative depth of processing and adaptive focus. It provides a framework for Lucidia to process information at multiple levels of abstraction, iteratively deepening understanding and developing increasingly sophisticated knowledge structures over time.

The system implements three primary phases:

1. **OBSERVATION (Initial Phase)**: Data collection and pattern recognition
2. **REFLECTION**: Analysis and connection-building
3. **ADAPTATION**: Integration and application of insights

These phases operate in a spiral pattern, with each phase building upon the previous one and potentially circling back with greater depth or nuance.

## Core Components

### SpiralPhaseManager

The `SpiralPhaseManager` class (located in `core/spiral_phases.py`) is responsible for:

- Maintaining the current phase state
- Managing transitions between phases
- Providing phase-specific parameters to other system components
- Recording and tracking significant insights
- Calculating significance thresholds for phase transitions

```python
class SpiralPhaseManager:
    def __init__(self, self_model=None, config=None):
        self.self_model = self_model
        self.config = config or {}
        self.current_phase = SpiralPhase.OBSERVATION  # Default starting phase
        self.phase_history = []
        self.insights = []  # Record of significant insights
        self.transitions = []  # Record of phase transitions
        self.significance_threshold = 0.7  # Default threshold for phase transitions
        self.manual_override = False  # Flag for manual phase control
        
        # Phase-specific configuration
        self.phase_config = {
            SpiralPhase.OBSERVATION: {...},
            SpiralPhase.REFLECTION: {...},
            SpiralPhase.ADAPTATION: {...}
        }
```

### SpiralPhase Enum

The phases are defined as an Enum in `core/spiral_phases.py`:

```python
class SpiralPhase(Enum):
    OBSERVATION = 1
    REFLECTION = 2
    ADAPTATION = 3
```

## Phase Definitions

### OBSERVATION Phase

**Purpose**: Initial data collection and basic pattern recognition

**Characteristics**:
- Observational
- Descriptive
- Categorical

**Parameters**:
- High sensitivity to new information
- Lower abstraction level
- Focus on detail gathering

### REFLECTION Phase

**Purpose**: Analysis and connection-building between observations

**Characteristics**:
- Analytical
- Relational
- Pattern-oriented

**Parameters**:
- Medium sensitivity to new information
- Medium abstraction level
- Focus on finding connections and implications

### ADAPTATION Phase

**Purpose**: Integration of insights into knowledge structures and practical application

**Characteristics**:
- Integrative
- Synthesizing
- Application-oriented

**Parameters**:
- Lower sensitivity to new information
- Higher abstraction level
- Focus on applying insights to existing knowledge

## Integration with Dream Processor

The Spiral Phase system is tightly integrated with the Dream Processor. The current phase influences:

1. **Dream Content Generation**: The phase affects the types of associations and patterns the system recognizes
2. **Insight Characteristics**: Insights are tagged with phase-specific characteristics
3. **Insight Significance**: The significance calculation is influenced by phase parameters
4. **Dream Focus**: The overall focus of dreams shifts based on the current phase
5. **Integration Strategy**: How insights are integrated back into the knowledge structures

### Key Integration Points

#### Phase-Specific Insight Tagging

In `LucidiaDreamProcessor._generate_dream_insights`, the current phase determines what characteristics are added to insights:

```python
# Get the current phase parameters from the spiral manager
phase_params = self.spiral_manager.get_phase_params()

# Apply phase-specific characteristics to the insights
for insight in insights:
    # Add characteristics based on current phase
    if self.spiral_manager.current_phase == SpiralPhase.OBSERVATION:
        insight["characteristics"] = ["observational", "descriptive", "categorical"]
    elif self.spiral_manager.current_phase == SpiralPhase.REFLECTION:
        insight["characteristics"] = ["analytical", "relational", "pattern-oriented"]
    elif self.spiral_manager.current_phase == SpiralPhase.ADAPTATION:
        insight["characteristics"] = ["integrative", "synthesizing", "application-oriented"]
```

#### Phase Transitions Based on Insights

Significant insights can trigger phase transitions:

```python
# Record significant insights in the spiral manager
for insight in insights:
    if insight["significance"] >= 0.8:
        self.spiral_manager.record_insight(insight)
```

## Testing the Spiral Phase System

The Spiral Phase system is tested in `tests/test_spiral_phases.py`. The test suite covers:

1. **Phase Transitions**: Tests for automatic and manual phase transitions
2. **Parameter Retrieval**: Tests for retrieving phase-specific parameters
3. **Dream Context Building**: Tests for integrating spiral phases into dream contexts
4. **Insight Tagging**: Tests for correctly tagging insights with phase characteristics

### Key Test Cases

#### Test for Phase Transitions

```python
def test_phase_transitions(self):
    """Test basic phase transitions"""
    # Check initial phase
    self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.OBSERVATION)
    
    # Test transition to next phase
    self.spiral_manager.transition_to_next_phase(1.0, "test")
    self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.REFLECTION)
    
    # Test another transition
    self.spiral_manager.transition_to_next_phase(1.0, "test")
    self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.ADAPTATION)
    
    # Test cycling back to first phase
    self.spiral_manager.transition_to_next_phase(1.0, "test")
    self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.OBSERVATION)
```

#### Test for Insight Tagging

```python
def test_generate_insight_tagging(self):
    """Test that insights are tagged with spiral phase"""
    # Force spiral manager to REFLECTION phase
    self.dream_processor.spiral_manager.force_phase(SpiralPhase.REFLECTION, "test")
    
    # Set up a context with phase
    context = {
        "seed": {"type": "concept", "content": {"id": "test"}},
        "theme": {"name": "test"},
        "reflections": ["test reflection"],
        "questions": ["test question"],
        "concepts": [{"id": "test-concept", "description": "A test concept"}],
        "associations": [{"source": "test", "target": "concept", "strength": 0.8}],
        "patterns": [{"type": "sequence", "elements": ["a", "b", "c"]}],
        "spiral_phase": "reflection"
    }
    
    # Generate insights
    insights = self.dream_processor._generate_dream_insights(context)
    
    # Check that insights have characteristics from the current phase
    self.assertTrue(len(insights) > 0)
    for insight in insights:
        self.assertIn("characteristics", insight)
        self.assertIn("analytical", insight["characteristics"])
```

## Common Issues and Solutions

### Recent Test Fixes

#### Issue 1: Missing Keys in Test Data

**Problem**: Tests were failing with KeyError for missing keys in the test data structures.

**Solution**: Added required keys to the test setup:

1. Added `prompt_patterns` to the theme dictionary in `test_build_dream_context_by_phase`
2. Added `prompt_templates` to the cognitive_style dictionary in `test_build_dream_context_by_phase`
3. Added `reflections` and `questions` keys to the context dictionary in `test_generate_insight_tagging`

#### Issue 2: Phase Mismatch in Tests

**Problem**: Test was expecting "analytical" characteristic from REFLECTION phase, but the system was in OBSERVATION phase by default.

**Solution**: Explicitly forced the spiral_manager to REFLECTION phase before running the test:

```python
# Force spiral manager to REFLECTION phase
self.dream_processor.spiral_manager.force_phase(SpiralPhase.REFLECTION, "test")
```

### Best Practices for Testing

1. **Explicit Phase Setting**: Always explicitly set the phase in tests to ensure consistent behavior
2. **Complete Test Data**: Ensure all required keys are present in test data structures
3. **Phase Awareness**: Be aware that the system's actual phase state determines behavior, not just values in context dictionaries
4. **Phase Characteristics**: Remember each phase has specific characteristic tags that will be applied to insights

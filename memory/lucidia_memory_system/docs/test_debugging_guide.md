# Lucidia Test Debugging Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Common Test Issues](#common-test-issues)
3. [Test Case Study: Spiral Phase Tests](#test-case-study-spiral-phase-tests)
4. [Understanding Dream Processor Dependencies](#understanding-dream-processor-dependencies)
5. [Testing Best Practices](#testing-best-practices)

## Introduction

This document provides guidance on debugging and resolving issues in the Lucidia test suite. It focuses on understanding the key dependencies and assumptions made in the test code and how to address common failures.

## Common Test Issues

### 1. KeyError Exceptions

Occurs when accessing a dictionary key that doesn't exist in the actual implementation. Common causes:

- Test fixtures not matching the implementation's expected structure
- Changes to the implementation without updating test fixtures
- Missing required fields in test data

### 2. AssertionError

Occurs when a test assertion fails. Common causes:

- Expected values differ from actual implementation behavior
- State management issues (e.g., tests depending on specific state that isn't properly initialized)
- Phase-dependent behavior not accounting for the current phase

## Test Case Study: Spiral Phase Tests

The following examples illustrate recent test failures and their resolutions in the spiral phase integration tests.

### Issue 1: KeyError in test_build_dream_context_by_phase

**Problem**: Test was failing with:
```
KeyError: 'prompt_patterns'
```

**Root Cause**: The test was creating a theme dictionary without the `prompt_patterns` key, but the implementation in `_build_dream_context` expected this key to exist.

**Solution**: Added the missing key to the test fixture:

```python
"theme": {"name": "identity", "keywords": ["test"], "prompt_patterns": ["What is {0}?", "How does {0} relate to {1}?"]},
```

### Issue 2: KeyError in cognitive_style dictionary

**Problem**: Test was failing with:
```
KeyError: 'prompt_templates'
```

**Root Cause**: The test was creating a cognitive_style dictionary without the `prompt_templates` key, but the implementation expected this key to exist.

**Solution**: Added the missing key to the test fixture:

```python
"cognitive_style": {"name": "analytical", "description": "test", "prompt_templates": ["Analyze {0}", "What are the components of {0}?"]},
```

### Issue 3: Missing keys in context for test_generate_insight_tagging

**Problem**: Test was failing with:
```
KeyError: 'reflections'
```

**Root Cause**: The context dictionary being passed to `_generate_dream_insights` was missing the required `reflections` and `questions` keys.

**Solution**: Added the missing keys to the context dictionary:

```python
context = {
    "seed": {"type": "concept", "content": {"id": "test"}},
    "theme": {"name": "test"},
    "reflections": ["test reflection"],  # Added this key
    "questions": ["test question"],      # Added this key
    # ... other existing keys
}
```

### Issue 4: Phase characteristics mismatch in test_generate_insight_tagging

**Problem**: Test was failing with:
```
AssertionError: 'analytical' not found in ['observational', 'descriptive', 'categorical']
```

**Root Cause**: The test was expecting the `analytical` characteristic (from REFLECTION phase), but the spiral_manager was in OBSERVATION phase by default, which adds ['observational', 'descriptive', 'categorical'] characteristics instead.

**Solution**: Explicitly set the spiral_manager to REFLECTION phase before running the test:

```python
# Force spiral manager to REFLECTION phase
self.dream_processor.spiral_manager.force_phase(SpiralPhase.REFLECTION, "test")
```

## Understanding Dream Processor Dependencies

The Dream Processor has several key dependencies and state management considerations:

### 1. Spiral Phase Manager State

- The actual phase of the spiral_manager determines various behaviors, not just values in context dictionaries
- The phase influences which characteristics are added to insights
- Phase transitions can happen based on insight significance

### 2. Required Context Structure

The dream context requires specific keys:

- `seed`: The starting point for the dream
- `theme`: Theme information including `prompt_patterns`
- `cognitive_style`: Style information including `prompt_templates`
- `reflections`: List of reflection prompts
- `questions`: List of questions to consider
- `associations`: Connections between concepts
- `patterns`: Identified patterns in the data

### 3. Integration with Other Components

- The SpiralPhaseManager provides phase-specific parameters via `get_phase_params()`
- The Dream Processor applies phase-specific processing to insights
- Phase transitions can be triggered by the significance of insights

## Testing Best Practices

### 1. State Initialization

- Always explicitly set the phase in tests using `force_phase()` to ensure consistent behavior
- Initialize all required dictionary keys in test fixtures
- Be aware of default values and how they might affect test results

### 2. Understanding Implementation Dependencies

- Review the actual implementation to identify required keys and their expected structure
- Pay attention to how components interact, especially regarding state management
- Understand which parameters influence behavior and ensure tests control them

### 3. Phase-Aware Testing

- Be explicit about which phase your test assumes
- Test behavior across different phases when relevant
- Remember that phase-specific behaviors include:
  - Characteristics added to insights
  - Parameters used for processing
  - Significance thresholds
  - Integration strategies

### 4. Maintaining Test Fixtures

- Update test fixtures when implementation changes
- Use helper methods to create consistent test data
- Document assumptions about test data structure

By following these guidelines, you can create more robust tests and more easily debug issues when they arise.

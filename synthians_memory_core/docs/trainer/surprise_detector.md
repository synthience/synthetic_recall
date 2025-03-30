# Surprise Detection

*This is a placeholder document for detailed documentation on the SurpriseDetector component.*

## Overview

The `SurpriseDetector` is responsible for quantifying the level of surprise or unexpectedness in the Neural Memory's predictions. It implements the principle that **"Surprise signals significance"** by measuring how much a new input deviates from the system's expectations based on prior learning.

## Core Functionality

### Surprise Measurement

The `SurpriseDetector` calculates surprise metrics based on the difference between predicted and actual values:

- **Loss-based Surprise**: Measures the magnitude of prediction error
- **Gradient-based Surprise**: Measures the magnitude of required weight updates
- **Distribution-based Surprise**: Compares current metrics to historical distributions

### Primary Metrics

#### Loss Value

The loss value represents the direct prediction error:

```python
def calculate_loss(predicted_value, actual_value):
    """Calculate L2 loss between prediction and actual value."""
    return 0.5 * np.sum((predicted_value - actual_value) ** 2)
```

Higher loss values indicate greater deviation from expectations, suggesting that the input contains information that the system had not adequately learned to predict.

#### Gradient Norm

The gradient norm measures the magnitude of the update needed to accommodate the new information:

```python
def calculate_gradient_norm(gradient):
    """Calculate the L2 norm of the gradient."""
    return np.linalg.norm(gradient)
```

Larger gradient norms indicate that more significant weight changes are needed to incorporate the new information, suggesting higher surprise or novelty.

### Normalization & Calibration

Raw surprise metrics can vary widely in scale, so the `SurpriseDetector` normalizes and calibrates them:

- **Historical Calibration**: Comparing current metrics to a moving window of recent values
- **Z-score Normalization**: Expressing surprise in terms of standard deviations from the mean
- **Min-Max Scaling**: Mapping surprise values to a fixed range (e.g., 0-1)

## Integration with QuickRecal Boost

The surprise metrics are used by the Context Cascade Engine to calculate QuickRecal boosts:

```python
# Example of how surprise metrics are converted to QuickRecal boosts
def calculate_boost(loss, grad_norm, boost_factor=0.1):
    # Combine loss and gradient norm, with optional weighting
    combined_surprise = loss + 0.5 * grad_norm
    
    # Scale to appropriate boost range
    boost = boost_factor * combined_surprise
    
    # Optional: Apply non-linear transformation (e.g., sigmoid)
    # boost = sigmoid(boost) * max_boost
    
    return boost
```

These boosts are applied to the original memory's QuickRecal score in the Memory Core, reinforcing memories that contained surprising or novel information.

## Technical Implementation

The `SurpriseDetector` functionality is primarily implemented within the Neural Memory Server's `/update_memory` endpoint, which:

1. Calculates the predicted value based on the current memory state
2. Computes the loss between the prediction and the actual value
3. Calculates the gradient of the loss with respect to the memory weights
4. Measures the gradient norm
5. Returns both the loss and gradient norm as surprise metrics

## Configuration Options

- `surprise_normalization`: Method for normalizing surprise metrics ("raw", "z-score", "min-max")
- `history_window_size`: Number of recent updates to consider for historical calibration
- `outlier_threshold`: Z-score threshold above which metrics are considered outliers
- `surprise_minimum_threshold`: Minimum value for surprise to be considered significant

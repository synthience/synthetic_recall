# Neural Memory Module (`NeuralMemoryModule`)

The core of the `synthians_trainer_server` is the `NeuralMemoryModule`, a TensorFlow/Keras model that implements an adaptive associative memory.

## Concept: Test-Time Learning

Unlike traditional models trained offline, this module adapts its internal weights (`M`) *during operation* based on the stream of incoming data. This allows it to continuously learn associations and adapt to changing patterns without requiring explicit retraining phases.

The implementation is heavily inspired by the concepts presented in the "Transformers are Meta-Learners" (Titans) paper, particularly focusing on the associative memory aspect.

## Architecture

```mermaid
graph TD
    Input(Input Embedding x_t) --> ProjK(Proj K - WK)
    Input --> ProjV(Proj V - WV)
    Input --> ProjQ(Proj Q - WQ)

    ProjK --> Key(Key k_t)
    ProjV --> Value(Value v_t)
    ProjQ --> Query(Query q_t)

    Key --> Memory(Memory M)
    Query --> Memory

    subgraph "Update (/update_memory)"
        Memory -- Recall --> PredictedValue(Predicted v_hat)
        Value --> LossFn(Loss ||v_t - v_hat||²)
        LossFn --> Gradient(∇ℓ w.r.t. M)
        Gradient --> Momentum(Momentum S_t)
        Momentum --> UpdateM(Update M_t)
        Gates(Gates α, θ, η) --> Momentum
        Gates --> UpdateM
    end

    subgraph "Retrieve (/retrieve)"
        Memory -- Recall --> RetrievedValue(Retrieved v_ret)
    end
```

**Key Components:**

1.  **Projection Layers (WK, WV, WQ):** Linear layers that project the input embedding `x_t` into different spaces to create the Key (`k_t`), Value (`v_t`), and Query (`q_t`) vectors.
2.  **Memory Network (M):** The core associative memory, typically implemented as a Multi-Layer Perceptron (MLP). Its weights are the parameters that are continuously updated.
3.  **Gates (α, θ, η):** Learnable or fixed parameters controlling the learning dynamics:
    *   `α`: Forget Rate Gate (how much of the old memory `M_{t-1}` to keep).
    *   `θ`: Inner Learning Rate Gate (how much influence the current gradient `∇ℓ` has).
    *   `η`: Momentum Decay Gate (how much momentum `S_{t-1}` persists).
4.  **Momentum State (S):** Tracks the recent history of gradient updates, helping to stabilize learning.

## Operations

### 1. Update (`/update_memory` endpoint)

*   **Input:** `embedding` (representing `x_t`).
*   **Process:**
    1.  Calculate `k_t` and `v_t` using `WK` and `WV`.
    2.  Recall the predicted value `pred_v = M_{t-1}(k_t)`. Pass the *current* key through the *current* memory `M`.
    3.  Calculate the loss `ℓ = ||pred_v - v_t||² / 2` (associative error).
    4.  Compute the gradient `∇ℓ` of the loss with respect to the weights of `M`.
    5.  Update the momentum: `S_t = η_t * S_{t-1} - θ_t * ∇ℓ`.
    6.  Update the memory weights: `M_t = (1 - α_t) * M_{t-1} + S_t`.
*   **Output:** `loss` and `grad_norm` (surprise metrics).

### 2. Retrieve (`/retrieve` endpoint)

*   **Input:** `query_embedding` (representing `x_t`).
*   **Process:**
    1.  Calculate `q_t` using `WQ`.
    2.  Pass the query `q_t` through the *current* memory `M_t`: `retrieved_embedding = M_t(q_t)`.
    3.  This uses the memory in a feed-forward manner **without** updating its weights.
*   **Output:** `retrieved_embedding`.

## Importance

This module allows the system to:

*   Form associations between concepts (embeddings) over time.
*   Adapt its internal representations based on ongoing experience.
*   Provide a mechanism for generating surprise signals (`loss`, `grad_norm`) that indicate novel or unexpected information, which can be used to influence other parts of the system (like QuickRecall scoring).

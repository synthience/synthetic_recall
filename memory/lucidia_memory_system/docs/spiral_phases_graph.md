Below are several detailed Mermaid diagrams that together illustrate the key structures, processes, and data flows within the **SpiralPhaseManager** component.

---

### 1. Phase Transition Flow (transition_phase)

This diagram maps the logic of the `transition_phase` method—from receiving a significance value to updating the phase, recording statistics, and optionally informing the self‐model.

```mermaid
flowchart TD
    %% Phase Transition Flow for transition_phase()
    A[Start: Call transition_phase(significance)]
    B[Retrieve current phase parameters<br/>from phase_config]
    C{Is significance<br/>>= transition_threshold?}
    D[Call _determine_next_phase()<br/>(Determine next_phase)]
    E[Call _update_phase_stats()<br/>(Update time in current phase)]
    F[Update current_phase to next_phase]
    G[Increment transitions<br/>and update last_entered<br/>in phase_stats for new phase]
    H[Append transition event<br/>to phase_history<br/>(timestamp, reason, previous_phase)]
    I[Log transition info]
    J{self_model exists?<br/>& has update_spiral_phase?}
    K[Call self_model.update_spiral_phase(new phase, significance)]
    L[Return True]
    M[Return False]

    A --> B
    B --> C
    C -- Yes --> D
    C -- No --> M
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J -- Yes --> K
    J -- No --> L
    K --> L
```

---

### 2. Forced Phase Transition Flow (force_phase)

This diagram outlines the `force_phase` method’s process—from checking if a forced phase change is needed to updating the state and logging the forced transition.

```mermaid
flowchart TD
    %% Forced Phase Transition Flow for force_phase()
    A[Start: Call force_phase(target_phase, reason)]
    B{Is target_phase<br/>== current_phase?}
    C[Log "Already in target phase" and exit]
    D[Call _update_phase_stats()<br/>for current phase]
    E[Update current_phase to target_phase]
    F[Increment transitions<br/>and update last_entered<br/>in phase_stats for target_phase]
    G[Append forced transition event<br/>to phase_history<br/>(timestamp, reason, previous_phase)]
    H[Log forced transition info]
    I{self_model exists?<br/>& has update_spiral_phase?}
    J[Call self_model.update_spiral_phase(target_phase, 1.0)]
    K[End]

    A --> B
    B -- Yes --> C
    B -- No --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I -- Yes --> J
    I -- No --> K
    J --> K
```

---

### 3. Status Reporting Flow (get_status)

This diagram shows the steps in the `get_status` method, which compiles current phase details, aggregated statistics, and recent history into a status report.

```mermaid
flowchart TD
    %% Status Reporting Flow for get_status()
    A[Start: Call get_status()]
    B[Get current time]
    C[Calculate time_in_current_phase =<br/>current_time - phase_stats[current_phase].last_entered]
    D[Retrieve current phase details<br/>from phase_config (name, description, focus_areas)]
    E[Aggregate spiral_stats:<br/>• cycle_count (from ADAPTATION transitions)<br/>• total transitions<br/>• total insights<br/>• phase_distribution]
    F[Retrieve recent_history (last 5 entries)<br/>from phase_history]
    G[Return status dictionary<br/>(current_phase, spiral_stats, recent_history)]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

---

### 4. Data Structure Overview

This diagram depicts the key data structures within **SpiralPhaseManager** and their interrelationships, including how they connect to the `SpiralPhase` enum and are used across the methods.

```mermaid
flowchart TD
    %% Data Structure Overview for SpiralPhaseManager
    A[SpiralPhaseManager]
    B[phase_config<br/>(Mapping of SpiralPhase → parameters)]
    C[phase_stats<br/>(Mapping of SpiralPhase → {total_time, transitions, insights, last_entered})]
    D[phase_history<br/>(List of transition events)]
    E[current_phase<br/>(Current SpiralPhase)]
    F[self_model<br/>(Optional external component)]
    G[SpiralPhase Enum<br/>(OBSERVATION, REFLECTION, ADAPTATION)]

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    E --- G

    %% Usage relationships:
    B -- Provides parameters for --> transition_phase(), get_phase_params(), get_status()
    C -- Updated by --> _update_phase_stats(), transition_phase(), force_phase()
    D -- Appended by --> transition_phase(), force_phase()
```

---

These diagrams together provide a comprehensive visual representation of the component’s design and operational logic. They highlight key decision points (e.g., checking significance thresholds), state updates (e.g., updating `phase_stats` and `phase_history`), and interactions (e.g., integration with `self_model`). If you require additional details—such as more information about the expected self_model interface—please let me know.
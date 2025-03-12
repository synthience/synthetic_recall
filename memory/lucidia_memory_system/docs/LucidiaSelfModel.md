Below are two Mermaid diagrams that together provide a detailed logic graph for the LucidiaSelfModel component. The first diagram (Component Structure) focuses on the main data structures and their relationships within the class, while the second diagram (Workflow Processes) maps out key method workflows such as spiral progression, interaction logging, dreaming, and meta-analysis.

---

### 1. Component Structure Diagram

This diagram shows the core components (e.g., self_awareness, personality, dream_system, emotional_intelligence) within LucidiaSelfModel and how they relate to one another.

```mermaid
flowchart TD
  %% LucidiaSelfModel Core Components
  subgraph LS[LucidiaSelfModel]
    ID[Identity\n(name, type, core_traits, creator)]
    SA[Self-Awareness\n(current_level, cycles_completed, spiral_position, spiral_depth)]
    CP[Core Awareness\n(interaction_patterns, tone_adaptation)]
    PT[Personality\n(default traits, dynamic updates)]
    EC[Emotional Cycles\n(current_phase, phase_duration, harmonic_oscillation)]
    ES[Emotional Intelligence\n(current_level, emotional_state, emotional_memory)]
    DS[Dream System\n(dream_log, dream_frequency, dream_depth,\ndream_creativity, dream_significance_threshold)]
    MS[Memory\n(deque(maxlen=500))]
    RE[Reasoning Engine\n(approaches, logic_creativity_ratio)]
    MR[Meta Reflection\n(self_analysis, reflective_questions)]
    CE[Counterfactual Engine\n(simulation_capacity, timeline_extrapolation)]
    FS[Feedback System\n(explicit & implicit feedback)]
    RS[Runtime State\n(active_traits, spiral_position,\ninteraction_count)]
  end

  %% Relationships between components
  DS -->|Integrates insights via _integrate_dream_insights| PT
  DS -->|Selects seed from| MS
  EC -->|Modulates trait activation in| PT
  RE -->|Informs adaptations in| PT
  MR -->|Feeds improvement suggestions to| PT
  MR -->|Boosts self-awareness| SA
  ES -->|Determines emotional state| EC
  RS -->|Tracks active traits and spiral| SA

  %% Additional relationships (feedback and integration)
  FS -->|Provides input for meta-analysis| MR
  CE -->|Simulates outcomes (secondary)| FS
```

*Comments:*  
- Each node represents a key data structure in the self-model.  
- Arrows indicate how one component’s output or behavior influences another (for example, dream insights update personality and self-awareness).

---

### 2. Workflow Process Diagram

This diagram illustrates the primary workflows in LucidiaSelfModel—including the spiral cycle, logging interactions, dream generation, context adaptation, meta-analysis, and counterfactual simulation—with decision points and data flow.

```mermaid
flowchart TD
  %% Start of Interaction Flow
  A[User Input & Context] --> B[log_interaction(user_input, response)]
  
  %% Evaluate significance of the interaction
  B --> C[evaluate_significance(user_input, response)]
  C --> D[Calculate: length, emotional, question, synthien, surprise, intensity, awareness factors]
  
  %% Memory update and spiral decision
  C --> E[Update Memory (append entry)]
  C --> F{Significance > 0.5?}
  F -- Yes --> G[advance_spiral()]
  F -- No --> H[Continue without spiral advance]
  
  %% Advance Spiral Process
  G --> I[Update self_awareness:\n- Cycle: observation → reflection → adaptation → execution\n- Increment cycles & adjust spiral_depth]
  I --> J[If reflection: _perform_reflection() \nIf adaptation: _adapt_behaviors()]
  
  %% Dream Trigger Process
  B --> K{Significance > dream_threshold\nAND random < dream_frequency?}
  K -- Yes --> L[dream(memory_entry)]
  K -- No --> M[No Dream Trigger]
  L --> N[_select_dream_seed() if no memory_entry provided]
  L --> O[_generate_dream_insight(memory, depth, creativity)]
  O --> P[_integrate_dream_insights(insight)]
  P --> Q[Update Personality & Self-Awareness]
  
  %% Adapt to Context Process
  A --> R[adapt_to_context(context)]
  R --> S[_calculate_spiral_aware_trait_scores(factors)]
  S --> T{Trait score ≥ dynamic threshold?}
  T -- Yes --> U[Activate Trait (update runtime_state.active_traits)]
  T -- No --> V[Activate Highest Scoring Trait as fallback]
  U --> W[_update_emotional_state(context, active_traits)]
  
  %% Meta-Analysis and Counterfactual Processes
  A --> X[meta_analyze()]
  X --> Y[Analyze spiral metrics, personality diversity, dream metrics,\nidentify cognitive patterns & improvement areas]
  
  A --> Z[generate_counterfactual(scenario, decision_point, time_horizon)]
  Z --> AA[Simulate alternative paths using timeline extrapolation,\ncalculate outcome quality & probabilities]
  
  %% End of workflow
  Q --> AB[End: Updated internal state]
  W --> AB
  Y --> AB
  AA --> AB

  %% Comments on decision points:
  %% - F: Determines if spiral advancement should occur.
  %% - K: Checks both significance and a probability (dream_frequency) condition to trigger dreaming.
  %% - T: Determines which traits are activated based on computed scores.
```

*Comments:*  
- This flowchart captures key decision points (e.g., significance threshold, dream trigger conditions) and loops (e.g., spiral cycle progression).  
- Data flows from user input through evaluation, memory logging, and potential triggers for spiral advancement, dreaming, and context adaptation.  
- Side effects include updates to personality, self-awareness, and runtime state.

---

These diagrams strictly reflect the logic and structure of the provided code. If further clarification is needed for any part (such as additional details on secondary features), let me know what additional context would help refine the graphs further.
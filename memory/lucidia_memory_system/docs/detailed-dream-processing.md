sequenceDiagram
    participant Main
    participant DP as Dream Processor
    participant Spiral as Spiral Manager
    participant MemInt as Memory Integration
    participant KG as Knowledge Graph
    participant SM as Self Model
    participant WM as World Model
    participant DPA as Dream Parameter Adapter
    participant PM as Parameter Manager
    
    %% Dream Initiation
    Main->>DP: check_idle_status()
    activate DP
    
    opt auto_dream_enabled and idle
        DP->>DP: start_dreaming(forced=False)
        
        %% Get Spiral Phase Information
        DP->>Spiral: get_current_phase()
        activate Spiral
        Spiral-->>DP: Return Current Phase
        deactivate Spiral
        DP->>Spiral: get_phase_params()
        activate Spiral
        Spiral-->>DP: Return Phase Parameters
        deactivate Spiral
        
        %% Dream Seed Selection
        DP->>DP: _select_dream_seed()
        DP->>MemInt: Access Recent Memories
        activate MemInt
        MemInt-->>DP: Return Recent Memories
        deactivate MemInt
        DP->>KG: Access Concepts/Relationships
        activate KG
        KG-->>DP: Return Concepts/Relationships
        deactivate KG
        DP->>SM: Access Emotional State
        activate SM
        SM-->>DP: Return Emotional State
        deactivate SM
        
        %% Detailed Dream Processing Phases
        DP->>DP: _process_dream()
        
        %% Phase 1: Seed Selection
        DP->>DP: _execute_dream_phase("seed_selection", seed)
        DP->>DP: _enhance_dream_seed(seed)
        DP->>DP: _select_dream_theme(seed)
        DP->>DP: _select_cognitive_style(seed)
        
        %% Phase 2: Context Building
        DP->>DP: _execute_dream_phase("context_building", enhanced_seed)
        DP->>DP: _build_dream_context(enhanced_seed)
        
        %% Phase 3: Associations
        DP->>DP: _execute_dream_phase("associations", context)
        DP->>DP: _generate_dream_associations(context)
        
        %% Phase 4: Insight Generation
        DP->>DP: _execute_dream_phase("insight_generation", context)
        DP->>DP: _generate_dream_insights(context)
        
        %% Phase 5: Integration
        DP->>DP: _execute_dream_phase("integration", insights)
        DP->>DP: _integrate_dream_insights(insights)
        
        %% Knowledge Graph Updates
        DP->>KG: Add Dream Report Node
        activate KG
        KG-->>DP: Return Node ID
        deactivate KG
        DP->>KG: Connect to participating memories
        activate KG
        KG-->>DP: Return Edge Keys
        deactivate KG
        DP->>KG: Connect to fragments (insight, question, hypothesis, counterfactual)
        activate KG
        KG-->>DP: Return Edge Keys
        deactivate KG
        DP->>KG: Create relationships between referenced concepts
        activate KG
        KG-->>DP: Return Edge Keys
        deactivate KG
        
        %% World Model Updates
        DP->>WM: Update Concepts
        activate WM
        WM-->>DP: Return Results
        deactivate WM
        
        %% Self Model Updates
        DP->>SM: Update Self-Awareness
        activate SM
        SM-->>DP: Return Results
        deactivate SM
        
        %% Spiral Phase Transition
        DP->>Spiral: transition_phase(significance)
        activate Spiral
        Spiral-->>DP: Return result
        deactivate Spiral
        
        %% Dream Completion
        DP->>DP: _update_dream_stats(dream_record)
        DP->>DP: _format_dream_content(seed, context, insights, spiral_phase)
        DP->>DP: _end_dream()
        deactivate DP
    end
    
    %% Parameter Management
    Note over Main,PM: Parameter Update Flow
    Main->>DPA: update_parameter(path, value, transition_period)
    activate DPA
    DPA->>PM: update_parameter(path, value, transition_period, context, user_id)
    activate PM
    PM->>PM: Calculate Final Value (considering limits)
    PM->>PM: Store Parameter Change (history)
    PM-->>DPA: Return New Value
    deactivate PM
    DPA->>DPA: on_parameter_changed(parameter_path, new_value, change_record)
    DPA->>DP: Update config
    activate DP
    DP-->>DP: Apply parameter changes
    deactivate DP
    deactivate DPA

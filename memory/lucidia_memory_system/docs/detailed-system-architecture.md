graph TD
    %% Main components
    User([User]) <-->|Input/Output| Main[Main System]
    Config[Configuration File] -->|Loaded by| Main
    
    %% Major subsystems
    Main -->|Initializes| Memory[Memory System]
    Main -->|Initializes| Knowledge[Knowledge System]
    Main -->|Initializes| Dream[Dream System]
    Main -->|Initializes| Hyper[Hypersphere System]
    
    %% Memory Components
    Memory --> MemInt[Memory Integration]
    MemInt --> MemCore[Memory Core]
    MemCore --> STM[Short-Term Memory]
    MemCore --> LTM[Long-Term Memory]
    MemCore --> MPL[Memory Prioritization Layer]
    
    %% Knowledge Components
    Knowledge --> KG[Knowledge Graph]
    KG -->|Import Initial Concepts & Beliefs| SM[Self Model]
    KG -->|Import Initial Concepts & Relationships| WM[World Model]
    
    %% Dream Components
    Dream --> DP[Dream Processor]
    DP -->|get_current_phase, transition_phase| Spiral[Spiral Manager]
    Dream --> DPA[Dream Parameter Adapter]
    DPA -->|update_parameter, get_parameter| PM[Parameter Manager]
    
    %% Hypersphere Components
    Hyper --> HD[Hypersphere Dispatcher]
    HD -->|Get Connection| TP[Tensor Pool]
    HD -->|Get Connection| HP[HPC Pool]
    HD -->|check_embedding_compatibility| GR[Geometry Registry]
    HD --> CM[Confidence Manager]
    HD --> DM[Decay Manager]
    HD -->|schedule_embedding_fetch| BS[Batch Scheduler]
    
    %% Server Components
    TP --> TensorServer[Tensor Server]
    HP --> HPCServer[HPC Server]
    
    %% Key Interactions
    MemCore -->|Request Embedding Generation| HD
    HD -->|Return Processed Embedding, Significance| MemCore
    
    MPL -->|get_recent, similarity search| STM
    MPL -->|search_memory| LTM
    MPL -->|fetch_relevant_embeddings| HD
    
    DP -->|Access Recent Memories| MemInt
    DP -->|add_node, add_edge, integrate_dream_insight| KG
    DP -->|Access Emotional State| SM
    DP -->|Access Concepts| WM
    
    Config -->|Provides Settings for| Memory & Knowledge & Dream & Hyper
    
    %% Parameter Flow
    User -->|Parameter Updates| Main
    Main -->|update_parameter| DPA
    DPA -->|Notifies| DP

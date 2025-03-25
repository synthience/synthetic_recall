sequenceDiagram
    participant MemCore as Memory Core
    participant MPL as Memory Prioritization Layer
    participant DP as Dream Processor
    participant HD as Hypersphere Dispatcher
    participant TP as Tensor Pool
    participant TS as Tensor Server
    participant HPPool as HPC Pool
    participant HPServer as HPC Server
    participant GR as Geometry Registry
    participant CM as Confidence Manager
    participant DM as Decay Manager
    participant BS as Batch Scheduler
    
    %% Similarity Search Operation
    MPL->>HD: batch_similarity_search(query_embedding, memory_embeddings, memory_ids, model_version, top_k)
    activate HD
    
    %% Embedding Compatibility Checks
    HD->>GR: check_embedding_compatibility(model_version, query_embedding)
    activate GR
    GR-->>HD: Return True/False
    deactivate GR
    
    alt Incompatible Query Embedding
        HD->>HD: _normalize_embedding(query_embedding)
    end
    
    HD->>GR: check_embedding_compatibility(model_version, memory_embeddings[0])
    activate GR
    GR-->>HD: Return True/False
    deactivate GR
    
    alt Incompatible Memory Embeddings
        HD->>HD: _normalize_embedding(memory_embeddings)
    end
    
    %% Get Server Connection
    HD->>HPPool: Get Connection
    activate HPPool
    HPPool-->>HD: Return Connection
    deactivate HPPool
    
    %% Execute Search
    HD->>HPServer: Send Similarity Search Request
    activate HPServer
    HPServer-->>HD: Return Similarities and Indices
    deactivate HPServer
    
    %% Process Results
    HD->>HD: _process_similarity_results(similarities, indices, memory_ids, top_k)
    HD-->>MPL: Return Results (memory_ids, similarities)
    deactivate HD
    
    %% Embedding Generation
    MemCore->>HD: process_embedding(content)
    activate HD
    HD->>TP: Get Connection
    activate TP
    TP-->>HD: Return Connection
    deactivate TP
    HD->>TS: Generate Embedding
    activate TS
    TS-->>HD: Return Raw Embedding
    deactivate TS
    HD->>HD: Calculate Quick recall score
    HD-->>MemCore: Return Embedding and Quick recall score
    deactivate HD
    
    %% Embedding Decay Operation
    DP->>HD: decay_embedding(embedding, decay_rate, decay_method)
    activate HD
    HD->>DM: decay(embedding, decay_rate, decay_method)
    activate DM
    DM->>DM: Apply Decay Function
    DM-->>HD: Return Decayed Embedding
    deactivate DM
    HD-->>DP: Return Decayed Embedding
    deactivate HD
    
    %% Fetch Relevant Embeddings
    MPL->>HD: fetch_relevant_embeddings(query, criteria, limit)
    activate HD
    HD->>BS: schedule_embedding_fetch(query, criteria, limit)
    activate BS
    
    BS->>HPPool: Get Connection
    activate HPPool
    HPPool-->>BS: Return Connection
    deactivate HPPool
    
    BS->>HPServer: Batch Embedding Requests
    activate HPServer
    HPServer-->>BS: Return Embeddings
    deactivate HPServer
    
    BS-->>HD: Return Embeddings
    deactivate BS
    HD-->>MPL: Return Embeddings
    deactivate HD
    
    %% Embedding History
    MemCore->>HD: get_embedding_history(memory_id)
    activate HD
    HD->>CM: get_confidence_scores(memory_id)
    activate CM
    CM-->>HD: Return Confidence History
    deactivate CM
    HD-->>MemCore: Return Confidence History
    deactivate HD
    
    %% Model Registration
    MemCore->>HD: register_model(model_name, geometry_config)
    activate HD
    HD->>GR: register_geometry(model_name, geometry_config)
    activate GR
    GR->>GR: Store geometry configuration
    GR-->>HD: Return success
    deactivate GR
    HD-->>MemCore: Return boolean
    deactivate HD

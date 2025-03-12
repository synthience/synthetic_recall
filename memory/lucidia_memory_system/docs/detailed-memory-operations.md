sequenceDiagram
    %% Participants
    participant User
    participant Main
    participant MemInt as Memory Integration
    participant MemCore as Memory Core
    participant STM
    participant LTM
    participant MPL
    participant HD as Hypersphere Dispatcher
    participant GR as Geometry Registry
    participant HPPool
    participant HPServer as HPC Server
    participant EC as Embedding Comparator
    
    %% Memory Storage Process
    User->>Main: Input (Content to Store)
    Main->>MemInt: store(content, metadata, importance)
    MemInt->>MemCore: process_and_store(content, memory_type, metadata)
    
    %% Embedding Process
    MemCore->>HD: process_embedding(content)
    activate HD
    HD->>GR: check_embedding_compatibility(model_version, embedding)
    GR-->>HD: Return True/False
    
    alt Incompatible Embedding
        HD->>HD: _normalize_embedding(embedding)
    end
    
    HD->>HPPool: Get Connection
    HPPool-->>HD: Return Connection
    HD->>HPServer: Send Embedding Request
    HPServer-->>HD: Return Embedding
    HD->>HD: Calculate Significance
    HD-->>MemCore: Return Embedding, Significance
    deactivate HD
    
    %% STM Storage
    MemCore->>STM: add_memory(content, embedding, metadata)
    STM-->>MemCore: Return STM ID
    
    %% LTM Storage (if significant)
    MemCore->>MemCore: Evaluate Significance
    
    alt Significance >= threshold
        MemCore->>LTM: store_memory(content, embedding, significance, metadata)
        activate LTM
        LTM-->>MemCore: Return LTM ID
        deactivate LTM
    end
    
    MemCore-->>MemInt: Return Result (success, stm_id, ltm_id)
    deactivate MemCore
    MemInt-->>Main: Return Result
    deactivate MemInt
    Main-->>User: Confirmation
    deactivate Main
    
    %% Memory Recall Process
    User->>Main: Query (Content to Retrieve)
    activate Main
    Main->>MemInt: recall(query, limit, min_importance)
    activate MemInt
    MemInt->>MPL: route_query(query, context)
    activate MPL
    
    %% Process Query Embedding
    MPL->>HD: process_embedding(query)
    activate HD
    HD->>GR: check_embedding_compatibility(model_version, query)
    GR-->>HD: Return True/False
    
    alt Incompatible Embedding
        HD->>HD: _normalize_embedding(query)
    end
    
    HD->>EC: get_embedding(query)
    EC-->>HD: Return Embedding
    HD-->>MPL: Return Query Embedding
    deactivate HD
    
    %% Check STM
    MPL->>STM: get_recent(limit)
    activate STM
    STM-->>MPL: Return Recent Memories
    deactivate STM
    MPL->>HD: batch_similarity_search(query_embedding, stm_embeddings)
    activate HD
    HD-->>MPL: Return STM Similarities
    deactivate HD
    
    %% Results Merging Logic
    MPL->>MPL: Initialize results collection
    
    alt Strong match in STM
        MPL->>MPL: Add STM results to collection
    end
    
    %% Check LTM
    MPL->>LTM: search_memory(query_embedding, limit)
    LTM-->>MPL: Return LTM Results
    
    alt Strong match in LTM
        MPL->>MPL: Add LTM results to collection
    end
    
    %% If needed, check HPC
    alt Insufficient results
        MPL->>HD: fetch_relevant_embeddings(query, criteria, limit)
        activate HD
        HD->>HPPool: Get Connection
        HPPool-->>HD: Return Connection
        HD->>HPServer: Similarity Search Request
        HPServer-->>HD: Return HPC Results
        HD-->>MPL: Return HPC Results
        deactivate HD
        MPL->>MPL: Add HPC results to collection
    end
    
    %% Merge and Return Results
    MPL->>MPL: Merge and rank all results
    MPL-->>MemInt: Return Combined Results
    MemInt-->>Main: Return Results
    Main-->>User: Response

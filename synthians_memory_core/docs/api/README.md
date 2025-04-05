# Synthians API Documentation

This directory contains reference documentation for the HTTP APIs exposed by the Synthians Cognitive Architecture services (Memory Core, Neural Memory, CCE) and usage guides for the Python client libraries.

## Contents

*   [**API Reference**](./API_REFERENCE.md): Comprehensive reference for all HTTP API endpoints, including currently implemented endpoints and those planned for Phase 5.9 such as Memory Core (`/explain_*`, `/diagnostics/*`, `/config/*`), Neural Memory, and CCE (`/metrics/recent_cce_responses`). Details request/response models, parameters, and status codes.
*   [**Client Usage Guide**](./client_usage.md): Guidelines and code examples for using the asynchronous Python clients (`SynthiansClient` for MC) to interact with the APIs. Includes examples for both current and planned endpoints.

## Existing API Endpoints (Currently Implemented)

*   **Memory Core**:
    *   `GET /`: Root endpoint
    *   `GET /health`: Check service health
    *   `GET /stats`: Retrieve system statistics
    *   `POST /process_memory`: Process and store a memory
    *   `POST /retrieve_memories`: Retrieve memories by similarity
    *   `POST /generate_embedding`: Generate embedding from text
    *   `POST /calculate_quickrecal`: Calculate QuickRecal score
    *   `POST /analyze_emotion`: Analyze emotions in text
    *   `POST /provide_feedback`: Provide relevance feedback
    *   `POST /detect_contradictions`: Detect contradictions
    *   `POST /process_transcription`: Process transcription data
    *   `GET /api/memories/{memory_id}`: Get memory by ID
    *   `GET /assemblies`: List all assemblies
    *   `GET /assemblies/{assembly_id}`: Get assembly details
    *   `GET /check_index_integrity`: Check FAISS index integrity
    *   `POST /repair_index`: Repair index issues
    *   `GET/POST /repair_vector_index_drift`: Repair vector drift

*   **Neural Memory**:
    *   `POST /update_memory`: Update memory in Neural Memory
    *   `POST /retrieve`: Retrieve similar embeddings
    *   `GET /diagnose_emoloop`: Get diagnostic metrics

*   **CCE**:
    *   `POST /process_memory`: Process memory through cognitive cycle
    *   `GET /metrics/recent_cce_responses`: Get recent CCE metrics

## Planned API Additions (Phase 5.9)

*   **Explainability** (Memory Core):
    *   `GET /assemblies/{id}/explain_activation`
    *   `GET /assemblies/{id}/explain_merge`
    *   `GET /assemblies/{id}/lineage`
    *   `GET /memories/{id}/explain_selection` (Optional/Basic)
*   **Diagnostics** (Memory Core):
    *   `GET /diagnostics/merge_log`
    *   `GET /config/runtime/{service_name}`
*   **Statistics**:
    *   Enhancement to `/stats` endpoint with assembly activation and pending update counts.
*   **CCE**:
    *   Enhancement to `/metrics/recent_cce_responses` with additional details on variant selection and LLM usage.

Refer to the detailed `API_REFERENCE.md` for full specifications of both existing and planned endpoints.

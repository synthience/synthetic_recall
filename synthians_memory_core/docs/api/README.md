# API Reference & Client Documentation

This directory contains documentation for the HTTP API exposed by the Synthians Memory Core service and guidelines for using the Python client.

## Contents

*   [API Reference](./API_REFERENCE.md): Comprehensive reference for all HTTP API endpoints exposed by the Synthians Memory Core (`http://localhost:5010`), including request/response models and parameters. Details cover memory processing, retrieval, embedding generation, QuickRecal scoring, emotion analysis, feedback mechanisms, and integration points for the Neural Memory / Orchestrator.
*   [Client Usage](./client_usage.md): Guidelines and code examples for using the asynchronous Python client (`SynthiansClient`) to interact with the Memory Core API. Demonstrates basic operations, utility endpoints, advanced features like feedback and contradiction detection, and error handling best practices.

## Technical Details

*   **Framework:** The API is built using FastAPI.
*   **Data Format:** Uses JSON for all request and response bodies. Pydantic models define the structure (see `synthians_memory_core/api/server.py`).
*   **Error Handling:** Follows standard HTTP status codes. Errors often include a `"detail"` field (FastAPI default) or a structured response with `"success": false` and `"error": "message"`.
*   **Asynchronous:** The server and client are designed for asynchronous operations using `asyncio`.
*   **Authentication:** Currently, no specific authentication is implemented in the provided code. Access control would need to be added (e.g., API keys, JWT) for production environments.
*   **Client:** The `SynthiansClient` library simplifies interaction by handling `aiohttp` requests, session management, and basic response parsing within an async context manager.

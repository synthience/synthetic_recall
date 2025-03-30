# Metrics and Diagnostics

The `synthians_trainer_server.metrics_store.MetricsStore` class is responsible for collecting and storing operational statistics from the Neural Memory server.

## Purpose

Tracking metrics allows for:

*   **Monitoring:** Observing the server's performance and health (e.g., request counts, processing times).
*   **Debugging:** Identifying bottlenecks or issues.
*   **Analysis:** Understanding the behavior of the neural memory model (e.g., average loss, gradient norms).

## Key Component: `MetricsStore`

*   **Functionality:**
    *   Provides methods to increment counters (`increment_request_count`), record timings (`record_processing_time`), and store specific values (`record_loss`, `record_grad_norm`).
    *   Stores metrics in memory, often using dictionaries or specialized data structures.
    *   Periodically calculates averages or aggregates (e.g., average processing time over the last minute).
*   **Integration:**
    *   Instantiated within the main FastAPI application.
    *   Accessed by endpoint handlers to record metrics after processing requests (e.g., `/update_memory`, `/retrieve`).

## Collected Metrics (Examples)

*   Total requests for each endpoint (`/update_memory`, `/retrieve`).
*   Average processing time for each endpoint.
*   Average loss (`ℓ`) calculated during `/update_memory` calls.
*   Average gradient norm (`∇ℓ`) calculated during `/update_memory` calls.
*   Number of successful updates vs. errors.
*   Current memory usage or other system-level stats (potentially).

## Diagnostic Endpoints

The server typically exposes endpoints to retrieve these collected metrics:

*   `/metrics`: Often returns metrics in a standard format (like Prometheus exposition format) for scraping by monitoring systems.
*   `/status` or `/health`: Provides a basic health check and potentially key operational statistics.
*   `/diagnostics` (Optional): Might return a more detailed, human-readable summary of the metrics.

## Importance

Monitoring and diagnostics are crucial for maintaining a reliable and performant service, especially for a component like the Neural Memory that undergoes continuous adaptation.

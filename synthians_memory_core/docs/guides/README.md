# Guides & Configuration Documentation

This directory contains guides and configuration documentation for the Synthians cognitive system.

## Contents

* [Configuration Guide](./CONFIGURATION_GUIDE.md): Explains key configuration parameters for the Memory Core and Neural Memory servers, often managed via environment variables or configuration files.
* [Implementation Guide](./implementation_guide.md): Provides deeper insights into the system's setup, including dependencies, running the services (e.g., using Docker Compose), and potential extension points.
* [Tooling Guide](./tooling_guide.md): Describes available utilities and scripts for maintenance, diagnostics, and repair tasks, such as index verification or data migration.

## User Guides

This directory contains practical guides for interacting with and utilizing the Synthians Memory Core API.

## Available Guides

*   [Client Usage Guide](./client_usage.md): Detailed instructions on how to use the `SynthiansClient` library to interact with the Memory Core API, including initialization, core operations (processing memories, retrieval), asynchronous context management, and basic examples.
*   [Error Handling Guide](./error_handling.md): Best practices for handling potential errors when interacting with the API, covering common HTTP status codes, API error responses, and client-side exception handling.
*   [Adaptive Threshold Feedback Loop Guide](./feedback_loop.md): Explanation of how to provide feedback on the relevance of retrieved memories using the `provide_feedback` method to help the system adapt its retrieval threshold.

Refer to the main [API Reference](../API_REFERENCE.md) for detailed endpoint specifications.

## Technical Details

* **Environment Variables**: How environment variables control service behavior, including model selection, embedding dimensions, logging levels, and variant selection.
* **Configuration Dictionaries**: How the services can be configured programmatically via configuration dictionaries passed to their constructors.
* **Service Integration**: How to set up and integrate the three core services (Memory Core, Neural Memory Server, Context Cascade Engine).
* **Deployment Options**: Different deployment configurations (local development, production, GPU vs CPU).
* **Maintenance Procedures**: Guidelines for backing up memory data, monitoring system health, and troubleshooting common issues.

# Changelog

All notable changes to the Synthians Cognitive Architecture will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Performance-Aware Variant Selection (Phase 5.5) enabling dynamic adaptation based on Neural Memory metrics
- Trend analysis for Neural Memory performance metrics to proactively select optimal variants
- Integration tests for Performance-Aware selection to verify functionality
- Comprehensive documentation for the Performance-Aware selection system
- Comprehensive documentation structure in the `docs/` directory
- Placeholders for component deep dives to be filled in future updates

### Changed
- Enhanced `VariantSelector` to consider performance metrics in addition to content and metadata
- Reorganized documentation into logical sections (core, api, orchestrator, trainer, guides, testing)
- Updated API_REFERENCE.md to include metadata_filter parameter for memory retrieval

### Fixed
- Documentation now accurately reflects the latest codebase state
- Links and references updated to match the new structure

## [1.0.0] - 2025-03-30

### Added
- Functional surprise feedback loop from Neural Memory to Memory Core's QuickRecal score
- Comprehensive configuration via environment variables and config dictionaries
- Robust handling of embedding dimension mismatches (384D vs 768D)
- Enhanced emotional gating for memory retrieval

### Changed
- Refactored Vector Index to use FAISS IndexIDMap for more robust ID handling
- Improved retrieval pipeline with lower pre-filter threshold (0.3) for better recall sensitivity
- Centralized embedding geometry operations in GeometryManager

### Fixed
- Embedding validation to check for NaN/Inf values
- Metadata enrichment in process_new_memory workflow
- Redundant emotion analysis by respecting API-passed emotion data

## [0.9.0] - 2025-03-15

### Added
- Initial implementation of the Context Cascade Engine for orchestrating the cognitive cycle
- Implementation of the three Titans variants (MAC, MAG, MAL)
- Initial API for Neural Memory Server
- Test-time learning capability for Neural Memory Module

### Changed
- Enhanced FAISS integration with GPU support
- Improved Memory Core persistence mechanism

### Fixed
- TensorFlow and NumPy compatibility issues via lazy loading
- Background task cancellation during application shutdown

## [0.8.0] - 2025-02-28

### Added
- UnifiedQuickRecallCalculator with HPC-QR factors
- Emotional analysis and gating service
- MetadataSynthesizer for enriching memory entries
- Basic API server and client

### Changed
- Improved memory persistence with async operations
- Enhanced embedding generation with model configuration

### Fixed
- Memory retrieval performance issues
- Vector index persistence reliability

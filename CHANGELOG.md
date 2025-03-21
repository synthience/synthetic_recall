# Lucidia Changelog

## v1.01 - 2025-03-21

### Fixed

- Fixed async loop issues in Quick Recall system tests with a more robust `safe_run` function
- Enhanced QR scoring adaptability to ensure proper handling of malformed embeddings
- Improved QR score capping mechanism for various edge cases:
  - NaN and Inf values now properly generate penalty vectors
  - QR scores are properly capped rather than multiplied by cap factor
  - Dimension mismatch handling now more consistent between calculator and flow manager
- Context information now properly passed between components to maintain QR capping integrity

### Technical Improvements

- Added comprehensive test suite for malformed embedding handling
- Implemented thread-based solution to handle async functions in already-running event loops
- Unified error handling approach across both calculator and flow manager components
- Enhanced NaN/Inf value detection and penalty vector generation

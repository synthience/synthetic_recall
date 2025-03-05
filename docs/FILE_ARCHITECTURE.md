# Lucid Recall File Architecture

## Required Directory Structure

```
lucid-recall/
├── docker/
│   ├── Dockerfile.lucid-recall-core
│   └── docker-entrypoint.sh
├── interface/
│   ├── index.html           # Main UI interface
│   ├── js/
│   │   └── client.js        # WebSocket client implementation
│   └── styles/
│       └── main.css         # UI styling
├── server/
│   ├── tensor_server.py     # Embedding and memory operations
│   ├── hpc_server.py        # HPC processing
│   ├── hpc_sig_flow_manager.py  # Significance calculation
│   ├── memory_index.py      # Memory storage and retrieval
│   ├── chat_processor.py    # Chat handling
│   └── test_chat_recall.py  # Integration tests
├── docs/
│   ├── LUCID_RECALL_ARCHITECTURE.md
│   ├── TECHNICAL_SPECIFICATION.md
│   ├── DEPLOYMENT.md
│   ├── DOCKER_ARCHITECTURE.md
│   ├── MEMORY_SYSTEM.md
│   ├── BUILD.md
│   └── FILE_ARCHITECTURE.md
├── docker-compose.yml       # Container orchestration
└── .env.example            # Environment template
```

## Critical Files for Distribution

### Required Files for Team Distribution
Create a zip file containing:

1. Core Implementation:
```
server/
├── tensor_server.py
├── hpc_server.py
├── hpc_sig_flow_manager.py
├── memory_index.py
└── chat_processor.py
```

2. Interface Files:
```
interface/
├── index.html
├── js/
│   └── client.js
└── styles/
    └── main.css
```

3. Docker Configuration:
```
docker/
├── Dockerfile.lucid-recall-core
└── docker-entrypoint.sh
docker-compose.yml
.env.example
```

4. Documentation:
```
docs/
├── LUCID_RECALL_ARCHITECTURE.md
├── TECHNICAL_SPECIFICATION.md
├── DEPLOYMENT.md
├── DOCKER_ARCHITECTURE.md
├── MEMORY_SYSTEM.md
├── BUILD.md
└── FILE_ARCHITECTURE.md
```

### Creating Distribution Package

1. Create distribution package:
```bash
# Create dist directory
mkdir lucid-recall-dist

# Copy required files
cp -r server interface docker docs lucid-recall-dist/
cp docker-compose.yml .env.example lucid-recall-dist/

# Create zip
zip -r lucid-recall-dist.zip lucid-recall-dist/
```

2. Verify package contents:
```bash
unzip -l lucid-recall-dist.zip
```

## File Dependencies

### Server Components
- tensor_server.py depends on:
  - memory_index.py
  - hpc_sig_flow_manager.py

- hpc_server.py depends on:
  - hpc_sig_flow_manager.py

- chat_processor.py depends on:
  - memory_index.py

### Interface Components
- client.js depends on:
  - WebSocket connections to tensor_server.py and hpc_server.py
  - index.html for DOM elements

### Docker Components
- Dockerfile.lucid-recall-core depends on:
  - docker-entrypoint.sh
  - Server component files

## Development Guidelines

### File Placement
1. Server Components:
   - All Python server files go in server/
   - Test files should be prefixed with test_
   - Configuration in root directory

2. Interface Components:
   - HTML files in interface/
   - JavaScript in interface/js/
   - CSS in interface/styles/

3. Docker Components:
   - Dockerfile in docker/
   - docker-compose.yml in root
   - Environment files in root

### File Naming Conventions
1. Server Files:
   - Use snake_case for Python files
   - Suffix test files with _test or test_
   - Use descriptive prefixes (e.g., hpc_, tensor_)

2. Interface Files:
   - Use camelCase for JavaScript files
   - Use kebab-case for CSS files
   - Use lowercase for HTML files

3. Documentation:
   - Use UPPERCASE for documentation files
   - Use .md extension
   - Include category prefix (e.g., TECHNICAL_, DEPLOYMENT_)

## Maintenance Notes

### File Updates
1. Server Files:
   - Maintain version comments
   - Update corresponding tests
   - Keep imports organized

2. Interface Files:
   - Maintain CSS class consistency
   - Keep JavaScript modular
   - Update HTML IDs consistently

3. Documentation:
   - Keep in sync with code changes
   - Update version numbers
   - Maintain consistent formatting

### Version Control
- Include all required files in version control
- Exclude environment-specific files
- Maintain .gitignore for build artifacts
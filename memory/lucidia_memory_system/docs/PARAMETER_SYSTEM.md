# Lucidia Parameter Management System

## Overview

The Parameter Management System is a core component of the Lucidia architecture that enables dynamic configuration and adjustment of system parameters at runtime. This document details the implementation, usage, and recent enhancements related to parameter persistence.

## Architecture

The parameter system consists of several integrated components:

### Components

1. **ParameterManager**: Core class that manages parameter state, validation, and persistence
   - Located in `memory/lucidia_memory_system/core/parameter_manager.py`
   - Provides methods for getting, setting, and updating parameters
   - Handles parameter validation, locks, and transition schedules
   - Manages persistence to disk and notifies observers of changes

2. **Dream API Endpoints**: REST API for parameter management
   - Located in `server/dream_api.py`
   - Provides endpoints for retrieving and updating parameters
   - Handles missing parameters by auto-creating them with sensible defaults
   - Ensures changes are persisted to disk

3. **CLI Interface**: Command-line interface for parameter management
   - Located in `lucidia_cli.py`
   - Provides commands for viewing and updating parameters
   - Ensures local configuration file is updated after successful API calls

### Configuration File

Parameters are stored in a JSON configuration file, typically `lucidia_config.json`, which serves as the single source of truth for parameter values across all system components.

## Parameter Persistence Implementation

### Key Features

1. **Auto-Save to Disk**: The `ParameterManager` now automatically saves changes to disk after successful parameter updates

2. **Unified Configuration Path**: Both the Dream API server and CLI use the same configuration file path

3. **Multiple File Support**: The system can update multiple configuration files to ensure changes are visible across components

4. **Auto-Creation of Parameters**: Missing parameters are created with sensible defaults when accessed or updated

5. **Docker Environment Support**: The persistence mechanism works seamlessly in containerized environments

### Implementation Details

#### ParameterManager Save Method

```python
def save_config_to_disk(self):
    """Save the current parameter configuration to disk."""
    try:
        # Determine the path to save to
        if hasattr(self, 'config_file') and self.config_file:
            config_path = self.config_file
        else:
            config_path = getattr(self, 'initial_config', None)
            
        # Create a clean copy of the config without internal attributes
        config_to_save = {}
        for key, value in self.config.items():
            if not key.startswith('_'):
                config_to_save[key] = value
        
        # Save to the config file
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        # Also try to save to lucidia_config.json in the current directory
        local_config_path = "lucidia_config.json"
        if os.path.exists(local_config_path):
            try:
                with open(local_config_path, 'r') as f:
                    local_config = json.load(f)
                
                # Update with our config values
                local_config.update(config_to_save)
                
                # Save back
                with open(local_config_path, 'w') as f:
                    json.dump(local_config, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Could not update local config file: {e}")
        
        return True
    except Exception as e:
        self.logger.error(f"Error saving configuration: {e}")
        return False
```

#### Auto-Creation of Parameters

The Dream API now includes logic to auto-create missing parameters with sensible defaults:

```python
if current_value is None:
    logger.info(f"Parameter {parameter_path} not found, creating it")
    
    # Initialize the parameter with a sensible default based on parameter name
    if "batch_size" in parameter_path:
        default_value = 50  # Default batch size
    elif "threshold" in parameter_path:
        default_value = 0.7  # Default threshold
    # ... additional defaults ...
    else:
        default_value = 0  # Generic default
        
    # Set the default value
    parameter_manager._set_nested_value(parameter_manager.config, parameter_path, default_value)
    current_value = default_value
    
    # Save the configuration to ensure the structure persists
    parameter_manager.save_config_to_disk()
```

## CLI Usage Examples

### Viewing Parameters

```bash
# View all parameters
python lucidia_cli.py params

# View a specific branch of parameters
python lucidia_cli.py params --branch memory
```

### Updating Parameters

```bash
# Update a parameter
python lucidia_cli.py params --path memory.batch_size --value 300

# Update with gradual transition
python lucidia_cli.py params --path memory.decay_rate --value 0.05 --transition 3600
```

## Docker Environment Considerations

In a Docker environment, parameter persistence requires special handling:

1. **Volume Mounting**: The configuration file should be mounted as a volume to persist across container restarts

```yaml
volumes:
  - ./data:/app/data
  - ./config:/app/config
  - ./lucidia_config.json:/app/lucidia_config.json
```

2. **File Access Synchronization**: Both the CLI (on host) and API (in container) should access the same file

3. **Container Restart**: After significant parameter changes, containers may need to be restarted to apply all changes

## History and Change Tracking

The parameter system maintains a history of all parameter changes, recording:

- The parameter path
- Old and new values
- Timestamp
- User ID (if available)
- Context (if provided)
- Transaction ID

This history can be accessed for auditing and rollback purposes.

## Troubleshooting

### Common Issues

1. **Parameter Changes Not Persisting**:
   - Verify the configuration file path is correctly set in both the API server and CLI
   - Check file permissions on the configuration file
   - Ensure the container has write access to the mounted volume

2. **Missing Parameters**:
   - Parameters will be auto-created when accessed, so this should be rare
   - Check for typos in parameter paths
   - Verify the configuration file is properly formatted JSON

3. **Errors During Update**:
   - Check for parameter locks that might be preventing updates
   - Verify proper JSON formatting in request payloads
   - Check log files for detailed error messages

## Recent Enhancements

1. **Improved Parameter Persistence**: Parameters now persist reliably across API restarts and between CLI and API

2. **Auto-Creation of Parameters**: Missing parameters are created with sensible defaults

3. **Better Error Handling**: More informative error messages and logging

4. **Unified Configuration Path**: Consistent configuration file path across components

5. **Docker-Aware Implementation**: Enhanced to work seamlessly in containerized environments

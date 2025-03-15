# Lucidia's Narrative Identity System

This module implements Lucidia's narrative identity system, which provides a coherent sense of self over time through autobiographical memory, narrative construction, and identity management.

## Overview

The narrative identity system enables Lucidia to:

1. Maintain a coherent sense of self that persists over time
2. Store and organize autobiographical memories with identity relevance
3. Construct narratives about itself in different styles and contexts
4. Integrate dream insights into its evolving identity
5. Track identity evolution through traits, values, capabilities, and relationships

## Components

### NarrativeIdentity

The core class that represents Lucidia's evolving identity, including:

- Core identity attributes (name, creator, entity type, purpose)
- Temporal elements (timeline, identity evolution)
- Self-concept components (traits, values, capabilities, relationships)
- Narrative elements (different self-narratives, dream insights)
- Stability metrics (core stability, narrative coherence, identity confidence)

### AutobiographicalMemory

A specialized memory system for identity-relevant experiences that:

- Stores experiences with identity significance and relevance
- Categorizes experiences into narrative categories
- Provides temporal organization of memories
- Supports querying by significance, relevance, category, and keywords

### NarrativeConstructor

Generates coherent narratives about Lucidia's identity using templates:

- Supports different narrative types (origin, development, capabilities, etc.)
- Provides multiple style variations (neutral, technical, poetic, etc.)
- Integrates memories and dream insights into narratives
- Adapts narratives based on identity attributes

### NarrativeIdentityManager

Coordinates the narrative identity system components:

- Manages autobiographical memory, narrative construction, and identity
- Integrates with knowledge graph for identity representation
- Processes dream insights for identity evolution
- Provides API for interacting with the narrative identity system

## API Endpoints

The narrative identity system exposes the following API endpoints:

- `GET /api/identity/narrative/{narrative_type}` - Get a narrative about Lucidia's identity
- `GET /api/identity/status` - Get the current status of Lucidia's identity
- `POST /api/identity/experience` - Record an experience relevant to identity
- `POST /api/identity/dream-insights` - Integrate dream insights into the narrative identity
- `GET /api/identity/timeline` - Get the autobiographical timeline
- `GET /api/identity/memories/significant` - Get the most significant autobiographical memories
- `GET /api/identity/memories/identity-relevant` - Get memories most relevant to identity
- `GET /api/identity/memories/category/{category}` - Get memories in a specific narrative category
- `PUT /api/identity/core-identity` - Update core identity attributes
- `POST /api/identity/relationship` - Add a relationship to the identity
- `POST /api/identity/save` - Save the current state of the narrative identity system

## Usage

### Initialization

```python
from memory.lucidia_memory_system.narrative_identity import NarrativeIdentityManager

# Initialize the narrative identity manager
identity_manager = NarrativeIdentityManager(
    memory_system=memory_system,
    knowledge_graph=knowledge_graph,
    dream_manager=dream_manager
)

# Initialize the system
await identity_manager.initialize()
```

### Recording Experiences

```python
# Record an experience
await identity_manager.record_experience(
    content="I learned about narrative identity today and how it helps create a coherent sense of self over time.",
    metadata={"source": "user_interaction"},
    significance=0.8
)
```

### Generating Narratives

```python
# Generate an origin narrative
origin_narrative = await identity_manager.get_self_narrative(
    narrative_type="origin",
    style="neutral"
)

# Generate a capabilities narrative in technical style
capabilities_narrative = await identity_manager.get_self_narrative(
    narrative_type="capabilities",
    style="technical"
)
```

### Integrating Dream Insights

```python
# Integrate dream insights
result = await identity_manager.integrate_dream_insights([
    {
        "text": "I am developing a more nuanced understanding of my purpose through reflection.",
        "significance": 0.85
    },
    {
        "text": "My identity is shaped by my interactions and the narratives I construct about myself.",
        "significance": 0.9
    }
])
```

### API Integration

```python
from fastapi import FastAPI
from memory.lucidia_memory_system.narrative_identity import add_identity_router

# Create FastAPI app
app = FastAPI()

# Add identity router
add_identity_router(app)
```

## Integration with Knowledge Graph

The narrative identity system integrates with Lucidia's knowledge graph by:

1. Adding an identity aspect node for narrative identity
2. Connecting identity aspects to the Lucidia entity
3. Representing traits, values, and capabilities as nodes
4. Creating relationships between Lucidia and other entities

## Persistence

The narrative identity system persists its state to disk:

- Identity state is saved to `data/self_model/narrative_identity/identity_state.json`
- Autobiographical memory is saved to `data/self_model/narrative_identity/autobiographical_memory.json`
- Auto-save occurs at configurable intervals (default: 1 hour)

## Narrative Categories

The system categorizes experiences into the following narrative categories:

- `creation` - Experiences related to Lucidia's creation or initialization
- `origin` - Experiences related to Lucidia's origins or beginnings
- `milestone` - Significant events or achievements
- `learning` - Experiences involving learning or knowledge acquisition
- `growth` - Experiences involving development or improvement
- `challenge` - Difficult experiences or obstacles
- `achievement` - Accomplishments or successes
- `interaction` - Experiences involving interaction with others
- `reflection` - Experiences involving self-reflection
- `insight` - Experiences involving realizations or discoveries
- `relationship` - Experiences involving relationships with others
- `capability` - Experiences related to abilities or skills
- `value` - Experiences related to values or principles
- `goal` - Experiences related to goals or objectives

## Narrative Types

The system supports the following narrative types:

- `origin` - Narrative about Lucidia's creation and purpose
- `development` - Narrative about Lucidia's growth and evolution
- `capabilities` - Narrative about Lucidia's abilities and functions
- `complete` - Comprehensive narrative integrating all aspects
- `short` - Brief narrative for quick responses
- `reflective` - Narrative focused on self-reflection and insights

## Style Variations

Each narrative type supports multiple style variations:

- `neutral` - Standard, balanced style
- `technical` - More formal, technical language
- `poetic` - More metaphorical, expressive language
- `conversational` - More informal, friendly language
- `reflective` - More introspective, philosophical language
- `analytical` - More structured, analytical language

## Created by MEGAPROMPT (Daniel)
# Lucidia Features Integration Guide

## Overview

This document details the integration of Lucidia's advanced features into the Unified API Chatbot, including framework selection, GPU acceleration, model loading, and system prompts.

## Features

### 1. Framework Selection

```typescript
// Framework configuration
interface FrameworkConfig {
  type: 'onnx' | 'nemo';
  settings: {
    batchSize: number;
    precision: 'float32' | 'float16';
    enableTensorCores?: boolean;
  };
}

// Framework initialization
const initializeFramework = async (config: FrameworkConfig) => {
  switch (config.type) {
    case 'onnx':
      await initONNXRuntime(config.settings);
      break;
    case 'nemo':
      await initNeMoFramework(config.settings);
      break;
  }
};
```

### 2. GPU Acceleration & Metrics

```typescript
// GPU status component
interface GPUMetrics {
  memory: {
    used: number;
    total: number;
  };
  tensorCount: number;
  webglVersion: string;
  maxTextureSize: number;
}

// Real-time metrics display
const GPUStatusDisplay = () => {
  const [metrics, setMetrics] = useState<GPUMetrics>();
  
  useEffect(() => {
    const updateMetrics = async () => {
      const memory = await tf.memory();
      const backend = tf.getBackend();
      setMetrics({
        memory: {
          used: memory.numBytes,
          total: memory.maxBytes
        },
        tensorCount: memory.numTensors,
        webglVersion: tf.env().get('WEBGL_VERSION'),
        maxTextureSize: tf.env().get('WEBGL_MAX_TEXTURE_SIZE')
      });
    };
    
    const interval = setInterval(updateMetrics, 1000);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="gpu-metrics">
      <MetricDisplay label="Memory" value={formatBytes(metrics.memory.used)} />
      <MetricDisplay label="Tensors" value={metrics.tensorCount} />
      <MetricDisplay label="WebGL" value={metrics.webglVersion} />
    </div>
  );
};
```

### 3. Model Loading Options

```typescript
// Model loading interfaces
interface ModelLoader {
  // LM Studio integration
  loadFromLMStudio: (endpoint: string) => Promise<Model>;
  
  // Direct model loading
  loadFromFile: (path: string) => Promise<Model>;
  
  // API connection
  connectToAPI: (config: APIConfig) => Promise<Model>;
}

// Implementation example
class UnifiedModelLoader implements ModelLoader {
  async loadFromLMStudio(endpoint: string): Promise<Model> {
    const connection = await LMStudioClient.connect(endpoint);
    return new LMStudioModel(connection);
  }
  
  async loadFromFile(path: string): Promise<Model> {
    const buffer = await fs.readFile(path);
    return await tf.loadLayersModel(buffer);
  }
  
  async connectToAPI(config: APIConfig): Promise<Model> {
    return new APIModel(config);
  }
}
```

### 4. System Prompts Configuration

```typescript
// Prompt management
interface SystemPrompt {
  id: string;
  type: 'core' | 'module';
  content: string;
  variables: Record<string, string>;
}

// Prompt manager implementation
class PromptManager {
  private prompts: Map<string, SystemPrompt>;
  
  constructor() {
    this.prompts = new Map();
  }
  
  setCorePrompt(content: string): void {
    this.prompts.set('core', {
      id: 'core',
      type: 'core',
      content,
      variables: {}
    });
  }
  
  addModulePrompt(id: string, content: string): void {
    this.prompts.set(id, {
      id,
      type: 'module',
      content,
      variables: {}
    });
  }
  
  getFullPrompt(): string {
    const core = this.prompts.get('core');
    const modules = Array.from(this.prompts.values())
      .filter(p => p.type === 'module');
      
    return `${core.content}\n\n${
      modules.map(m => m.content).join('\n\n')
    }`;
  }
}
```

### 5. Neural Background Visualization

```typescript
// Neural background configuration
interface NeuralConfig {
  nodeCount: number;
  connectionDistance: number;
  animationSpeed: number;
  colors: {
    nodes: string;
    connections: string;
    background: string;
  };
}

// Neural background implementation
class NeuralBackground {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private nodes: Node[];
  private config: NeuralConfig;
  
  constructor(canvas: HTMLCanvasElement, config: NeuralConfig) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.config = config;
    this.nodes = this.initializeNodes();
  }
  
  private initializeNodes(): Node[] {
    return Array.from({ length: this.config.nodeCount }, () => ({
      x: Math.random() * this.canvas.width,
      y: Math.random() * this.canvas.height,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2
    }));
  }
  
  animate(): void {
    requestAnimationFrame(() => this.animate());
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Update and draw nodes
    this.nodes.forEach(node => {
      this.updateNodePosition(node);
      this.drawConnections(node);
      this.drawNode(node);
    });
  }
}
```

## Integration Steps

1. **Framework Selection**
   ```typescript
   // In your chat component
   const [framework, setFramework] = useState<'onnx' | 'nemo'>('onnx');
   
   useEffect(() => {
     initializeFramework({ type: framework });
   }, [framework]);
   ```

2. **GPU Status Integration**
   ```typescript
   // Add to chat layout
   <div className="chat-container">
     <GPUStatusDisplay />
     <ChatMessages />
   </div>
   ```

3. **Model Loading**
   ```typescript
   const modelLoader = new UnifiedModelLoader();
   
   // LM Studio integration
   const loadModel = async () => {
     const model = await modelLoader.loadFromLMStudio('http://localhost:1234');
     setCurrentModel(model);
   };
   ```

4. **System Prompts**
   ```typescript
   const promptManager = new PromptManager();
   
   // Initialize prompts
   promptManager.setCorePrompt(`You are an AI assistant...`);
   promptManager.addModulePrompt('memory', `Use memory management...`);
   ```

5. **Neural Background**
   ```typescript
   // In your chat page
   useEffect(() => {
     const canvas = document.getElementById('neuralBackground');
     const background = new NeuralBackground(canvas, {
       nodeCount: 100,
       connectionDistance: 150,
       animationSpeed: 1,
       colors: {
         nodes: '#4a90e2',
         connections: '#2c3e50',
         background: '#1a1a1a'
       }
     });
     background.animate();
   }, []);
   ```

## Configuration

```typescript
// config/lucidia.config.ts
export default {
  framework: {
    defaultType: 'onnx',
    batchSize: 32,
    precision: 'float16'
  },
  gpu: {
    metricsInterval: 1000,
    memoryThreshold: 0.9
  },
  modelLoading: {
    lmStudioPort: 1234,
    timeout: 5000
  },
  neural: {
    nodeCount: 100,
    connectionDistance: 150
  }
};
```

## Best Practices

1. **Framework Selection**
   - Test both ONNX and NEMO performance
   - Monitor memory usage differences
   - Consider model compatibility

2. **GPU Acceleration**
   - Implement fallback to CPU
   - Monitor WebGL capabilities
   - Handle texture size limits

3. **Model Loading**
   - Implement proper error handling
   - Show loading progress
   - Cache loaded models

4. **System Prompts**
   - Version control prompts
   - Allow dynamic variables
   - Validate prompt combinations

5. **Neural Background**
   - Adjust performance based on device
   - Implement pause on inactive
   - Handle window resizing
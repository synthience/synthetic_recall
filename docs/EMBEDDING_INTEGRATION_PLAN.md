# Embedding Integration Plan
*Replacing simulated embeddings with real ML models*

## 1. Model Integration Options

### Option A: GGUF Integration
```javascript
// src/services/implementations/gguf/EmbeddingService.cjs
const { spawn } = require('child_process');
const path = require('path');

class GGUFEmbeddingService {
    constructor(modelPath) {
        this.modelPath = modelPath;
        this.process = null;
    }

    async initialize() {
        // Launch Python subprocess for GGUF
        this.process = spawn('python', [
            path.join(__dirname, 'gguf_embeddings.py'),
            '--model', this.modelPath
        ]);
        
        // Handle process events
        this.process.on('error', (err) => console.error('GGUF Error:', err));
        this.process.on('exit', (code) => console.log('GGUF Exit:', code));
    }

    async generateEmbedding(text) {
        // Send text to Python process
        return new Promise((resolve, reject) => {
            this.process.stdin.write(JSON.stringify({ text }) + '\n');
            
            this.process.stdout.once('data', (data) => {
                const embedding = JSON.parse(data);
                resolve(embedding);
            });
        });
    }
}
```

### Option B: Hugging Face Integration
```javascript
// src/services/implementations/huggingface/EmbeddingService.cjs
const { HfInference } = require('@huggingface/inference');

class HuggingFaceEmbeddingService {
    constructor(modelId, apiKey) {
        this.hf = new HfInference(apiKey);
        this.modelId = modelId;
    }

    async initialize() {
        // Verify model access and warm up
        await this.hf.featureExtraction({
            model: this.modelId,
            inputs: 'test'
        });
    }

    async generateEmbedding(text) {
        const result = await this.hf.featureExtraction({
            model: this.modelId,
            inputs: text
        });
        return result;
    }
}
```

## 2. Integration Steps

1. Environment Setup
```bash
# Install dependencies
npm install @huggingface/inference
pip install torch transformers

# Set environment variables
export HF_API_KEY=your_key_here
export GGUF_MODEL_PATH=/path/to/model.gguf
```

2. Service Factory Update
```javascript
// src/services/EmbeddingServiceFactory.cjs
class EmbeddingServiceFactory {
    static async create(type = 'huggingface') {
        switch (type) {
            case 'gguf':
                const GGUFService = require('./implementations/gguf/EmbeddingService');
                return new GGUFService(process.env.GGUF_MODEL_PATH);
                
            case 'huggingface':
                const HFService = require('./implementations/huggingface/EmbeddingService');
                return new HFService(
                    'sentence-transformers/all-MiniLM-L6-v2',
                    process.env.HF_API_KEY
                );
                
            default:
                throw new Error(`Unknown embedding type: ${type}`);
        }
    }
}
```

3. Integration Test Dataset
```javascript
// test/embedding-validation.js
const testData = [
    {
        text: 'function declaration in JavaScript',
        expected_length: 384  // Model-specific dimension
    },
    {
        text: 'class inheritance in Python',
        expected_length: 384
    },
    // Add 8-10 more programming concepts
];
```

## 3. Validation Process

1. Basic Validation
```javascript
async function validateEmbeddings(service) {
    for (const test of testData) {
        const embedding = await service.generateEmbedding(test.text);
        
        // Validate dimension
        assert(embedding.length === test.expected_length);
        
        // Validate range (-1 to 1)
        assert(embedding.every(v => v >= -1 && v <= 1));
        
        // Validate norm
        const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
        assert(Math.abs(norm - 1.0) < 1e-6);
    }
}
```

2. Performance Metrics
```javascript
async function measurePerformance(service) {
    const start = process.hrtime();
    
    await Promise.all(testData.map(test => 
        service.generateEmbedding(test.text)
    ));
    
    const [seconds, nanoseconds] = process.hrtime(start);
    return seconds + nanoseconds / 1e9;
}
```

## 4. Integration with Existing Architecture

1. Update HPCService
```javascript
// src/services/implementations/cuda/HPCService.cjs
class CUDAHPCService {
    async initialize() {
        // ... existing initialization ...
        this.embeddingService = await EmbeddingServiceFactory.create();
        await this.embeddingService.initialize();
    }

    async processEmbeddings(params) {
        const embedding = await this.embeddingService.generateEmbedding(
            params.text
        );
        
        // Apply any additional GPU processing
        return {
            embeddings: embedding,
            metadata: {
                model: this.embeddingService.modelId,
                dimension: embedding.length
            }
        };
    }
}
```

2. Error Handling
```javascript
try {
    const embedding = await this.embeddingService.generateEmbedding(text);
} catch (error) {
    if (error.name === 'HuggingFaceError') {
        // API-specific error handling
    } else if (error.name === 'GGUFError') {
        // Local model error handling
    }
    throw error;
}
```

## 5. Minimal Test Run

```bash
# Run validation
node test/embedding-validation.js

# Expected output:
# ✓ Dimension validation passed
# ✓ Value range validation passed
# ✓ Vector normalization passed
# ✓ Average generation time: 0.15s
```

## Next Steps

1. Choose between GGUF (local) or Hugging Face (API) based on:
   - Performance requirements
   - Privacy considerations
   - Cost constraints

2. Implement the chosen solution:
   - Install required dependencies
   - Set up environment variables
   - Run validation tests
   - Monitor initial performance

3. Gradually migrate:
   - Start with a small subset of requests
   - Monitor error rates and performance
   - Scale up based on stability
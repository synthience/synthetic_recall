import express, { Router, Request, Response, RequestHandler } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import axios, { AxiosError } from "axios";

// Define API endpoints for the various services
const MEMORY_CORE_URL = process.env.MEMORY_CORE_URL || "http://localhost:5010";
const NEURAL_MEMORY_URL = process.env.NEURAL_MEMORY_URL || "http://localhost:8001";
const CCE_URL = process.env.CCE_URL || "http://localhost:8002";

// Enable mock mode for development without backend services
const USE_MOCK_DATA = process.env.USE_MOCK_DATA === "true" || false;

// Console logging helper for server-side logs
const log = (message: string) => {
  console.log(`[Dashboard Server] ${message}`);
};

// Mock data for development
const mockData = {
  memoryCore: {
    health: { status: "OK", uptime: "2d 5h 32m" },
    stats: {
      memory_count: 1250,
      assembly_count: 48,
      vector_index_size: 1298,
      assembly_stats: {
        activation_counts: {
          "asm_1": 87,
          "asm_2": 65,
          "asm_3": 42
        }
      }
    },
    assemblies: [
      { id: "asm_1", name: "Core Concepts", created_at: "2025-03-28T14:32:11", memory_count: 28, vector_index_updated_at: "2025-03-28T14:35:21", merged_from: ["asm_10", "asm_15"] },
      { id: "asm_2", name: "System Architecture", created_at: "2025-03-29T09:12:05", memory_count: 42, vector_index_updated_at: "2025-03-29T09:15:30" },
      { id: "asm_3", name: "Implementation Details", created_at: "2025-03-30T17:05:22", memory_count: 35, vector_index_updated_at: "2025-03-30T17:10:15" }
    ],
    assembly: {
      id: "asm_1",
      name: "Core Concepts",
      created_at: "2025-03-28T14:32:11",
      memory_count: 28,
      vector_index_updated_at: "2025-03-28T14:35:21",
      merged_from: ["asm_10", "asm_15"],
      memories: [
        { id: "mem_1", title: "Memory System Architecture", content: "The memory system architecture consists of...", created_at: "2025-03-28T14:30:00" },
        { id: "mem_2", title: "Vector Indexing Approach", content: "Our vector indexing approach uses FAISS to...", created_at: "2025-03-28T14:31:15" }
      ]
    },
    explainActivation: {
      data: {
        assembly_id: "asm_1",
        memory_id: "mem_1",
        timestamp: "2025-04-01T15:30:22",
        context: "User query about memory architecture",
        similarity_score: 0.89,
        threshold: 0.75,
        passed_threshold: true,
        notes: "Strong match based on vector similarity and recency boost"
      }
    },
    explainMerge: {
      data: {
        target_id: "asm_1",
        event_id: "merge_ev_123",
        timestamp: "2025-03-28T14:32:11",
        sources: [
          { id: "asm_10", name: "Memory Concepts Draft" },
          { id: "asm_15", name: "Architecture Notes" }
        ],
        similarity_at_merge: 0.82,
        threshold_used: 0.75,
        cleanup_status: "completed",
        cleanup_details: "Source assemblies archived successfully",
        notes: "Merge triggered by high conceptual overlap"
      }
    },
    lineage: [
      { depth: 0, id: "asm_1", name: "Core Concepts", status: "current", created_at: "2025-03-28T14:32:11", memory_count: 28 },
      { depth: 1, id: "asm_10", name: "Memory Concepts Draft", status: "merged", created_at: "2025-03-27T10:15:30", memory_count: 15 },
      { depth: 1, id: "asm_15", name: "Architecture Notes", status: "merged", created_at: "2025-03-27T11:42:18", memory_count: 13 },
      { depth: 2, id: "asm_5", name: "Initial Notes", status: "archived", created_at: "2025-03-26T09:30:00", memory_count: 8 }
    ],
    mergeLog: [
      {
        event_id: "merge_ev_123",
        creation_time: "2025-03-28T14:32:11",
        sources: ["asm_10", "asm_15"],
        target: "asm_1",
        similarity_at_merge: 0.82,
        threshold_used: 0.75,
        final_status: "completed",
        cleanup_time: "2025-03-28T14:35:21",
        error: null
      },
      {
        event_id: "merge_ev_124",
        creation_time: "2025-03-29T09:12:05",
        sources: ["asm_20", "asm_25"],
        target: "asm_2",
        similarity_at_merge: 0.79,
        threshold_used: 0.75,
        final_status: "completed",
        cleanup_time: "2025-03-29T09:15:30",
        error: null
      }
    ],
    config: {
      memory_core: {
        ENABLE_EXPLAINABILITY: true,
        ASSEMBLY_METRICS_PERSIST_INTERVAL: 300,
        MAX_LINEAGE_DEPTH: 5,
        MERGE_LOG_PATH: "/var/log/memory-core/merge_log.jsonl"
      },
      neural_memory: {
        LEARNING_RATE: 0.001,
        BATCH_SIZE: 32,
        TITANS_VARIANTS: ["MAC", "MAG", "MAL"]
      },
      cce: {
        DEFAULT_THRESHOLD: 0.75,
        LLM_GUIDANCE_ENABLED: true,
        VARIANT_SELECTION_STRATEGY: "adaptive"
      }
    }
  },
  neuralMemory: {
    health: { status: "OK", uptime: "2d 4h 15m" },
    status: { state: "ready", mode: "training" },
    config: {
      LEARNING_RATE: 0.001,
      BATCH_SIZE: 32,
      TITANS_VARIANTS: ["MAC", "MAG", "MAL"]
    },
    diagnoseEmoloop: {
      trainingLoss: [
        { timestamp: "2025-04-01T00:00:00", value: 0.15 },
        { timestamp: "2025-04-01T06:00:00", value: 0.12 },
        { timestamp: "2025-04-01T12:00:00", value: 0.10 },
        { timestamp: "2025-04-01T18:00:00", value: 0.09 },
        { timestamp: "2025-04-02T00:00:00", value: 0.08 }
      ],
      emotionDistribution: {
        joy: 0.25,
        sadness: 0.15,
        anger: 0.10,
        fear: 0.05,
        surprise: 0.20,
        disgust: 0.05,
        trust: 0.20
      }
    }
  },
  cce: {
    health: { status: "OK", uptime: "2d 5h 10m" },
    status: { state: "ready", mode: "production" },
    config: {
      DEFAULT_THRESHOLD: 0.75,
      LLM_GUIDANCE_ENABLED: true,
      VARIANT_SELECTION_STRATEGY: "adaptive"
    },
    metrics: {
      recentResponses: [
        {
          timestamp: "2025-04-01T15:30:22",
          input_text: "Tell me about the memory architecture",
          selected_variant: "MAC",
          selection_reason: "High similarity to previous successful interactions",
          response_time_ms: 245,
          llm_advice_used: true
        },
        {
          timestamp: "2025-04-01T15:35:16",
          input_text: "How does vector indexing work?",
          selected_variant: "MAG",
          selection_reason: "Input complexity suggests deeper reasoning required",
          response_time_ms: 310,
          llm_advice_used: true
        },
        {
          timestamp: "2025-04-01T15:40:05",
          input_text: "What are memory assemblies?",
          selected_variant: "MAL",
          selection_reason: "Topic requires extensive conceptual integration",
          response_time_ms: 380,
          llm_advice_used: false
        }
      ]
    }
  }
};

// Type definition for service names to avoid TypeScript errors
type ServiceName = 'memory_core' | 'neural_memory' | 'cce';

// Helper function for proxying requests
async function proxyRequest(req: Request, res: Response, targetUrl: string, serviceName: string) {
  const method = req.method;
  // Construct target URL: remove the /api/<service-name> prefix
  const targetPath = req.originalUrl.replace(`/api/${serviceName}`, '');
  const url = targetUrl + targetPath;

  log(`Proxying ${method} ${req.originalUrl} to ${url}`);

  try {
    const response = await axios({
      method: method as any,
      url: url,
      params: req.query, // Forward query parameters
      data: method !== 'GET' && method !== 'HEAD' ? req.body : undefined, // Forward body for non-GET/HEAD
      headers: {
        'Content-Type': req.headers['content-type'] || 'application/json',
      },
      timeout: 20000 // 20 second timeout
    });
    res.status(response.status).json(response.data);
  } catch (error: any) {
    log(`Proxy Error for ${serviceName} to ${url}: ${error.message}`);
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      const status = axiosError.response?.status || 500;
      const errorData = axiosError.response?.data || axiosError.message;
      
      // Extract a more specific error message if available
      const message = (typeof errorData === 'object' && errorData !== null && 'detail' in errorData)
                      ? errorData.detail
                      : (typeof errorData === 'object' && errorData !== null && 'error' in errorData)
                        ? errorData.error
                        : String(errorData);

      res.status(status).json({
        success: false,
        message: `Failed request to ${serviceName}: ${message}`,
        details: errorData 
      });
    } else {
      res.status(500).json({
        success: false,
        message: `Unknown proxy error for ${serviceName}: ${error.message}`
      });
    }
  }
}

export async function registerRoutes(app: express.Express): Promise<Server> {
  // Create a router instance for API routes
  const apiRouter = Router();
  
  // Memory Core routes
  apiRouter.get("/memory-core/health", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.memoryCore.health);
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.get("/memory-core/stats", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.memoryCore.stats);
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.memoryCore.assemblies);
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      const assembly = mockData.memoryCore.assembly;
      if (assembly.id === req.params.id) {
        res.json(assembly);
      } else {
        res.status(404).json({ status: "Error", message: "Assembly not found" });
      }
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  // Phase 5.9 Explainability endpoints
  apiRouter.get("/memory-core/assemblies/:id/lineage", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      const lineage = mockData.memoryCore.lineage;
      if (lineage[0].id === req.params.id) {
        res.json(lineage);
      } else {
        res.status(404).json({ status: "Error", message: "Assembly not found" });
      }
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id/explain_merge", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      const explainMerge = mockData.memoryCore.explainMerge;
      if (explainMerge.data.target_id === req.params.id) {
        res.json(explainMerge);
      } else {
        res.status(404).json({ status: "Error", message: "Assembly not found or no merge data available" });
      }
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id/explain_activation", ((req: Request, res: Response) => {
    const memory_id = req.query.memory_id;
    if (!memory_id) {
      return res.status(400).json({ status: "Error", message: "memory_id parameter is required" });
    }
    
    if (USE_MOCK_DATA) {
      const explainActivation = mockData.memoryCore.explainActivation;
      if (explainActivation.data.assembly_id === req.params.id && explainActivation.data.memory_id === memory_id) {
        res.json(explainActivation);
      } else {
        res.status(404).json({ status: "Error", message: "Assembly or memory not found" });
      }
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.get("/memory-core/diagnostics/merge_log", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.memoryCore.mergeLog);
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.get("/memory-core/config/runtime/:service", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      const serviceParam = req.params.service;
      let service: ServiceName;
      
      // Map the URL parameter to our internal service names
      if (serviceParam === 'memory-core') {
        service = 'memory_core';
      } else if (serviceParam === 'neural-memory') {
        service = 'neural_memory';
      } else if (serviceParam === 'cce') {
        service = 'cce';
      } else {
        return res.status(404).json({ status: "Error", message: "Service not found" });
      }
      
      const config = mockData.memoryCore.config;
      res.json(config[service]);
    } else {
      // Special handling for config endpoint - directly construct the URL with the correct path structure
      const serviceParam = req.params.service;
      // Use the correct endpoint path structure for Memory Core config
      const targetUrl = `${MEMORY_CORE_URL}/config/runtime/${serviceParam}`;
      
      log(`Proxying GET ${req.originalUrl} to ${targetUrl} (special handling for config)`);
      
      axios.get(targetUrl, {
        params: req.query,
        headers: {
          'Content-Type': req.headers['content-type'] || 'application/json',
        },
        timeout: 20000 // 20 second timeout
      })
      .then(response => {
        res.status(response.status).json(response.data);
      })
      .catch(error => {
        log(`Proxy Error for memory-core config to ${targetUrl}: ${error.message}`);
        if (axios.isAxiosError(error)) {
          const axiosError = error as AxiosError;
          const status = axiosError.response?.status || 500;
          const errorData = axiosError.response?.data || axiosError.message;
          
          // Extract a more specific error message if available
          const message = (typeof errorData === 'object' && errorData !== null && 'detail' in errorData)
                          ? errorData.detail
                          : (typeof errorData === 'object' && errorData !== null && 'error' in errorData)
                            ? errorData.error
                            : String(errorData);

          res.status(status).json({
            success: false,
            message: `Failed request to memory-core config: ${message}`,
            details: errorData 
          });
        } else {
          res.status(500).json({
            success: false,
            message: `Unknown proxy error for memory-core config: ${error.message}`
          });
        }
      });
    }
  }) as RequestHandler);

  // Neural Memory routes
  apiRouter.get("/neural-memory/health", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.neuralMemory.health);
    } else {
      proxyRequest(req, res, NEURAL_MEMORY_URL, 'neural-memory');
    }
  }) as RequestHandler);

  apiRouter.get("/neural-memory/status", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.neuralMemory.status);
    } else {
      proxyRequest(req, res, NEURAL_MEMORY_URL, 'neural-memory');
    }
  }) as RequestHandler);

  apiRouter.get("/neural-memory/diagnose_emoloop", ((req: Request, res: Response) => {
    const window = req.query.window || "24h";
    if (USE_MOCK_DATA) {
      res.json(mockData.neuralMemory.diagnoseEmoloop);
    } else {
      proxyRequest(req, res, NEURAL_MEMORY_URL, 'neural-memory');
    }
  }) as RequestHandler);

  // Context Cascade Engine routes
  apiRouter.get("/cce/health", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.cce.health);
    } else {
      proxyRequest(req, res, CCE_URL, 'cce');
    }
  }) as RequestHandler);

  apiRouter.get("/cce/status", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.cce.status);
    } else {
      proxyRequest(req, res, CCE_URL, 'cce');
    }
  }) as RequestHandler);

  apiRouter.get("/cce/metrics/recent_cce_responses", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.cce.metrics.recentResponses);
    } else {
      proxyRequest(req, res, CCE_URL, 'cce');
    }
  }) as RequestHandler);

  // Configuration endpoints
  apiRouter.get("/neural-memory/config", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.neuralMemory.config);
    } else {
      proxyRequest(req, res, NEURAL_MEMORY_URL, 'neural-memory');
    }
  }) as RequestHandler);

  apiRouter.get("/cce/config", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json(mockData.cce.config);
    } else {
      proxyRequest(req, res, CCE_URL, 'cce');
    }
  }) as RequestHandler);

  // Admin action endpoints
  apiRouter.post("/memory-core/admin/verify_index", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json({ status: "OK", message: "Mock index verification successful" });
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.post("/memory-core/admin/trigger_retry_loop", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json({ status: "OK", message: "Mock retry loop triggered successfully" });
    } else {
      proxyRequest(req, res, MEMORY_CORE_URL, 'memory-core');
    }
  }) as RequestHandler);

  apiRouter.post("/neural-memory/init", ((req: Request, res: Response) => {
    if (USE_MOCK_DATA) {
      res.json({ status: "OK", message: "Mock Neural Memory initialization successful" });
    } else {
      proxyRequest(req, res, NEURAL_MEMORY_URL, 'neural-memory');
    }
  }) as RequestHandler);

  apiRouter.post("/cce/set_variant", ((req: Request, res: Response) => {
    const { variant } = req.body;
    if (!variant) {
      return res.status(400).json({ status: "Error", message: "Variant parameter is required" });
    }
    if (USE_MOCK_DATA) {
      res.json({ status: "OK", message: `Mock variant set to ${variant}` });
    } else {
      proxyRequest(req, res, CCE_URL, 'cce');
    }
  }) as RequestHandler);

  // Alerts API (for demonstration)
  apiRouter.get("/alerts", ((req: Request, res: Response) => {
    storage.getAlerts()
      .then(alerts => {
        res.json(alerts);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch alerts" });
      });
  }) as RequestHandler);

  // Mount the router on the app
  app.use("/api", apiRouter);

  // Create the HTTP server
  const server = createServer(app);
  return server;
}

import express, { Router, Request, Response, RequestHandler } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import axios from "axios";

// Define API endpoints for the various services
const MEMORY_CORE_URL = process.env.MEMORY_CORE_URL || "http://memory-core:8080";
const NEURAL_MEMORY_URL = process.env.NEURAL_MEMORY_URL || "http://neural-memory:8080";
const CCE_URL = process.env.CCE_URL || "http://cce:8080";

export async function registerRoutes(app: express.Express): Promise<Server> {
  // Create a router instance for API routes
  const apiRouter = Router();
  
  // Memory Core routes
  apiRouter.get("/memory-core/health", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/health`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to connect to Memory Core service" });
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/stats", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/stats`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch Memory Core stats" });
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/assemblies`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch assemblies" });
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/assemblies/${req.params.id}`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          res.status(404).json({ status: "Error", message: "Assembly not found" });
        } else {
          res.status(500).json({ status: "Error", message: "Failed to fetch assembly" });
        }
      });
  }) as RequestHandler);

  // Phase 5.9 Explainability endpoints
  apiRouter.get("/memory-core/assemblies/:id/lineage", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/assemblies/${req.params.id}/lineage`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          res.status(404).json({ status: "Error", message: "Assembly not found" });
        } else {
          res.status(500).json({ status: "Error", message: "Failed to fetch assembly lineage" });
        }
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id/explain_merge", ((req: Request, res: Response) => {
    axios.get(`${MEMORY_CORE_URL}/assemblies/${req.params.id}/explain_merge`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          res.status(404).json({ status: "Error", message: "Assembly not found or no merge data available" });
        } else {
          res.status(500).json({ status: "Error", message: "Failed to fetch merge explanation" });
        }
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/assemblies/:id/explain_activation", ((req: Request, res: Response) => {
    const memory_id = req.query.memory_id;
    if (!memory_id) {
      return res.status(400).json({ status: "Error", message: "memory_id parameter is required" });
    }
    
    axios.get(`${MEMORY_CORE_URL}/assemblies/${req.params.id}/explain_activation`, { params: { memory_id } })
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
          res.status(404).json({ status: "Error", message: "Assembly or memory not found" });
        } else {
          res.status(500).json({ status: "Error", message: "Failed to fetch activation explanation" });
        }
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/diagnostics/merge_log", ((req: Request, res: Response) => {
    const limit = req.query.limit || 50;
    axios.get(`${MEMORY_CORE_URL}/diagnostics/merge_log`, { params: { limit } })
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch merge log" });
      });
  }) as RequestHandler);

  apiRouter.get("/memory-core/config/runtime/:service", ((req: Request, res: Response) => {
    const service = req.params.service;
    axios.get(`${MEMORY_CORE_URL}/config/runtime/${service}`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch runtime configuration" });
      });
  }) as RequestHandler);

  // Neural Memory routes
  apiRouter.get("/neural-memory/health", ((req: Request, res: Response) => {
    axios.get(`${NEURAL_MEMORY_URL}/health`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to connect to Neural Memory service" });
      });
  }) as RequestHandler);

  apiRouter.get("/neural-memory/status", ((req: Request, res: Response) => {
    axios.get(`${NEURAL_MEMORY_URL}/status`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch Neural Memory status" });
      });
  }) as RequestHandler);

  apiRouter.get("/neural-memory/diagnose_emoloop", ((req: Request, res: Response) => {
    const window = req.query.window || "24h";
    axios.get(`${NEURAL_MEMORY_URL}/diagnose_emoloop?window=${window}`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch emotional loop diagnostics" });
      });
  }) as RequestHandler);

  // Context Cascade Engine routes
  apiRouter.get("/cce/health", ((req: Request, res: Response) => {
    axios.get(`${CCE_URL}/health`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to connect to CCE service" });
      });
  }) as RequestHandler);

  apiRouter.get("/cce/status", ((req: Request, res: Response) => {
    axios.get(`${CCE_URL}/status`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch CCE status" });
      });
  }) as RequestHandler);

  apiRouter.get("/cce/metrics/recent_cce_responses", ((req: Request, res: Response) => {
    axios.get(`${CCE_URL}/metrics/recent_cce_responses`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch recent CCE responses" });
      });
  }) as RequestHandler);

  // Configuration endpoints
  apiRouter.get("/neural-memory/config", ((req: Request, res: Response) => {
    axios.get(`${NEURAL_MEMORY_URL}/config`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch Neural Memory config" });
      });
  }) as RequestHandler);

  apiRouter.get("/cce/config", ((req: Request, res: Response) => {
    axios.get(`${CCE_URL}/config`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to fetch CCE config" });
      });
  }) as RequestHandler);

  // Admin action endpoints
  apiRouter.post("/memory-core/admin/verify_index", ((req: Request, res: Response) => {
    axios.post(`${MEMORY_CORE_URL}/admin/verify_index`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to trigger index verification" });
      });
  }) as RequestHandler);

  apiRouter.post("/memory-core/admin/trigger_retry_loop", ((req: Request, res: Response) => {
    axios.post(`${MEMORY_CORE_URL}/admin/trigger_retry_loop`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to trigger retry loop" });
      });
  }) as RequestHandler);

  apiRouter.post("/neural-memory/init", ((req: Request, res: Response) => {
    axios.post(`${NEURAL_MEMORY_URL}/init`)
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to initialize Neural Memory" });
      });
  }) as RequestHandler);

  apiRouter.post("/cce/set_variant", ((req: Request, res: Response) => {
    const { variant } = req.body;
    if (!variant) {
      return res.status(400).json({ status: "Error", message: "Variant parameter is required" });
    }
    axios.post(`${CCE_URL}/set_variant`, { variant })
      .then(response => {
        res.json(response.data);
      })
      .catch(error => {
        res.status(500).json({ status: "Error", message: "Failed to set CCE variant" });
      });
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

import express, { type Request, type Response, type NextFunction } from "express"; 
import axios, { AxiosError, Method } from "axios";
import { storage, LogMessage } from "./storage"; 

const MEMORY_CORE_URL = process.env.MEMORY_CORE_URL || "http://localhost:5020";
const NEURAL_MEMORY_URL = process.env.NEURAL_MEMORY_URL || "http://localhost:8001";
const CCE_URL = process.env.CCE_URL || "http://localhost:8002";

const log = (message: string): void => {
  console.log(`[Dashboard Server] ${message}`);
};

interface SuccessPayload<T = any> {
  success: true;
  data: T;
}

interface ErrorPayload {
  success: false;
  error: string;
  details: any;
  proxy_target: string;
  original_path: string;
}

interface FallbackConfig {
  service: string;
  config: {
    ENABLE_EXPLAINABILITY: boolean;
    DEBUG_MODE: boolean;
    LOG_LEVEL: string;
  };
  retrieval_timestamp: string;
  _note: string;
  _original_error: string;
}

type ProxyResponse = SuccessPayload | ErrorPayload;

function isConfigRequest(path: string): boolean {
  return (
    /\/api\/(proxy\/)?(memory-core|neural-memory|cce)\/config\/runtime\//.test(path)
  );
}

async function proxyRequest(
  req: Request,
  res: Response,
  targetUrlBase: string,
  serviceName: string,
  overridePath?: string
): Promise<Response | void> {
  const method = req.method as Method;
  const originalPath = req.originalUrl;

  let relativePath = overridePath ?? originalPath.replace(`/api/${serviceName}`, "");
  if (relativePath === `/${serviceName}` || relativePath === "") {
    relativePath = "/";
  } else if (!relativePath.startsWith("/")) {
    relativePath = "/" + relativePath;
  }

  const targetUrl = targetUrlBase + relativePath;

  log(`[Proxy] ${method} ${originalPath} -> ${targetUrl}`);

  try {
    const axiosOptions = {
      method,
      url: targetUrl,
      headers: {
        ...req.headers,
        host: new URL(targetUrlBase).host, 
      },
      data: req.body,
      validateStatus: () => true,
      timeout: 120000,
    };

    delete axiosOptions.headers["content-length"];

    const response = await axios(axiosOptions);

    if (response.status >= 200 && response.status < 300) {
      const successResponse: SuccessPayload = {
        success: true,
        data: response.data,
      };

      return res.status(response.status).json(successResponse);
    }

    log(`[Proxy] Error response from ${targetUrl}: ${response.status}`);

    if (isConfigRequest(originalPath) && response.status >= 400) {
      log(
        `[Proxy] Providing fallback config data for failed ${serviceName} config request`
      );

      const fallbackConfig: FallbackConfig = {
        service: serviceName,
        config: {
          ENABLE_EXPLAINABILITY: true,
          DEBUG_MODE: true,
          LOG_LEVEL: "debug",
        },
        retrieval_timestamp: new Date().toISOString(),
        _note: "FALLBACK CONFIG - The actual service may be unavailable",
        _original_error: `HTTP ${response.status} from ${targetUrl}`,
      };

      const fallbackResponse: SuccessPayload = {
        success: true,
        data: fallbackConfig,
      };

      return res.status(200).json(fallbackResponse);
    }

    const errorResponse: ErrorPayload = {
      success: false,
      error: `Error proxying to ${serviceName}: ${response.status} ${response.statusText}`,
      details: response.data,
      proxy_target: targetUrl,
      original_path: originalPath,
    };

    return res.status(response.status).json(errorResponse);
  } catch (error: unknown) {
    let errorMessage = "Unknown error";

    if (error instanceof AxiosError) {
      errorMessage = error.message;
      if (error.code === "ECONNREFUSED") {
        errorMessage = `Connection to ${serviceName} refused. Service may be down.`;
      } else if (error.code === "ETIMEDOUT") {
        errorMessage = `Connection to ${serviceName} timed out. Service may be overloaded.`;
      }
    } else if (error instanceof Error) {
      errorMessage = error.message;
    }

    log(`[Proxy] Error connecting to ${targetUrl}: ${errorMessage}`);

    if (isConfigRequest(originalPath)) {
      log(
        `[Proxy] Providing fallback config data for failed ${serviceName} config request`
      );

      const fallbackConfig: FallbackConfig = {
        service: serviceName,
        config: {
          ENABLE_EXPLAINABILITY: false, 
          DEBUG_MODE: false,
          LOG_LEVEL: "info",
        },
        retrieval_timestamp: new Date().toISOString(),
        _note: "FALLBACK CONFIG - The actual service appears to be unreachable",
        _original_error: errorMessage,
      };

      const fallbackResponse: SuccessPayload = {
        success: true,
        data: fallbackConfig,
      };

      return res.status(200).json(fallbackResponse);
    }

    const errorResponse: ErrorPayload = {
      success: false,
      error: `Error connecting to ${serviceName}: ${errorMessage}`,
      details: { message: errorMessage },
      proxy_target: targetUrl,
      original_path: originalPath,
    };

    return res.status(503).json(errorResponse);
  }
}

// --- Route Handler for internal log endpoint ---
function handleInternalLog(req: Request, res: Response): void {
  try {
    const logEntry = req.body as Partial<LogMessage>;
    
    // Basic validation
    if (!logEntry || typeof logEntry !== 'object' || !logEntry.service || !logEntry.level || !logEntry.message) {
      log(`[INTERNAL LOG] Received invalid log: Missing required fields`);
      res.status(400).json({ success: false, error: 'Missing required log fields' });
      return;
    }
    
    const service = logEntry.service as LogMessage['service']; 
    if (!['memory-core', 'neural-memory', 'cce'].includes(service)) {
      log(`[INTERNAL LOG] Received invalid log: Invalid service '${service}'`);
      res.status(400).json({ success: false, error: `Invalid service name: ${service}` });
      return;
    }
    
    const level = logEntry.level as LogMessage['level']; 
    if (!['debug', 'info', 'warning', 'error'].includes(level)) {
      log(`[INTERNAL LOG] Received invalid log: Invalid level '${level}'`);
      res.status(400).json({ success: false, error: `Invalid log level: ${level}` });
      return;
    }
    
    // Generate ID and timestamp if missing (less critical, but good practice)
    const finalLogEntry: LogMessage = {
      id: logEntry.id || (Date.now().toString(36) + Math.random().toString(36).substr(2, 5)), // Ensure ID exists
      timestamp: logEntry.timestamp || new Date().toISOString(), // Ensure timestamp exists
      service: service,
      level: level,
      message: logEntry.message,
      context: logEntry.context || undefined // Use undefined to match schema expectation
    };
    log(`[INTERNAL LOG] Received valid log from service: ${finalLogEntry.service} (Level: ${finalLogEntry.level})`);
    storage.addLog(finalLogEntry);
    
    res.status(200).json({ success: true });
    return;
  
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    log(`[INTERNAL LOG] Error processing log: ${errorMessage}`);
    res.status(500).json({ success: false, error: 'Failed to process log' });
    return;
  }
}

export async function registerRoutes(app: express.Express): Promise<void> {
  log("Registering API proxy routes...");

  const router = express.Router(); 

  const memoryCoreUrl = MEMORY_CORE_URL;
  const neuralMemoryUrl = NEURAL_MEMORY_URL;
  const cceUrl = CCE_URL;

  router.get("/health", (_req: Request, res: Response) => { 
    res.json({ status: "healthy", service: "dashboard-proxy" });
  });

  router.post("/internal/log", handleInternalLog); 

  app.use(router);

  app.use("/api/memory-core", async (req: Request, res: Response, next: NextFunction) => {
    const relativePath = req.originalUrl.replace("/api/memory-core", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> MC Path: ${relativePath}`);
    await proxyRequest(req, res, memoryCoreUrl, "memory-core", relativePath);
  });

  app.use("/api/neural-memory", async (req: Request, res: Response, next: NextFunction) => {
    const relativePath = req.originalUrl.replace("/api/neural-memory", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> NM Path: ${relativePath}`);
    await proxyRequest(req, res, neuralMemoryUrl, "neural-memory", relativePath);
  });

  app.use("/api/cce", async (req: Request, res: Response, next: NextFunction) => {
    const relativePath = req.originalUrl.replace("/api/cce", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> CCE Path: ${relativePath}`);
    await proxyRequest(req, res, cceUrl, "cce", relativePath);
  });

  app.use("/api/proxy/memory-core", async (req: Request, res: Response, next: NextFunction) => {
    const relativePath = req.originalUrl.replace("/api/proxy/memory-core", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> MC Proxy Path: ${relativePath}`);
    await proxyRequest(req, res, memoryCoreUrl, "memory-core", relativePath);
  });

  app.use("/api/proxy/neural-memory", async (req: Request, res: Response, next: NextFunction) => {
    const relativePath = req.originalUrl.replace("/api/proxy/neural-memory", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> NM Proxy Path: ${relativePath}`);
    await proxyRequest(req, res, neuralMemoryUrl, "neural-memory", relativePath);
  });

  app.use("/api/proxy/cce", async (req: Request, res: Response, next: NextFunction) => {
    const relativePath = req.originalUrl.replace("/api/proxy/cce", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> CCE Proxy Path: ${relativePath}`);
    await proxyRequest(req, res, cceUrl, "cce", relativePath);
  });

  log("API proxy routes registration complete.");
}

import express, { Router, Request, Response } from "express";
import axios, { AxiosError, Method } from "axios";
import { storage } from "./storage"; // Assuming storage is correctly set up for alerts

// --- Environment & Constants ---

/**
 * These environment variables should point to the actual services
 * or default to localhost when not set.
 */
const MEMORY_CORE_URL = process.env.MEMORY_CORE_URL || "http://localhost:5020";
const NEURAL_MEMORY_URL = process.env.NEURAL_MEMORY_URL || "http://localhost:8001";
const CCE_URL = process.env.CCE_URL || "http://localhost:8002";

/**
 * Console logging helper for server-side logs.
 */
const log = (message: string): void => {
  console.log(`[Dashboard Server] ${message}`);
};

// --- Type Definitions ---

/**
 * Standard shape for successful API responses sent back from proxy.
 */
interface SuccessPayload<T = any> {
  success: true;
  data: T;
}

/**
 * Standard shape for error responses sent back from proxy.
 */
interface ErrorPayload {
  success: false;
  error: string;
  details: any;
  proxy_target: string;
  original_path: string;
}

/**
 * Shape of the fallback config response when a config-related request fails.
 */
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

/**
 * Union type for possible proxy responses (success or error).
 */
type ProxyResponse = SuccessPayload | ErrorPayload;

/**
 * A utility function to determine if the route is hitting a configuration endpoint,
 * in which case we provide fallback data on error.
 */
function isConfigRequest(path: string): boolean {
  return (
    /\/api\/(proxy\/)?(memory-core|neural-memory|cce)\/config\/runtime\//.test(path)
  );
}

// --- Proxy Function ---

/**
 * Proxies an incoming Express request to the target service URL, then returns the response.
 * Provides fallback configs for certain config endpoints if the proxy fails.
 *
 * @param req   Express Request object
 * @param res   Express Response object
 * @param targetUrlBase Base URL of the downstream service
 * @param serviceName   A short name used in logs and error messages
 * @param overridePath  An optional override for the path portion of the request
 */
async function proxyRequest(
  req: Request,
  res: Response,
  targetUrlBase: string,
  serviceName: string,
  overridePath?: string
): Promise<void> {
  const method = req.method as Method;
  const originalPath = req.originalUrl;

  // Determine the path to proxy; if none provided, derive from originalUrl
  let relativePath = overridePath ?? originalPath.replace(`/api/${serviceName}`, "");
  if (relativePath === `/${serviceName}` || relativePath === "") {
    relativePath = "/";
  } else if (!relativePath.startsWith("/")) {
    relativePath = "/" + relativePath;
  }

  // Ensure no trailing slash on base and append relative path
  const targetUrl = targetUrlBase.replace(/\/$/, "") + relativePath;

  // Logging
  log(`Proxying ${method} ${originalPath} (Target Path: ${relativePath}) to ${targetUrl}`);
  log(` -> Query Params: ${JSON.stringify(req.query)}`);
  if (method !== "GET" && method !== "HEAD") {
    log(` -> Request Body (keys): ${JSON.stringify(Object.keys(req.body || {}))}`);
  }

  try {
    const response = await axios({
      method,
      url: targetUrl,
      params: req.query,
      data: method !== "GET" && method !== "HEAD" ? req.body : undefined,
      headers: {
        "Content-Type": req.headers["content-type"] || "application/json",
        Accept: req.headers["accept"] || "application/json",
      },
      timeout: 20000,
    });

    log(`Proxy Success: ${method} ${targetUrl} returned status ${response.status}`);

    // Normalize the response payload to always have { success: true, data: ... }
    let responsePayload: any;
    if (
      typeof response.data === "object" &&
      response.data !== null &&
      "success" in response.data
    ) {
      responsePayload = response.data; // Assume it already has desired shape
    } else {
      responsePayload = { success: true, data: response.data };
    }

    log(` -> Sending SUCCESS payload (keys): ${JSON.stringify(Object.keys(responsePayload))}`);
    res.status(response.status).json(responsePayload);
  } catch (error: unknown) {
    const err = error as Error;
    log(`[!!!] Proxy Error for ${method} ${originalPath} -> ${targetUrl}: ${err.message}`);

    let status = 500;
    let errorMessage = `Unknown proxy error for ${serviceName}`;
    let errorDetails: any = null;

    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      if (axiosError.response) {
        status = axiosError.response.status;
        errorDetails = axiosError.response.data;
        errorMessage = `Backend service '${serviceName}' responded with status ${status}`;

        // Attempt to append any "detail" field in error response
        const detail =
          typeof errorDetails === "object" && errorDetails !== null && "detail" in errorDetails
            ? errorDetails.detail
            : null;
        if (detail) {
          errorMessage += `: ${detail}`;
        }
      } else if (axiosError.request) {
        status = 504;
        errorMessage = `No response received from backend service '${serviceName}' at ${targetUrlBase}. Is it running and accessible?`;
        errorDetails = { code: axiosError.code, message: err.message };
      } else {
        errorMessage = `Error setting up request to backend service '${serviceName}': ${err.message}`;
        errorDetails = { message: err.message };
      }
    } else {
      // Handle non-Axios errors (rare, but could happen)
      errorMessage = `Unexpected error during proxy for ${serviceName}: ${err.message}`;
      errorDetails = {
        message: err.message,
        stack_preview: err.stack?.substring(0, 200),
      };
    }

    // Fallback logic if this is a request for config endpoints
    if (isConfigRequest(originalPath) && status !== 404) { // Don't fallback on 404 (service might be valid but config not found)
      const pathParts = originalPath.split('/');
      const serviceParam = pathParts[pathParts.length - 1].split('?')[0]; // Get service name cleanly
      log(
        `[CONFIG_FALLBACK] Proxy error ${status} for config endpoint. Returning dev fallback for service: ${serviceParam}`
      );

      const fallbackResponse: SuccessPayload<FallbackConfig> = {
        success: true,
        data: {
          service: serviceParam || 'unknown',
          config: {
            // Provide essential defaults, especially ENABLE_EXPLAINABILITY
            ENABLE_EXPLAINABILITY: true, // Default to TRUE for dev/testing
            DEBUG_MODE: true,
            LOG_LEVEL: "INFO",
            // Add other minimal essential defaults if needed by FeaturesContext
          },
          retrieval_timestamp: new Date().toISOString(),
          _note: `FALLBACK CONFIG - Proxy Error ${status} accessing ${serviceName}: ${errorMessage.substring(0,100)}`,
          _original_error: errorMessage,
        },
      };

      // IMPORTANT: Always respond with 200 for fallback, then exit
      res.status(200).json(fallbackResponse);
      return; // Stop further error processing
    }

    // Default error response
    const errorPayload: ErrorPayload = {
      success: false,
      error: errorMessage,
      details: errorDetails,
      proxy_target: targetUrl,
      original_path: originalPath,
    };

    log(
      ` -> Sending ERROR payload for ${originalPath}: ${JSON.stringify(errorPayload).substring(
        0,
        200
      )}...`
    );
    res.status(status > 0 ? status : 500).json(errorPayload);
  }
}

// --- Route Registration ---

/**
 * Registers all routes for the Dashboard server.
 * - /api routes (memory-core, neural-memory, cce, etc.)
 * - /api/proxy routes (mirrors above)
 * - /health for the top-level health check
 */
export async function registerRoutes(app: express.Express): Promise<void> {
  log("Registering API proxy routes...");

  // Reuse environment URLs
  const memoryCoreUrl = MEMORY_CORE_URL;
  const neuralMemoryUrl = NEURAL_MEMORY_URL;
  const cceUrl = CCE_URL;

  // --- Top-level health check route ---
  app.get("/health", (_req: Request, res: Response) => {
    res.json({ status: "healthy", service: "dashboard-proxy" });
  });

  // --- Alerts Route (Mocked) ---
  app.get("/api/alerts", async (_req: Request, res: Response) => {
    log("Handling /api/alerts request using mock storage.");
    try {
      const alerts = await storage.getAlerts();
      res.json({ success: true, data: alerts });
    } catch (err: any) {
      log(`Error fetching mock alerts: ${err.message}`);
      res.status(500).json({ success: false, error: "Failed to fetch mock alerts" });
    }
  });

  // --- API Middleware for Memory Core Routes ---
  app.use("/api/memory-core", async (req: Request, res: Response) => {
    const relativePath = req.originalUrl.replace("/api/memory-core", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> MC Path: ${relativePath}`);
    await proxyRequest(req, res, memoryCoreUrl, "memory-core", relativePath);
  });

  // --- API Route for Memory Core Alerts ---
  app.get("/api/memory-core/alerts", async (req: Request, res: Response) => {
    log(`[Direct Hit] GET /api/memory-core/alerts`);
    await proxyRequest(req, res, memoryCoreUrl, "memory-core", "/alerts");
  });

  // --- API Middleware for Neural Memory Routes ---
  app.use("/api/neural-memory", async (req: Request, res: Response) => {
    const relativePath = req.originalUrl.replace("/api/neural-memory", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> NM Path: ${relativePath}`);
    await proxyRequest(req, res, neuralMemoryUrl, "neural-memory", relativePath);
  });

  // --- API Middleware for CCE Routes ---
  app.use("/api/cce", async (req: Request, res: Response) => {
    const relativePath = req.originalUrl.replace("/api/cce", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> CCE Path: ${relativePath}`);
    await proxyRequest(req, res, cceUrl, "cce", relativePath);
  });

  // --- API/Proxy Middleware for Memory Core Routes ---
  app.use("/api/proxy/memory-core", async (req: Request, res: Response) => {
    const relativePath = req.originalUrl.replace("/api/proxy/memory-core", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> MC Proxy Path: ${relativePath}`);
    await proxyRequest(req, res, memoryCoreUrl, "memory-core", relativePath);
  });

  // --- API/Proxy Middleware for Neural Memory Routes ---
  app.use("/api/proxy/neural-memory", async (req: Request, res: Response) => {
    const relativePath = req.originalUrl.replace("/api/proxy/neural-memory", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> NM Proxy Path: ${relativePath}`);
    await proxyRequest(req, res, neuralMemoryUrl, "neural-memory", relativePath);
  });

  // --- API/Proxy Middleware for CCE Routes ---
  app.use("/api/proxy/cce", async (req: Request, res: Response) => {
    const relativePath = req.originalUrl.replace("/api/proxy/cce", "") || "/";
    log(`[Middleware Hit] ${req.method} ${req.originalUrl} -> CCE Proxy Path: ${relativePath}`);
    await proxyRequest(req, res, cceUrl, "cce", relativePath);
  });

  log("API proxy routes registration complete.");
}

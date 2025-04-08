import express, { type Request, type Response, type NextFunction } from "express";
import { join } from "path";
import { createServer } from "http";
import { WebSocketServer, WebSocket as WS, Data as WSData } from "ws";
import fs from 'fs'; 
import readline from 'readline'; 
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import { storage, type LogMessage } from './storage'; 

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// --- Keep existing middleware ---
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});
// --- End of existing middleware ---


// Add a global error handler to prevent Node.js process crashes
process.on('uncaughtException', (error) => {
  log(`[!!!] CRITICAL: Uncaught Exception! ${error.message}`);
  log(`[!!!] Stack trace: ${error.stack}`);
  // Don't exit the process, just log the error
  // process.exit(1); 
});

process.on('unhandledRejection', (reason, promise) => {
  log(`[!!!] CRITICAL: Unhandled Promise Rejection at: ${promise}, reason: ${reason}`);
  // Don't exit the process, just log the error
});

(async () => {
  // Create HTTP server from Express app
  const httpServer = createServer(app);

  // Register API routes (pass the app, not the server)
  await registerRoutes(app);

  // Setup WebSocket Server
  const wss = new WebSocketServer({ server: httpServer, path: '/logs' });

  // Track connected clients
  const connectedClients = new Set<WS>();

  // Function to broadcast a log to all connected clients
  const broadcastLog = (logEntry: LogMessage): void => {
    const message = JSON.stringify(logEntry);
    connectedClients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  };

  // Function to broadcast all logs to a specific client
  const sendLogsToClient = (client: WS): void => {
    const logs = storage.getLogs();
    if (logs.length === 0) return;
    
    client.send(JSON.stringify({ type: 'history', logs }));
  };

  // WebSocket connection handling
  wss.on('connection', (ws: WS, req: any) => {
    connectedClients.add(ws);
    log(`[WebSocket] Client connected. Total clients: ${connectedClients.size}`);

    // Send existing logs on connection if requested
    ws.on('message', (data: WSData) => {
      try {
        const message = JSON.parse(data.toString());
        if (message.type === 'getLogs') {
          log(`[WebSocket] Client requested logs`);
          sendLogsToClient(ws);
        } else if (message.type === 'clearLogs') {
          log(`[WebSocket] Client requested to clear logs`);
          storage.clearLogs();
          // Notify all clients that logs were cleared
          connectedClients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({ type: 'logsCleared' }));
            }
          });
        }
      } catch (err) {
        log(`[WebSocket] Error processing message: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    });

    // Handle client disconnection
    ws.on('close', () => {
      connectedClients.delete(ws);
      log(`[WebSocket] Client disconnected. Total clients: ${connectedClients.size}`);
    });

    // Handle errors
    ws.on('error', (err) => {
      log(`[WebSocket] Client error: ${err.message}`);
      connectedClients.delete(ws);
    });
  });

  // Periodically check for new logs and broadcast them
  setInterval(() => {
    if (storage.hasNewLogs()) {
      const logs = storage.getLogs();
      // Get the newest log (index 0 if sorted descending by timestamp)
      const newestLog = logs[0];
      if (newestLog) {
        log(`[WebSocket] Broadcasting new log: ${newestLog.id} from ${newestLog.service}`);
        broadcastLog(newestLog);
      }
      storage.markLogsRead();
    }
  }, 500); // Check every 500ms

  log('[WebSocket] Server initialized on path /logs');

  // --- Global Error Handler - MODIFIED ---
  app.use((err: any, req: Request, res: Response, next: NextFunction) => {
    // Check if headers have already been sent
    if (res.headersSent) {
      log(`[ERROR HANDLER] Headers already sent for ${req.originalUrl}, cannot send error response.`);
      // If headers sent, delegate to default Express error handler
      return next(err);
    }

    // Determine status and message
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    // Log the error *before* sending response
    log(`[GLOBAL ERROR] ${status} - ${req.method} ${req.originalUrl} - ${message}`);
    if (err.stack) {
      log(`[GLOBAL ERROR STACK] ${err.stack}`);
    }

    // Send the JSON error response ONLY IF headers not sent
    res.status(status).json({
      success: false, // Align with our standard error format
      error: "Internal Server Error", // Generic message
      details: message // Put original message in details
    });
  });
  // --- End Global Error Handler ---


  if (app.get("env") === "development") {
    await setupVite(app, httpServer);
  } else {
    serveStatic(app);
  }

  // ALWAYS serve the app on port from environment variables, with a fallback to 5000
  const port = process.env.PORT || 5000;
  httpServer.listen(port, () => {
    log(`serving on port ${port}`);
  });
})();

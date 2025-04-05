// server.mjs - Modern ESM entry point for the Synthians Cognitive Dashboard
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import express from 'express';
import { createServer } from 'http';
import { createServer as createViteServer } from 'vite';

// Set environment for development
process.env.NODE_ENV = 'development';

// Setup paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Create Express app
const app = express();

// Body parser middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Create routes directly instead of importing from routes.ts
async function setupSimpleRoutes() {
  // Create a simple router and server for testing
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Simple server running' });
  });
  
  app.get('/api/test', (req, res) => {
    res.json({ message: 'API test endpoint working' });
  });

  // Simplified memory core routes
  app.get('/api/memory-core/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  app.get('/api/memory-core/stats', (req, res) => {
    // Return mock stats data based on Phase 5.8 memory assembly stats requirements
    res.json({
      assemblies: {
        count: 24,
        average_size: 8,
        activation_count: 156,
        pending_updates: 0
      },
      memories: {
        count: 512,
        by_type: {
          declarative: 320,
          procedural: 125,
          episodic: 67
        }
      },
      system: {
        uptime: "3h 22m",
        version: "0.9.5-beta"
      }
    });
  });

  // Phase 5.9 explainability endpoints
  app.get('/api/memory-core/assemblies/:id/explain_activation', (req, res) => {
    res.json({
      assembly_id: req.params.id,
      explanation: "This assembly was activated because Memory #M-12345 matched the input query with similarity score 0.87 (threshold: 0.75).",
      activation_details: {
        memory_id: "M-12345",
        similarity_score: 0.87,
        threshold: 0.75,
        activated_at: "2025-04-05T11:58:22Z"
      }
    });
  });

  app.get('/api/memory-core/assemblies/:id/explain_merge', (req, res) => {
    res.json({
      assembly_id: req.params.id,
      explanation: "This assembly was formed by merging 3 source assemblies based on semantic similarity.",
      merge_details: {
        source_assemblies: ["ASM-001", "ASM-002", "ASM-005"],
        similarity_threshold: 0.82,
        merge_time: "2025-04-05T10:12:45Z",
        cleanup_status: "completed"
      }
    });
  });

  app.get('/api/memory-core/assemblies/:id/lineage', (req, res) => {
    res.json({
      assembly_id: req.params.id,
      lineage: [
        {
          level: 0,
          assembly_id: req.params.id,
          created_at: "2025-04-05T10:12:45Z",
          merge_source: "direct_merge"
        },
        {
          level: 1,
          assembly_id: "ASM-001",
          created_at: "2025-04-05T09:35:12Z",
          merge_source: "direct_creation"
        },
        {
          level: 1,
          assembly_id: "ASM-002",
          created_at: "2025-04-05T08:22:31Z",
          merge_source: "direct_creation"
        },
        {
          level: 1,
          assembly_id: "ASM-005",
          created_at: "2025-04-05T07:45:19Z",
          merge_source: "previous_merge"
        }
      ]
    });
  });

  app.get('/api/diagnostics/merge_log', (req, res) => {
    res.json({
      entries: [
        {
          merge_event_id: "merge-123",
          timestamp: "2025-04-05T11:12:45Z",
          source_assembly_ids: ["ASM-007", "ASM-009"],
          result_assembly_id: "ASM-012",
          similarity_score: 0.85,
          threshold_used: 0.8,
          cleanup_status: "completed"
        },
        {
          merge_event_id: "merge-122",
          timestamp: "2025-04-05T10:55:32Z",
          source_assembly_ids: ["ASM-003", "ASM-004"],
          result_assembly_id: "ASM-011",
          similarity_score: 0.91,
          threshold_used: 0.8,
          cleanup_status: "completed"
        },
        {
          merge_event_id: "merge-121",
          timestamp: "2025-04-05T10:22:18Z",
          source_assembly_ids: ["ASM-001", "ASM-002", "ASM-005"],
          result_assembly_id: "ASM-010",
          similarity_score: 0.83,
          threshold_used: 0.8,
          cleanup_status: "failed",
          error: "Timeout during vector index update"
        }
      ]
    });
  });

  app.get('/api/neural-memory/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  app.get('/api/cce/health', (req, res) => {
    res.json({ status: 'ok' });
  });
  
  const server = createServer(app);
  return server;
}

// Setup Vite for the frontend
async function setupVite(server) {
  try {
    // Create Vite server with proper path resolution for @ alias
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: { server },
      },
      // Use root directory to match our vite.config.ts
      root: resolve(__dirname, 'client'),
      // Configure path aliases - must match vite.config.ts
      resolve: {
        alias: {
          '@': resolve(__dirname, 'client/src'),
          '@shared': resolve(__dirname, 'shared'),
          '@assets': resolve(__dirname, 'attached_assets')
        }
      },
      // When using Windows paths, ensure proper path resolution
      appType: 'spa',
      optimizeDeps: {
        include: [
          'react',
          'react-dom',
          '@radix-ui/react-toast',
          'class-variance-authority',
          'clsx',
          'tailwind-merge'
        ]
      }
    });

    // Use Vite's connect instance as middleware
    app.use(vite.middlewares);

    // Handle all non-API routes with Vite
    app.use('*', async (req, res, next) => {
      // Skip API routes
      if (req.originalUrl.startsWith('/api')) {
        return next();
      }

      try {
        // Serve index.html through Vite's transform for all non-API routes
        const url = req.originalUrl;
        const indexPath = resolve(__dirname, 'client', 'index.html');

        // Transform the index.html with proper React imports
        let template = await vite.transformIndexHtml(url, `
          <!DOCTYPE html>
          <html lang="en">
            <head>
              <meta charset="UTF-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1" />
              <link rel="icon" type="image/ico" href="/favicon.ico" />
              <title>Synthians Cognitive Dashboard</title>
            </head>
            <body>
              <div id="root"></div>
              <script type="module" src="/src/main.tsx"></script>
            </body>
          </html>
        `);
        
        res.status(200).set({ 'Content-Type': 'text/html' }).end(template);
      } catch (error) {
        console.error('Error serving frontend:', error);
        vite.ssrFixStacktrace(error);
        res.status(500).send('Internal Server Error');
      }
    });

    console.log('Vite middleware configured successfully');
  } catch (error) {
    console.error('Failed to initialize Vite middleware:', error);
  }
}

// Start server
async function startServer() {
  console.log('Starting Synthians Cognitive Dashboard server...');
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  
  try {
    // Create simplified routes for testing
    const server = await setupSimpleRoutes();
    
    // Setup Vite for frontend
    await setupVite(server);
    
    // Start server - use a different port (5500) to avoid conflicts
    const PORT = process.env.PORT || 5500;
    server.listen(PORT, () => {
      console.log(`Server running on port ${PORT} in ${process.env.NODE_ENV} mode`);
      console.log(`Dashboard available at http://localhost:${PORT}`);
      console.log(`Test API at http://localhost:${PORT}/api/health`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();

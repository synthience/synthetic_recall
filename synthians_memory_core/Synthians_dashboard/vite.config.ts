import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import themePlugin from "@replit/vite-plugin-shadcn-theme-json";
import path from "path";
import { fileURLToPath } from "url";
import runtimeErrorOverlay from "@replit/vite-plugin-runtime-error-modal";

// Correctly define __dirname for ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  plugins: [
    // Use React plugin without extra JSX options - let it handle automatic JSX runtime
    react(),
    runtimeErrorOverlay(),
    themePlugin(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "client", "src"),
      "@shared": path.resolve(__dirname, "shared"),
      "@assets": path.resolve(__dirname, "attached_assets"),
    },
  },
  root: path.resolve(__dirname, "client"),
  build: {
    outDir: path.resolve(__dirname, "dist", "public"),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      // Proxy /api/memory-core requests to Memory Core service
      // Assuming 'synthians_core' is the service name in docker-compose.yml
      '/api/memory-core': {
        target: 'http://synthians_core:5010',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api\/memory-core/, ''),
      },
      // Proxy /api/neural-memory requests to Neural Memory service
      // Assuming 'trainer-server' is the service name in docker-compose.yml
      '/api/neural-memory': {
        target: 'http://trainer-server:8001',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api\/neural-memory/, ''),
      },
      // Proxy /api/cce requests to Context Cascade Engine service
      // Assuming 'context-cascade-orchestrator' is the service name in docker-compose.yml
      '/api/cce': {
        target: 'http://context-cascade-orchestrator:8002',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api\/cce/, ''),
      },
      // Optional: Proxy for alerts if handled by a separate backend part
      // '/api/alerts': {
      //   target: 'http://localhost:YOUR_ALERT_PORT', // Adjust if needed
      //   changeOrigin: true,
      //   secure: false,
      //   rewrite: (path) => path.replace(/^\/api\/alerts/, ''),
      // },
    },
    hmr: false,
  },
});

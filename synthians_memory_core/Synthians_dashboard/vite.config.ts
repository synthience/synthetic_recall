import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import themePlugin from "@replit/vite-plugin-shadcn-theme-json";
import path from "path";
import runtimeErrorOverlay from "@replit/vite-plugin-runtime-error-modal";

// Use path.resolve directly based on __dirname
const __dirname = path.resolve();

export default defineConfig({
  plugins: [
    // Use React plugin without extra JSX options - let it handle automatic JSX runtime
    react(),
    runtimeErrorOverlay(),
    themePlugin(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "client", "src"),
      "@shared": path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "shared"),
      "@assets": path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "attached_assets"),
    },
  },
  root: path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "client"),
  build: {
    outDir: path.resolve(__dirname, "synthians_memory_core", "Synthians_dashboard", "dist", "public"),
    emptyOutDir: true,
  },
});

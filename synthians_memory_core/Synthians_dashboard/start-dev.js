#!/usr/bin/env node

import { spawn } from 'child_process';
import { createServer } from 'http';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Start the server
const serverProcess = spawn('node', ['--import=tsx', './server/index.ts'], {
  stdio: 'inherit',
  cwd: __dirname,
  env: { ...process.env, NODE_ENV: 'development' }
});

console.log('Starting Synthians Cognitive Dashboard development server...');

serverProcess.on('close', (code) => {
  console.log(`Server process exited with code ${code}`);
  process.exit(code);
});

process.on('SIGINT', () => {
  console.log('Shutting down development server...');
  serverProcess.kill('SIGINT');
});

process.on('SIGTERM', () => {
  console.log('Shutting down development server...');
  serverProcess.kill('SIGTERM');
});

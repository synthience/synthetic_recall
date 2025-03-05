// src/lib/memoryService.ts

type MemoryMetric = {
    id: string;
    text: string;
    similarity: number;
    significance: number;
    surprise: number;
    timestamp: number;
  };
  
  type MemoryStats = {
    memory_count: number;
    gpu_memory: number;
    active_connections: number;
  };
  
  type MemoryCallback = (data: any) => void;
  
  class MemoryService {
    private tensorServer: WebSocket | null = null;
    private hpcServer: WebSocket | null = null;
    private status: string = 'Disconnected';
    private hpcStatus: string = 'Disconnected';
    private memoryCount: number = 0;
    private callbacks: Map<string, MemoryCallback[]> = new Map();
    private memoryCache: Map<string, any> = new Map();
    private selectedMemories: Set<string> = new Set();
    private reconnectInterval: NodeJS.Timeout | null = null;
    private retryCount: number = 0;
    private maxRetries: number = 5;
    private enabled: boolean = false;
  
    constructor(
      private tensorUrl: string = 'ws://localhost:5001',
      private hpcUrl: string = 'ws://localhost:5005'
    ) {}
  
    /**
     * Initialize the memory service connections
     */
    initialize(tensorUrl?: string, hpcUrl?: string): Promise<boolean> {
      // Update URLs if provided
      if (tensorUrl) this.tensorUrl = tensorUrl;
      if (hpcUrl) this.hpcUrl = hpcUrl;
      
      return new Promise((resolve) => {
        // Log the URLs being used
        console.log(`Initializing memory service with tensorUrl: ${this.tensorUrl}, hpcUrl: ${this.hpcUrl}`);
        
        this.connectTensorServer()
          .then(() => {
            this.connectHPCServer();
            resolve(true);
          })
          .catch(() => {
            resolve(false);
          });
      });
    }
  
    /**
     * Set whether the memory system is enabled
     */
    setEnabled(enabled: boolean): void {
      console.log(`Memory service enabled state changing from ${this.enabled} to ${enabled}`);
      if (this.enabled === enabled) {
        console.log('Memory service enabled state unchanged, skipping update');
        return;
      }
      
      this.enabled = enabled;
      this.emit('enabled_changed', { enabled });
      console.log(`Memory service enabled state set to: ${enabled}`);
      
      // When enabling, try to connect if not already connected
      if (enabled && this.status !== 'Connected') {
        console.log('Attempting to connect memory service after enabling');
        this.initialize();
      }
    }
  
    /**
     * Get whether the memory system is enabled
     */
    isEnabled(): boolean {
      return this.enabled;
    }
  
    /**
     * Update selection
     */
    updateSelection(selectedIds: string[]): void {
      this.selectedMemories = new Set(selectedIds);
      this.emit('selection_changed', { selectedMemories: Array.from(this.selectedMemories) });
    }
  
    /**
     * Search memory
     */
    search(query: string, limit: number = 5): boolean {
      console.log(`Searching for: "${query}" with limit ${limit}`);
      // Mock search results for now
      setTimeout(() => {
        if (!this.enabled) {
          console.log("Memory search skipped - memory system disabled");
          return;
        }
        
        const results = [
          { id: '1', text: 'Sample memory result 1 matching ' + query, significance: 0.85, surprise: 0.75 },
          { id: '2', text: 'Sample memory result 2 matching ' + query, significance: 0.65, surprise: 0.45 },
          { id: '3', text: 'Sample memory result 3 matching ' + query, significance: 0.55, surprise: 0.35 }
        ];
        
        this.emit('search_results', { results });
      }, 300);
      
      return true;
    }
  
    /**
     * Search memory (legacy alias)
     */
    searchMemory(query: string, limit: number = 5): boolean {
      return this.search(query, limit);
    }
  
    /**
     * Toggle memory selection
     */
    toggleMemorySelection(id: string): void {
      const newSelection = new Set(this.selectedMemories);
      if (newSelection.has(id)) {
        newSelection.delete(id);
      } else {
        newSelection.add(id);
      }
      this.updateSelection(Array.from(newSelection));
    }
  
    /**
     * Clear selected memories
     */
    clearSelectedMemories(): void {
      this.updateSelection([]);
    }
  
    /**
     * Connect to the tensor server
     */
    private connectTensorServer(): Promise<void> {
      return new Promise((resolve, reject) => {
        try {
          console.log(`Connecting to tensor server at ${this.tensorUrl}`);
          this.tensorServer = new WebSocket(this.tensorUrl);
  
          this.tensorServer.onopen = () => {
            console.log("Tensor Server Connected");
            this.status = 'Connected';
            this.retryCount = 0;
            this.emit('status', { status: this.status });
            
            // Request initial stats
            this.sendToTensorServer({
              type: 'get_stats'
            });
            
            resolve();
          };
  
          this.tensorServer.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              console.log("Tensor Server received:", data);
              
              if (data.type === 'embeddings') {
                this.processEmbedding(data.embeddings);
              } else if (data.type === 'search_results') {
                this.emit('search_results', data);
              } else if (data.type === 'stats') {
                this.updateMemoryStats(data);
              }
            } catch (error) {
              console.error('Error parsing tensor server message:', error);
            }
          };
  
          this.tensorServer.onclose = () => {
            console.log("Tensor Server Disconnected");
            this.status = 'Disconnected';
            this.emit('status', { status: this.status });
            this.tensorServer = null;
            
            this.attemptReconnect();
          };
  
          this.tensorServer.onerror = (error) => {
            console.error('Tensor Server error:', error);
            this.status = 'Error';
            this.emit('status', { status: this.status });
            reject(error);
          };
        } catch (error) {
          console.error('Failed to connect to tensor server:', error);
          this.status = 'Error';
          this.emit('status', { status: this.status });
          reject(error);
        }
      });
    }
  
    /**
     * Connect to the HPC server
     */
    private connectHPCServer(): void {
      try {
        console.log(`Connecting to HPC server at ${this.hpcUrl}`);
        this.hpcServer = new WebSocket(this.hpcUrl);
  
        this.hpcServer.onopen = () => {
          console.log("HPC Server Connected");
          this.hpcStatus = 'Connected';
          this.emit('hpc_status', { status: 'Connected' });
        };
  
        this.hpcServer.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log("HPC Server received:", data);
            
            if (data.type === 'processed') {
              this.memoryCount++;
              this.emit('memory_count', { count: this.memoryCount });
  
              // If we received significance data, store it
              if (data.significance !== undefined) {
                this.emit('memory_processed', { 
                  significance: data.significance,
                  surprise: data.surprise || Math.random() * 0.5 // Fallback if not provided
                });
              }
            }
          } catch (error) {
            console.error('Error parsing HPC server message:', error);
          }
        };
  
        this.hpcServer.onclose = () => {
          console.log("HPC Server Disconnected");
          this.hpcStatus = 'Disconnected';
          this.emit('hpc_status', { status: 'Disconnected' });
          this.hpcServer = null;
        };
  
        this.hpcServer.onerror = (error) => {
          console.error('HPC Server error:', error);
          this.hpcStatus = 'Error';
          this.emit('hpc_status', { status: 'Error' });
        };
      } catch (error) {
        console.error('Failed to connect to HPC server:', error);
        this.hpcStatus = 'Error';
        this.emit('hpc_status', { status: 'Error' });
      }
    }
  
    /**
     * Process embedding data
     */
    private async processEmbedding(embeddings: number[]): Promise<void> {
      try {
        // Send to HPC server
        if (this.hpcServer?.readyState === WebSocket.OPEN) {
          const hpcRequest = {
            type: 'process',
            embeddings: embeddings
          };
          this.hpcServer.send(JSON.stringify(hpcRequest));
        }
        
        // Cache locally with timestamp
        const timestamp = Date.now();
        this.memoryCache.set(timestamp.toString(), embeddings);
        
        // Update status
        this.status = 'Processing';
        this.emit('status', { status: this.status });
      } catch (error) {
        console.error('Error processing embedding:', error);
      }
    }
  
    /**
     * Update memory statistics
     */
    private updateMemoryStats(stats: MemoryStats): void {
      if (stats.memory_count !== undefined) {
        this.memoryCount = stats.memory_count;
      }
      
      this.emit('stats', stats);
    }
  
    /**
     * Send data to tensor server
     */
    sendToTensorServer(data: any): boolean {
      if (this.tensorServer?.readyState === WebSocket.OPEN) {
        this.tensorServer.send(JSON.stringify(data));
        return true;
      }
      return false;
    }
  
    /**
     * Process user input
     */
    processInput(text: string): Promise<boolean> {
      return new Promise((resolve) => {
        if (!text.trim() || !this.enabled) {
          resolve(false);
          return;
        }
  
        // Store in memory (on Tensor server)
        const storeSuccess = this.sendToTensorServer({
          type: 'embed',
          text: text
        });
  
        // Search memory
        const searchSuccess = this.sendToTensorServer({
          type: 'search',
          text: text,
          limit: 5
        });
  
        resolve(storeSuccess && searchSuccess);
      });
    }
  
    /**
     * Get selected memories
     */
    getSelectedMemories(): string[] {
      return Array.from(this.selectedMemories);
    }
  
    /**
     * Register callback for events
     */
    on(event: string, callback: MemoryCallback): void {
      if (!this.callbacks.has(event)) {
        this.callbacks.set(event, []);
      }
      
      this.callbacks.get(event)?.push(callback);
    }
  
    /**
     * Unregister callback
     */
    off(event: string, callback: MemoryCallback): void {
      const callbacks = this.callbacks.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index !== -1) {
          callbacks.splice(index, 1);
        }
      }
    }
  
    /**
     * Emit event
     */
    private emit(event: string, data: any): void {
      const callbacks = this.callbacks.get(event);
      if (callbacks) {
        callbacks.forEach(callback => callback(data));
      }
    }
  
    /**
     * Attempt to reconnect to servers
     */
    private attemptReconnect(): void {
      if (this.reconnectInterval) {
        clearInterval(this.reconnectInterval);
      }
      
      this.reconnectInterval = setInterval(() => {
        this.retryCount++;
        console.log(`Attempting to reconnect... (${this.retryCount}/${this.maxRetries})`);
        
        if (this.retryCount > this.maxRetries) {
          if (this.reconnectInterval) {
            clearInterval(this.reconnectInterval);
            this.reconnectInterval = null;
          }
          console.log('Max retry attempts reached');
          return;
        }
        
        if (!this.tensorServer) {
          this.connectTensorServer()
            .then(() => {
              if (this.reconnectInterval) {
                clearInterval(this.reconnectInterval);
                this.reconnectInterval = null;
              }
            })
            .catch(() => {});
        }
        
        if (!this.hpcServer) {
          this.connectHPCServer();
        }
      }, 5000);
    }
  
    /**
     * Disconnect from servers
     */
    disconnect(): void {
      if (this.tensorServer) {
        this.tensorServer.close();
        this.tensorServer = null;
      }
      
      if (this.hpcServer) {
        this.hpcServer.close();
        this.hpcServer = null;
      }
      
      if (this.reconnectInterval) {
        clearInterval(this.reconnectInterval);
        this.reconnectInterval = null;
      }
      
      this.status = 'Disconnected';
      this.hpcStatus = 'Disconnected';
      this.emit('status', { status: this.status });
      this.emit('hpc_status', { status: this.hpcStatus });
    }
  
    /**
     * Get connection status
     */
    getStatus(): string {
      return this.status;
    }
  
    /**
     * Get HPC status
     */
    getHPCStatus(): string {
      return this.hpcStatus;
    }
  }
  
  // Singleton instance
  let memoryServiceInstance: MemoryService | null = null;
  
  /**
   * Get memory service instance
   */
  export const getMemoryService = (
    tensorUrl?: string,
    hpcUrl?: string
  ): MemoryService => {
    if (!memoryServiceInstance) {
      memoryServiceInstance = new MemoryService(tensorUrl, hpcUrl);
    } else if (tensorUrl || hpcUrl) {
      // Update the URLs in the existing instance if provided
      memoryServiceInstance.initialize(tensorUrl, hpcUrl);
    }
    return memoryServiceInstance;
  };
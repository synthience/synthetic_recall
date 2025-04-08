import { toast } from "@/hooks/use-toast";

/**
 * WebSocketManager
 * 
 * Phase 5.9.3 WebSocket implementation for real-time logging
 * Handles connection, reconnection, and message processing for log streams
 */
export class WebSocketManager {
  private socket: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1s delay
  private messageHandlers: Array<(data: any) => void> = [];
  private connectionStatusHandlers: Array<(status: ConnectionStatus) => void> = [];
  private connectionStatus: ConnectionStatus = 'disconnected';

  constructor(url: string) {
    this.url = url;
  }

  /**
   * Connect to the WebSocket server
   * @returns Promise that resolves when connected or rejects on failure
   */
  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.setConnectionStatus('connecting');
        
        // Close existing socket if any
        if (this.socket) {
          this.socket.close();
          this.socket = null;
        }
        
        this.socket = new WebSocket(this.url);
        
        // Setup event handlers
        this.socket.onopen = () => {
          this.onConnected();
          resolve();
        };
        
        this.socket.onclose = this.onDisconnected.bind(this);
        this.socket.onerror = (error) => {
          console.warn(`WebSocket connection to ${this.url} failed, will retry automatically`);
          this.onError(error);
          reject(error);
        };
        
        this.socket.onmessage = this.onMessage.bind(this);
      } catch (error) {
        this.setConnectionStatus('error');
        console.error('WebSocket connection error:', error);
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  public disconnect(): void {
    if (this.socket) {
      // Prevent reconnection attempts during intentional disconnect
      this.reconnectAttempts = this.maxReconnectAttempts;
      this.socket.close();
      this.socket = null;
      this.setConnectionStatus('disconnected');
    }
  }

  /**
   * Register a handler for incoming messages
   * @param handler Function to call with received message data
   * @returns Function to unregister the handler
   */
  public addMessageHandler(handler: (data: any) => void): () => void {
    this.messageHandlers.push(handler);
    return () => {
      this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
    };
  }

  /**
   * Register a handler for connection status changes
   * @param handler Function to call with new connection status
   * @returns Function to unregister the handler
   */
  public onConnectionStatus(handler: (status: ConnectionStatus) => void): () => void {
    this.connectionStatusHandlers.push(handler);
    // Immediately notify with current status
    handler(this.connectionStatus);
    return () => {
      this.connectionStatusHandlers = this.connectionStatusHandlers.filter(h => h !== handler);
    };
  }

  /**
   * Get the current connection status
   */
  public getStatus(): ConnectionStatus {
    return this.connectionStatus;
  }

  /**
   * Send data to the WebSocket server
   * @param data Data to send
   * @returns Promise that resolves on success or rejects on failure
   */
  public send(data: any): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket is not connected'));
        return;
      }

      try {
        this.socket.send(typeof data === 'string' ? data : JSON.stringify(data));
        resolve();
      } catch (error) {
        reject(error);
      }
    });
  }

  // Private methods
  private setConnectionStatus(status: ConnectionStatus): void {
    this.connectionStatus = status;
    this.connectionStatusHandlers.forEach(handler => handler(status));
  }

  private onConnected(): void {
    this.reconnectAttempts = 0;
    this.reconnectDelay = 1000;
    this.setConnectionStatus('connected');
    
    // Notify user of successful connection
    toast({
      title: 'Log Stream Connected',
      description: 'Receiving real-time log updates',
    });
  }

  private onDisconnected(event: CloseEvent): void {
    this.setConnectionStatus('disconnected');
    this.socket = null;

    // Don't attempt to reconnect if we've exceeded max attempts or disconnect was clean
    if (this.reconnectAttempts >= this.maxReconnectAttempts || event.wasClean) {
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.warn(`WebSocket reconnection failed after ${this.maxReconnectAttempts} attempts`);
        toast({
          title: 'Log Stream Connection Failed',
          description: 'Unable to connect to log stream after multiple attempts. Check server availability.',
          variant: 'destructive',
        });
      }
      return;
    }

    // Implement exponential backoff for reconnection
    const delay = Math.min(30000, this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts));
    this.reconnectAttempts++;

    console.log(`WebSocket reconnecting in ${delay/1000}s (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    // Only show toast on first reconnect attempt to avoid spamming
    if (this.reconnectAttempts === 1) {
      toast({
        title: 'Log Stream Disconnected',
        description: `Attempting to reconnect...`,
        variant: 'default',
      });
    }

    setTimeout(() => this.connect().catch(() => {
      // Silent catch to prevent unhandled promise rejection
      // The reconnection logic will handle retries
    }), delay);
  }

  private onError(error: Event): void {
    // Only show error in console, don't spam UI with errors
    console.error('WebSocket error:', error);
    this.setConnectionStatus('error');
    
    // The connection will automatically be retried by onDisconnected
  }

  private onMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);
      this.messageHandlers.forEach(handler => handler(data));
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }
}

/**
 * Connection status for WebSocket
 */
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

/**
 * Log level enumeration
 */
export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
}

/**
 * Log message structure from the server
 */
export interface LogMessage {
  id: string;
  timestamp: string;
  service: 'memory-core' | 'neural-memory' | 'cce';
  level: LogLevel;
  message: string;
  context?: Record<string, any>;
}

/**
 * Create and configure the WebSocket manager singleton
 */
let wsManagerInstance: WebSocketManager | null = null;

export const getWebSocketManager = (): WebSocketManager => {
  if (!wsManagerInstance) {
    // Use environment variable or default to localhost
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:5000/logs'; // Updated default
    wsManagerInstance = new WebSocketManager(wsUrl);
  }
  return wsManagerInstance;
};

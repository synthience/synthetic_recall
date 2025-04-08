import { users, type User, type InsertUser, type Alert } from "@shared/schema";

// Define Log type to match what we expect from backend services
export interface LogMessage {
  id: string;
  timestamp: string; // ISO string
  service: 'memory-core' | 'neural-memory' | 'cce' | 'dashboard-proxy' | string; // Allow other sources
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  context?: Record<string, any>;
}

// Extend the storage interface with additional methods
export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  getAlerts(): Promise<Alert[]>;
  getLogs(): LogMessage[]; // Method to retrieve logs
  addLog(log: LogMessage): void; // Method to add a log
  clearLogs(): void; // Method to clear logs
  hasNewLogs(): boolean; // Method to check for new logs
  markLogsRead(): void; // Method to reset new logs flag
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private alerts: Alert[];
  private logs: LogMessage[]; // Define logs property
  private maxLogs: number = 1000; // Max logs to keep
  public newLogsAvailable: boolean = false; // Define flag
  currentId: number;

  constructor() {
    this.users = new Map();
    this.currentId = 1;
    this.logs = []; // Initialize logs array
    this.newLogsAvailable = false; // Initialize flag
    
    // Initialize with some sample alerts
    this.alerts = [
      {
        id: "alert-1",
        type: "warning",
        title: "High gradient norm detected in Neural Memory",
        description: "The gradient norm of 0.8913 exceeds the recommended threshold of 0.7500.",
        timestamp: new Date(Date.now() - 12 * 60 * 1000).toISOString(), // 12 minutes ago
        source: "NeuralMemory"
      },
      {
        id: "alert-2",
        type: "info",
        title: "Memory Core index verification completed",
        description: "Successfully verified 342,891 memories and 6,452 assemblies. No inconsistencies found.",
        timestamp: new Date(Date.now() - 43 * 60 * 1000).toISOString(), // 43 minutes ago
        source: "MemoryCore"
      },
      {
        id: "alert-3",
        type: "warning",
        title: "CCE variant selection fluctuating",
        description: "Unusual switching between MAC and MAG variants detected (8 switches in 2 hours).",
        timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(), // 1 hour ago
        source: "CCE"
      }
    ];
  }

  // --- User methods (keep as is) ---
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  // --- Alert methods (keep as is) ---
  async getAlerts(): Promise<Alert[]> {
    return [...this.alerts].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }

  // --- Log methods (NEW/Refactored) ---
  getLogs(): LogMessage[] {
    // Return a copy to prevent external modification
    return [...this.logs];
  }

  addLog(log: LogMessage): void {
    // Add to beginning (newest first)
    this.logs.unshift(log);
    
    // Limit size
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(0, this.maxLogs);
    }
    
    this.newLogsAvailable = true; // Set flag
  }

  clearLogs(): void {
    this.logs = [];
    this.newLogsAvailable = false;
  }

  hasNewLogs(): boolean {
    return this.newLogsAvailable;
  }

  markLogsRead(): void {
    this.newLogsAvailable = false;
  }
}

export const storage = new MemStorage();

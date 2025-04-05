import { users, type User, type InsertUser, type Alert } from "@shared/schema";

// Extend the storage interface with additional methods
export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  getAlerts(): Promise<Alert[]>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private alerts: Alert[];
  currentId: number;

  constructor() {
    this.users = new Map();
    this.currentId = 1;
    
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
        description: "Unusual switching between MAC-7b and MAC-13b variants detected (8 switches in 2 hours).",
        timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(), // 1 hour ago
        source: "CCE"
      }
    ];
  }

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

  async getAlerts(): Promise<Alert[]> {
    return this.alerts;
  }
}

export const storage = new MemStorage();

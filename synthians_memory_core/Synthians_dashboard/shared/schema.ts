import { pgTable, text, serial, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Keep original user table
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Define types needed for dashboard
export interface ServiceStatus {
  name: string;
  status: 'Healthy' | 'Unhealthy' | 'Checking...' | 'Error';
  url: string;
  details?: string;
  uptime?: string;
  version?: string;
}

export interface MemoryStats {
  total_memories: number;
  total_assemblies: number;
  dirty_items: number;
  pending_vector_updates: number;
  vector_index: {
    count: number;
    mapping_count: number;
    drift_count: number;
    index_type: string;
    gpu_enabled: boolean;
  };
  assembly_stats: {
    total_count: number;
    indexed_count: number;
    vector_indexed_count: number;
    average_size: number;
    pruning_enabled: boolean;
    merging_enabled: boolean;
  };
  persistence: {
    last_update: string;
    last_backup: string;
  };
  performance: {
    quick_recall_rate: number;
    threshold_recall_rate: number;
  };
}

export interface NeuralMemoryStatus {
  initialized: boolean;
  config: {
    dimensions: number;
    hidden_size: number;
    layers: number;
  };
}

export interface NeuralMemoryDiagnostics {
  avg_loss: number;
  avg_grad_norm: number;
  avg_qr_boost: number;
  emotional_loop: {
    dominant_emotions: string[];
    entropy: number;
    bias_index: number;
    match_rate: number;
  };
  alerts: string[];
  recommendations: string[];
}

export interface CCEMetrics {
  recent_responses: CCEResponse[];
}

export interface CCEResponse {
  timestamp: string;
  status: 'success' | 'error';
  variant_output: {
    variant_type: string;
  };
  variant_selection?: {
    selected_variant: string;
    reason: string;
    performance_used: boolean;
  };
  llm_advice_used?: {
    raw_advice?: string;
    adjusted_advice: string;
    confidence_level: number;
    adjustment_reason?: string;
  };
  error_details?: string;
}

export interface Assembly {
  id: string;
  name: string;
  description: string;
  member_count: number;
  keywords: string[];
  tags: string[];
  topics: string[];
  created_at: string;
  updated_at: string;
  vector_index_updated_at?: string;
  memory_ids: string[];
}

export interface CCEConfig {
  active_variant: string;
  variant_confidence_threshold: number;
  llm_guidance_enabled: boolean;
  retry_attempts: number;
}

export interface Alert {
  id: string;
  type: 'error' | 'warning' | 'info';
  title: string;
  description: string;
  timestamp: string;
  source: 'MemoryCore' | 'NeuralMemory' | 'CCE';
  action?: string;
}

// Phase 5.9 Explainability types
export interface LineageEntry {
  assembly_id: string;
  merged_from: string[];
  timestamp: string;
  depth?: number; // For UI rendering
}

export interface ExplainMergeData {
  assembly_id: string;
  source_assembly_ids: string[];
  merge_timestamp: string;
  similarity_threshold: number;
  cleanup_status: 'pending' | 'completed' | 'failed';
  error?: string;
}

export interface ExplainActivationData {
  memory_id: string;
  assembly_id: string;
  activation_query_or_context: string;
  similarity_score: number;
  activation_threshold: number;
  activated: boolean;
  notes?: string;
}

export interface MergeLogEntry {
  event_type: 'merge' | 'cleanup_update';
  merge_event_id: string;
  timestamp: string;
  source_assembly_ids?: string[];
  status?: 'completed' | 'failed';
  error?: string;
}

export interface RuntimeConfig {
  // This will vary based on which service config is being fetched
  [key: string]: any;
}

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

// --- Phase 5.9 Explainability Interfaces ---

export interface ExplainActivationData {
  assembly_id: string;
  memory_id?: string | null;
  check_timestamp: string; // ISO string
  trigger_context?: string | null;
  assembly_state_before_check?: Record<string, any> | null;
  calculated_similarity?: number | null;
  activation_threshold?: number | null;
  passed_threshold?: boolean | null;
  notes?: string | null;
}

export interface ExplainActivationEmpty {
  assembly_id: string;
  memory_id?: string | null;
  notes: string;
}

export interface ExplainActivationResponse {
  success: boolean;
  explanation: ExplainActivationData | ExplainActivationEmpty;
  error?: string | null;
}

export interface ExplainMergeData {
  assembly_id: string;
  source_assembly_ids: string[];
  merge_timestamp: string;
  similarity_at_merge?: number | null;
  merge_threshold?: number | null;
  cleanup_status: 'pending' | 'completed' | 'failed';
  cleanup_timestamp?: string | null;
  cleanup_error?: string | null;
  notes?: string | null;
}

export interface ExplainMergeEmpty {
  assembly_id: string;
  notes: string;
}

export interface ExplainMergeResponse {
  success: boolean;
  explanation: ExplainMergeData | ExplainMergeEmpty;
  error?: string | null;
}

export interface LineageEntry {
  assembly_id: string;
  name?: string | null;
  depth: number;
  status?: string | null; // "origin", "merged", "cycle_detected", etc.
  created_at?: string | null; // ISO string
  memory_count?: number | null;
  parent_ids?: string[]; // IDs of source assemblies this was merged from
}

export interface LineageResponse {
  success: boolean;
  target_assembly_id: string;
  lineage: LineageEntry[];
  max_depth_reached: boolean;
  cycles_detected: boolean;
  error?: string | null;
}

// --- Phase 5.9 Diagnostics Interfaces ---

export interface ReconciledMergeLogEntry {
  merge_event_id: string;
  creation_timestamp: string; // ISO string
  source_assembly_ids: string[];
  target_assembly_id: string;
  similarity_at_merge?: number | null;
  merge_threshold?: number | null;
  final_cleanup_status: string; // "pending", "completed", "failed"
  cleanup_timestamp?: string | null; // ISO string
  cleanup_error?: string | null;
}

export interface MergeLogResponse {
  success: boolean;
  reconciled_log_entries: ReconciledMergeLogEntry[];
  count: number;
  query_limit: number;
  error?: string | null;
}

export interface RuntimeConfigResponse {
  success: boolean;
  service: string;
  config: Record<string, any>; // Sanitized config keys/values
  retrieval_timestamp: string; // ISO string
  error?: string | null;
}

export interface ActivationStats {
  total_activations: number;
  activations_by_assembly: Record<string, number>; // assembly_id -> count
  last_updated: string; // ISO timestamp
}

export interface ServiceMetrics {
  service_name: string;
  vector_operations: {
    avg_latency_ms: number;
    operation_counts: Record<string, number>; // operation -> count
  };
  persistence_operations: {
    avg_latency_ms: number;
    operation_counts: Record<string, number>; // operation -> count
  };
  // Other metrics fields
}

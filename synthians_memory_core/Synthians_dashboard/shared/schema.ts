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

// ServiceStatus interfaces for health endpoints
export interface ServiceStatusData {
  status: string; // 'healthy' or 'unhealthy'
  uptime_seconds?: number;
  uptime?: string; // For Neural Memory which returns uptime as string
  version?: string;
  memory_count?: number;
  assembly_count?: number;
  error?: string | null;
  timestamp?: string; // For Neural Memory which includes timestamp
  tensorflow_version?: string; // For Neural Memory
  neural_memory_initialized?: boolean; // For Neural Memory
}

export interface ServiceStatusResponse {
  success: boolean;
  data?: ServiceStatusData;
  error?: string | null;
}

// UI representation (used in components)
export interface ServiceStatus {
  name: string;
  status: 'Healthy' | 'Unhealthy' | 'Checking...' | 'Error';
  url: string;
  details?: string;
  uptime?: string;
  version?: string;
}

// Memory Stats interfaces for stats endpoints
export interface MemoryVectorIndexStats {
  count: number;
  mapping_count: number;
  drift_count: number;
  index_type: string;
  is_gpu: boolean;
  is_id_map: boolean;
  drift_warning?: boolean;
  drift_critical?: boolean;
  // Added fields from Phase 5.8
  total_vectors?: number;
  index_size_mb?: number;
  vector_dimensions?: number;
  healthy?: boolean;
  pending_updates?: number;
  last_update_at?: string;
}

export interface MemoryAssemblyStats {
  total_count: number;
  indexed_count: number;
  vector_indexed_count: number;
  average_size: number;
  pruning_enabled: boolean;
  merging_enabled: boolean;
  activation_threshold?: number;
  total_activations?: number;
  avg_activation_level?: number;
}

export interface MemoryCoreStatsData {
  total_memories: number;
  total_assemblies: number;
  dirty_memories: number;
  pending_vector_updates: number;
  initialized: boolean;
  uptime_seconds?: number;
}

export interface MemoryStatsData {
  core_stats: MemoryCoreStatsData;
  persistence_stats?: {
    last_update?: string;
    last_backup?: string;
  };
  quick_recall_stats?: {
    recall_rate?: number;
    avg_latency_ms?: number;
    count?: number;
  };
  threshold_stats?: {
    recall_rate?: number;
    avg_latency_ms?: number;
    count?: number;
  };
  vector_index_stats: MemoryVectorIndexStats;
  assemblies: MemoryAssemblyStats;
}

export interface MemoryStatsResponse {
  success: boolean;
  data?: MemoryStatsData;
  error?: string | null;
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
  history?: Array<{
    timestamp: string;
    loss: number;
    grad_norm: number;
  }>;
  alerts: string[];
  recommendations: string[];
}

export interface NeuralMemoryDiagnosticsResponse {
  success: boolean;
  data?: NeuralMemoryDiagnostics;
  error?: string | null;
}

export interface CCEResponse {
  timestamp: string;
  status: 'success' | 'error';
  input?: string;
  error?: string;
  variant_output: {
    variant_type: string;
  };
  variant_selection?: {
    selected_variant: string;
    reason: string;
    performance_used: boolean;
  };
  llm_advice?: string;
  llm_advice_used?: {
    raw_advice?: string;
    adjusted_advice: string;
    confidence_level: number;
    adjustment_reason?: string;
  };
  error_details?: string;
}

export interface CCEConfig {
  active_variant: string;
  variant_confidence_threshold: number;
  llm_guidance_enabled: boolean;
  retry_attempts: number;
}

export interface CCEConfigResponse {
  success: boolean;
  data?: CCEConfig;
  error?: string | null;
}

export interface CCEMetricsData {
  recent_responses: CCEResponse[];
  avg_response_time_ms?: number;
}

export interface CCEMetricsResponse {
  success: boolean;
  data?: CCEMetricsData;
  error?: string | null;
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

export interface AssembliesResponse {
  success: boolean;
  data?: Assembly[];
  error?: string | null;
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

export interface AlertsResponse {
  success: boolean;
  data?: Alert[];
  error?: string | null;
}

// CCE Status interfaces for status endpoints
export interface CCEStatusData {
  status: string; // 'OK' or 'INITIALIZING', etc.
  uptime: string;
  is_processing: boolean;
  current_variant: string;
  dev_mode: boolean;
  memory_stats?: {
    used_mb: number;
    total_mb: number;
    percentage: number;
  };
}

export interface CCEStatusResponse {
  success: boolean;
  data?: CCEStatusData;
  error?: string | null;
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

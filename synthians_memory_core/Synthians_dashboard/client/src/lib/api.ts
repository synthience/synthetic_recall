import axios from 'axios';
import { useQuery, QueryFunction } from '@tanstack/react-query';
import { 
  ServiceStatus, 
  ServiceStatusResponse,
  MemoryStatsResponse, 
  NeuralMemoryStatus, 
  NeuralMemoryDiagnosticsResponse,
  CCEResponse,
  CCEMetricsResponse,
  CCEConfig,
  CCEConfigResponse,
  Assembly,
  AssembliesResponse,
  Alert,
  AlertsResponse,
  ExplainActivationResponse,
  ExplainMergeResponse,
  LineageResponse,
  MergeLogResponse,
  RuntimeConfigResponse,
  CCEStatusResponse
} from '@shared/schema';

const api = axios.create({
  baseURL: '/api'
});

const defaultQueryFn = async <TData>({ queryKey }: { queryKey: readonly unknown[] }): Promise<TData> => {
  let url = '';
  const params: Record<string, any> = {};
  queryKey.forEach(part => {
    if (typeof part === 'string') {
      url += `/${part}`;
    } else if (typeof part === 'object' && part !== null) {
      Object.assign(params, part);
    }
  });
  if (url.startsWith('/')) {
    url = url.substring(1);
  }
  try {
    const { data } = await api.get(url, { params });
    return data as TData;
  } catch (error: any) {
    console.error(`API Query Error for ${url}:`, error.response?.data || error.message);
    throw new Error(error.response?.data?.message || error.message || `Failed to fetch ${url}`);
  }
};

export const useMemoryCoreHealth = () => {
  return useQuery<ServiceStatusResponse>({
    queryKey: ['memory-core', 'health'],
    queryFn: () => defaultQueryFn<ServiceStatusResponse>({ queryKey: ['memory-core', 'health'] }),
    refetchInterval: false, 
    retry: 2
  });
};

export const useNeuralMemoryHealth = () => {
  return useQuery<ServiceStatusResponse>({
    queryKey: ['neural-memory', 'health'],
    queryFn: () => defaultQueryFn<ServiceStatusResponse>({ queryKey: ['neural-memory', 'health'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEHealth = () => {
  return useQuery<ServiceStatusResponse>({
    queryKey: ['cce', 'health'],
    queryFn: () => defaultQueryFn<ServiceStatusResponse>({ queryKey: ['cce', 'health'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useMemoryCoreStats = () => {
  return useQuery<MemoryStatsResponse>({
    queryKey: ['memory-core', 'stats'],
    queryFn: () => defaultQueryFn<MemoryStatsResponse>({ queryKey: ['memory-core', 'stats'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useAssemblies = () => {
  return useQuery<AssembliesResponse>({
    queryKey: ['memory-core', 'assemblies'],
    queryFn: () => defaultQueryFn<AssembliesResponse>({ queryKey: ['memory-core', 'assemblies'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useAssembly = (id: string | null) => {
  return useQuery<Assembly>({
    queryKey: ['memory-core', 'assemblies', id],
    queryFn: () => defaultQueryFn<Assembly>({ queryKey: ['memory-core', 'assemblies', id] }),
    enabled: !!id,
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryStatus = () => {
  return useQuery<NeuralMemoryStatus>({
    queryKey: ['neural-memory', 'status'],
    queryFn: () => defaultQueryFn<NeuralMemoryStatus>({ queryKey: ['neural-memory', 'status'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryDiagnostics = (window: string = '24h') => {
  return useQuery<NeuralMemoryDiagnosticsResponse>({
    queryKey: ['neural-memory', 'diagnose_emoloop', { window }],
    queryFn: () => defaultQueryFn<NeuralMemoryDiagnosticsResponse>({ queryKey: ['neural-memory', 'diagnose_emoloop', { window }] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEStatus = () => {
  return useQuery<CCEStatusResponse>({
    queryKey: ['cce', 'status'],
    queryFn: () => defaultQueryFn<CCEStatusResponse>({ queryKey: ['cce', 'status'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useRecentCCEResponses = () => {
  return useQuery<CCEMetricsResponse>({
    queryKey: ['cce', 'metrics', 'recent_cce_responses'],
    queryFn: () => defaultQueryFn<CCEMetricsResponse>({ queryKey: ['cce', 'metrics', 'recent_cce_responses'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryConfig = () => {
  return useQuery<CCEConfigResponse>({
    queryKey: ['neural-memory', 'config'],
    queryFn: () => defaultQueryFn<CCEConfigResponse>({ queryKey: ['neural-memory', 'config'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEConfig = () => {
  return useQuery<CCEConfigResponse>({
    queryKey: ['cce', 'config'],
    queryFn: () => defaultQueryFn<CCEConfigResponse>({ queryKey: ['cce', 'config'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useAlerts = () => {
  return useQuery<AlertsResponse>({
    queryKey: ['alerts'],
    queryFn: () => defaultQueryFn<AlertsResponse>({ queryKey: ['alerts'] }),
    refetchInterval: false,
    retry: 2
  });
};

export const useExplainActivation = (assemblyId: string | null, memoryId?: string | null) => {
  const queryParams = memoryId ? { memory_id: memoryId } : {};
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'explain_activation', queryParams] as const;
  return useQuery<ExplainActivationResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<ExplainActivationResponse>({ queryKey }),
    enabled: false, 
    retry: 1,
    staleTime: Infinity,
  });
};

export const useExplainMerge = (assemblyId: string | null) => {
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'explain_merge'] as const;
  return useQuery<ExplainMergeResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<ExplainMergeResponse>({ queryKey }),
    enabled: false, 
    retry: 1,
    staleTime: Infinity,
  });
};

export const useAssemblyLineage = (assemblyId: string | null) => {
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'lineage'] as const;
  return useQuery<LineageResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<LineageResponse>({ queryKey }),
    enabled: !!assemblyId, 
    retry: 1,
    staleTime: 5 * 60 * 1000, 
  });
};

export const useMergeLog = (limit: number = 50) => {
  const queryKey = ['memory-core', 'diagnostics', 'merge_log', { limit }] as const;
  return useQuery<MergeLogResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<MergeLogResponse>({ queryKey }),
    refetchInterval: 30000, 
    staleTime: 15000, 
  });
};

export const useRuntimeConfig = (serviceName: string | null) => {
  const queryKey = ['memory-core', 'config', 'runtime', serviceName] as const;
  return useQuery<RuntimeConfigResponse>({
    queryKey: queryKey,
    queryFn: () => defaultQueryFn<RuntimeConfigResponse>({ queryKey }),
    enabled: !!serviceName,
    staleTime: 10 * 60 * 1000, 
  });
};

export const verifyMemoryCoreIndex = async () => { // Renamed back to original name to fix import error
  // Correct method and path
  return api.get('/memory-core/check_index_integrity');
};

export const triggerMemoryCoreRetryLoop = async () => {
  // Correct method and path (uses the NEWLY implemented endpoint)
  return api.post('/memory-core/diagnostics/trigger_retry_loop');
};

// Add missing function from plan
export const repairMemoryCoreIndex = async (repairType: string = 'auto') => {
  // Correct method and path
  return api.post('/memory-core/repair_index', { repair_type: repairType });
};

export const initializeNeuralMemory = async () => {
  return api.post('/neural-memory/init');
};

export const setCCEVariant = async (variant: string) => {
  return api.post('/cce/set_variant', { variant });
};

export const refreshAllData = async (queryClient: any) => {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'health'] }),
    queryClient.invalidateQueries({ queryKey: ['neural-memory', 'health'] }),
    queryClient.invalidateQueries({ queryKey: ['cce', 'health'] }),
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'stats'] }),
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'assemblies'] }),
    queryClient.invalidateQueries({ queryKey: ['neural-memory', 'status'] }),
    queryClient.invalidateQueries({ queryKey: ['neural-memory', 'diagnose_emoloop'] }),
    queryClient.invalidateQueries({ queryKey: ['cce', 'status'] }),
    queryClient.invalidateQueries({ queryKey: ['cce', 'metrics', 'recent_cce_responses'] }),
    queryClient.invalidateQueries({ queryKey: ['alerts'] }),
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'config', 'runtime'] }),
    queryClient.invalidateQueries({ queryKey: ['memory-core', 'diagnostics', 'merge_log'] })
  ]);
};

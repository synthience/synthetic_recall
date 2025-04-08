import axios from 'axios';
import { useQuery, QueryFunction, QueryFunctionContext, UseQueryOptions } from '@tanstack/react-query';
import { 
  ServiceStatusResponse, 
  MemoryStatsResponse, 
  AssembliesResponse, 
  AssemblyResponse, 
  NeuralMemoryStatusResponse, 
  NeuralMemoryStatus,
  NeuralMemoryDiagnosticsResponse, 
  CCEResponse, 
  CCEMetricsResponse, 
  CCEMetricsData,
  CCEConfig, 
  CCEConfigResponse, 
  NeuralMemoryConfigResponse, 
  Assembly, 
  Alert, 
  AlertsResponse, 
  ExplainActivationResponse, 
  ExplainMergeResponse, 
  LineageResponse, 
  MergeLogResponse, 
  RuntimeConfigResponse, 
  ApiResponse,
  ServiceStatusData,
  MemoryStatsData,
  NeuralMemoryDiagnostics,
  CCEStatusData
} from '@shared/schema';

type FetchDirection = 'forward' | 'backward';

const api = axios.create({
  baseURL: '/api/proxy',
});

type DefaultQueryContext = QueryFunctionContext<readonly unknown[], any>;

const defaultQueryFn: QueryFunction<any, readonly unknown[], any> = async (context) => {
  const { queryKey, signal, meta } = context;
  // Ensure pageParam and direction have sensible defaults
  const finalPageParam = context.pageParam !== undefined ? context.pageParam : null;
  // We use 'as any' cast on context to extract direction, defaulting to 'forward'
  const finalDirection = (context as any).direction !== undefined ? (context as any).direction as FetchDirection : 'forward';
  
  // Build URL from query key parts
  const url = queryKey
    .filter(key => typeof key === 'string' || key === null)
    .map(key => encodeURIComponent(String(key)))
    .join('/');

  // Extract query parameters from the last item in queryKey if it's an object
  const lastItem = queryKey[queryKey.length - 1];
  const params = typeof lastItem === 'object' && lastItem !== null && !Array.isArray(lastItem)
    ? lastItem
    : {};

  // Log the request details for better debugging
  console.log(`[API] Making request to: /api/proxy/${url}`, { params });

  try {
    const response = await api.get<ApiResponse<any>>(url, { params, signal });
    const responseData = response.data;
    
    if (responseData.success) {
      // Return the full API response so that UI code accessing properties like data, lineage, explanation, etc., remains valid
      console.log(`[API] Successful response for ${url}:`, { 
        success: responseData.success, 
        dataKeys: responseData.data ? Object.keys(responseData.data) : 'none' 
      });
      return responseData;
    } else {
      console.error(`[API] Error response for ${url}:`, responseData.error || 'No error details');
      throw new Error(responseData.error || 'Request failed');
    }
  } catch (error) {
    console.error(`[API] Exception for ${url}:`, error);
    throw error; // Re-throw to let TanStack Query handle retries
  }
};

export const useMemoryCoreHealth = (options?: Partial<UseQueryOptions<ApiResponse<ServiceStatusData>>>) => {
  return useQuery<ApiResponse<ServiceStatusData>>(
    {
      queryKey: ['memory-core', 'health'],
      queryFn: defaultQueryFn,
      refetchInterval: false, 
      retry: 2,
      ...options
    }
  );
};

export const useNeuralMemoryHealth = (options?: Partial<UseQueryOptions<ApiResponse<ServiceStatusData>>>) => {
  return useQuery<ApiResponse<ServiceStatusData>>(
    {
      queryKey: ['neural-memory', 'health'],
      queryFn: defaultQueryFn,
      refetchInterval: false,
      retry: 2,
      ...options
    }
  );
};

export const useCCEHealth = (options?: Partial<UseQueryOptions<ApiResponse<ServiceStatusData>>>) => {
  return useQuery<ApiResponse<ServiceStatusData>>(
    {
      queryKey: ['cce', 'health'],
      queryFn: defaultQueryFn,
      refetchInterval: false,
      retry: 2,
      ...options
    }
  );
};

export const useMemoryCoreStats = (options?: Partial<UseQueryOptions<ApiResponse<MemoryStatsData>>>) => {
  return useQuery<ApiResponse<MemoryStatsData>>(
    {
      queryKey: ['memory-core', 'stats'],
      queryFn: defaultQueryFn,
      refetchInterval: false,
      retry: 2,
      ...options
    }
  );
};

export const useAssemblies = (options?: Partial<UseQueryOptions<ApiResponse<Assembly[]>>>) => {
  return useQuery<ApiResponse<Assembly[]>>(
    {
      queryKey: ['memory-core', 'assemblies'],
      queryFn: defaultQueryFn,
      refetchInterval: false,
      retry: 2,
      ...options
    }
  );
};

export const useNeuralMemoryDiagnostics = (window: string = '24h', options?: Partial<UseQueryOptions<ApiResponse<NeuralMemoryDiagnostics>>>) => {
  return useQuery<ApiResponse<NeuralMemoryDiagnostics>>(
    {
      queryKey: ['neural-memory', 'diagnose_emoloop', { window }],
      queryFn: defaultQueryFn,
      refetchInterval: false,
      retry: 2,
      ...options
    }
  );
};

export const useRecentCCEResponses = (options?: Partial<UseQueryOptions<ApiResponse<CCEMetricsData>>>) => {
  return useQuery<ApiResponse<CCEMetricsData>>(
    {
      queryKey: ['cce', 'metrics', 'recent_cce_responses'],
      queryFn: defaultQueryFn,
      refetchInterval: false,
      retry: 2,
      ...options
    }
  );
};

export const useAlerts = (options?: Partial<UseQueryOptions<ApiResponse<Alert[]>>>) => {
  return useQuery<ApiResponse<Alert[]>>(
    {
      queryKey: ['memory-core', 'alerts'],
      queryFn: defaultQueryFn,
      refetchInterval: false,
      retry: 2,
      ...options
    }
  );
};

export const useAssembly = (id: string | null) => {
  return useQuery<ApiResponse<Assembly>>({
    queryKey: ['memory-core', 'assemblies', id],
    queryFn: defaultQueryFn,
    enabled: !!id,
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryStatus = () => {
  return useQuery<ApiResponse<NeuralMemoryStatus>>({
    queryKey: ['neural-memory', 'status'],
    queryFn: defaultQueryFn,
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEStatus = () => {
  return useQuery<ApiResponse<CCEStatusData>>({
    queryKey: ['cce', 'status'],
    queryFn: defaultQueryFn,
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryConfig = () => {
  return useQuery<ApiResponse<Record<string, any>>>({
    queryKey: ['neural-memory', 'config'],
    queryFn: defaultQueryFn,
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEConfig = () => {
  return useQuery<ApiResponse<CCEConfig>>({
    queryKey: ['cce', 'config'],
    queryFn: defaultQueryFn,
    refetchInterval: false,
    retry: 2
  });
};

export const useExplainActivation = (assemblyId: string | null, memoryId: string | undefined) => {
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'explain_activation', { memory_id: memoryId }] as const;
  return useQuery<ApiResponse<ExplainActivationResponse>, Error>({
    queryKey,
    queryFn: defaultQueryFn, 
    enabled: false, 
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
};

export const useExplainMerge = (assemblyId: string | null) => {
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'explain_merge'] as const;
  return useQuery<ApiResponse<ExplainMergeResponse>, Error>({
    queryKey,
    queryFn: defaultQueryFn, 
    enabled: false, 
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
};

export const useAssemblyLineage = (assemblyId: string | null, maxDepth: number = 5) => {
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'lineage', { max_depth: maxDepth }] as const;
  return useQuery<ApiResponse<LineageResponse>, Error>({
    queryKey,
    queryFn: defaultQueryFn, 
    enabled: false, 
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
};

// This is a duplicate of useAssemblyLineage with a different name - keeping for backward compatibility
// Eventually, this should be removed and all components should use useAssemblyLineage
export const useLineage = (assemblyId: string | null, maxDepth: number = 5) => {
  const queryKey = ['memory-core', 'assemblies', assemblyId, 'lineage', { max_depth: maxDepth }] as const;
  return useQuery<ApiResponse<LineageResponse>, Error>({
    queryKey,
    queryFn: defaultQueryFn,
    enabled: false, 
    retry: 1,
    staleTime: Infinity,
  });
};

export const useMergeLog = (limit: number = 50, options?: Partial<UseQueryOptions<ApiResponse<MergeLogResponse>, Error>>) => {
  const queryKey = ['memory-core', 'diagnostics', 'merge_log', { limit }] as const;
  return useQuery<ApiResponse<MergeLogResponse>, Error>({
    queryKey,
    queryFn: defaultQueryFn,
    refetchInterval: 30000, 
    staleTime: 15000,
    ...options
  });
};

export const useRuntimeConfig = (serviceName: string | null) => {
  const queryKey = ['memory-core', 'config', 'runtime', serviceName] as const;
  return useQuery<ApiResponse<RuntimeConfigResponse>, Error>({
    queryKey,
    queryFn: defaultQueryFn,
    enabled: !!serviceName,
    staleTime: 10 * 60 * 1000, 
    // Reduce retries since our proxy handles fallbacks
    retry: 1,
    retryDelay: 500,
    // Set a longer timeout for config operations which might be slow
    meta: { timeoutMs: 30000 }
  });
};

export const verifyMemoryCoreIndex = async () => { 
  return api.post('/memory-core/admin/verify_index');
};

export const triggerMemoryCoreRetryLoop = async () => {
  return api.post('/memory-core/admin/trigger_retry_loop');
};

export const repairMemoryCoreIndex = async (repairType: string = 'auto') => {
  return api.post('/memory-core/admin/repair_index', { repair_type: repairType });
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

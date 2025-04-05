import axios from 'axios';
import { useQuery } from '@tanstack/react-query';
import { 
  ServiceStatus, 
  MemoryStats, 
  NeuralMemoryStatus, 
  NeuralMemoryDiagnostics,
  CCEMetrics,
  Assembly,
  Alert,
  CCEConfig
} from '@shared/schema';

const api = axios.create({
  baseURL: '/api'
});

// Services health checks
export const useMemoryCoreHealth = () => {
  return useQuery({
    queryKey: ['/api/memory-core/health'],
    refetchInterval: false, // Polling managed by the store
    retry: 2
  });
};

export const useNeuralMemoryHealth = () => {
  return useQuery({
    queryKey: ['/api/neural-memory/health'],
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEHealth = () => {
  return useQuery({
    queryKey: ['/api/cce/health'],
    refetchInterval: false,
    retry: 2
  });
};

// Memory Core data
export const useMemoryCoreStats = () => {
  return useQuery({
    queryKey: ['/api/memory-core/stats'],
    refetchInterval: false,
    retry: 2
  });
};

export const useAssemblies = () => {
  return useQuery({
    queryKey: ['/api/memory-core/assemblies'],
    refetchInterval: false,
    retry: 2
  });
};

export const useAssembly = (id: string | null) => {
  return useQuery({
    queryKey: ['/api/memory-core/assemblies', id],
    enabled: !!id,
    refetchInterval: false,
    retry: 2
  });
};

// Neural Memory data
export const useNeuralMemoryStatus = () => {
  return useQuery({
    queryKey: ['/api/neural-memory/status'],
    refetchInterval: false,
    retry: 2
  });
};

export const useNeuralMemoryDiagnostics = (window: string = '24h') => {
  return useQuery({
    queryKey: ['/api/neural-memory/diagnose_emoloop', window],
    refetchInterval: false,
    retry: 2
  });
};

// CCE data
export const useCCEStatus = () => {
  return useQuery({
    queryKey: ['/api/cce/status'],
    refetchInterval: false,
    retry: 2
  });
};

export const useRecentCCEResponses = () => {
  return useQuery({
    queryKey: ['/api/cce/metrics/recent_cce_responses'],
    refetchInterval: false,
    retry: 2
  });
};

// Configuration data
export const useNeuralMemoryConfig = () => {
  return useQuery({
    queryKey: ['/api/neural-memory/config'],
    refetchInterval: false,
    retry: 2
  });
};

export const useCCEConfig = () => {
  return useQuery({
    queryKey: ['/api/cce/config'],
    refetchInterval: false,
    retry: 2
  });
};

// Alerts
export const useAlerts = () => {
  return useQuery({
    queryKey: ['/api/alerts'],
    refetchInterval: false,
    retry: 2
  });
};

// Admin actions
export const verifyMemoryCoreIndex = async () => {
  return api.post('/memory-core/admin/verify_index');
};

export const triggerMemoryCoreRetryLoop = async () => {
  return api.post('/memory-core/admin/trigger_retry_loop');
};

export const initializeNeuralMemory = async () => {
  return api.post('/neural-memory/init');
};

export const setCCEVariant = async (variant: string) => {
  return api.post('/cce/set_variant', { variant });
};

// Helper for manual refresh of all data
export const refreshAllData = async (queryClient: any) => {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: ['/api/memory-core/health'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/neural-memory/health'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/cce/health'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/memory-core/stats'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/memory-core/assemblies'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/neural-memory/status'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/neural-memory/diagnose_emoloop'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/cce/status'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/cce/metrics/recent_cce_responses'] }),
    queryClient.invalidateQueries({ queryKey: ['/api/alerts'] })
  ]);
};

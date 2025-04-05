import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Assembly } from '@shared/schema';

/**
 * Hook to fetch the details of a specific assembly
 */
export const useAssemblyDetails = (assemblyId: string | null | undefined) => {
  return useQuery<Assembly>({
    queryKey: ['/api/memory-core/assemblies/details', assemblyId],
    queryFn: async () => {
      if (!assemblyId) {
        throw new Error('Assembly ID is required');
      }
      const response = await axios.get(`/api/memory-core/assemblies/${assemblyId}`);
      return response.data;
    },
    enabled: !!assemblyId,
    retry: 2,
  });
};

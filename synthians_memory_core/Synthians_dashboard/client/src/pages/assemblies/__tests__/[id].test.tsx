import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { useAssembly, useAssemblyLineage, useExplainMerge, useExplainActivation } from '@/lib/api';
import { useFeatures } from '@/contexts/FeaturesContext';
import AssemblyDetail from '../[id]';
import { toast } from '@/hooks/use-toast';

// Mock the hooks and context
jest.mock('@/lib/api', () => ({
  useAssembly: jest.fn(),
  useAssemblyLineage: jest.fn(),
  useExplainMerge: jest.fn(),
  useExplainActivation: jest.fn(),
}));

jest.mock('@/contexts/FeaturesContext', () => ({
  useFeatures: jest.fn(),
}));

jest.mock('@/hooks/use-toast', () => ({
  toast: jest.fn(),
}));

jest.mock('wouter', () => ({
  useParams: () => ({ id: 'assembly123' }),
  Link: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

describe('AssemblyDetail', () => {
  const mockAssembly = {
    id: 'assembly123',
    name: 'Test Assembly',
    created_at: '2025-04-05T10:00:00Z',
    updated_at: '2025-04-05T12:00:00Z',
    member_count: 42,
    vector_index_updated_at: '2025-04-05T11:00:00Z',
    members: [
      { id: 'memory1', content: 'Test memory 1', type: 'semantic', created_at: '2025-04-01T10:00:00Z' },
      { id: 'memory2', content: 'Test memory 2', type: 'episodic', created_at: '2025-04-02T10:00:00Z' },
    ],
  };

  const mockLineage = {
    nodes: [{ id: 'assembly123', type: 'assembly' }],
    edges: [],
  };

  const mockMergeExplanation = {
    source_assembly: 'source123',
    target_assembly: 'assembly123',
    merge_timestamp: '2025-04-04T10:00:00Z',
    similarity_score: 0.85,
    merge_decision: 'Merged due to high similarity score',
    key_similarities: ['Similar content themes', 'Shared key entities'],
  };

  const mockActivationExplanation = {
    memory_id: 'memory1',
    activation_score: 0.92,
    activation_timestamp: '2025-04-05T13:00:00Z',
    query_similarity: 0.88,
    context_boost: 0.04,
    features_used: ['Content similarity', 'Recency bias'],
  };

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();

    // Mock hook implementations
    (useAssembly as jest.Mock).mockReturnValue({
      data: mockAssembly,
      isLoading: false,
      isError: false,
      error: null,
      refetch: jest.fn(),
    });

    (useAssemblyLineage as jest.Mock).mockReturnValue({
      data: { lineage: mockLineage },
      isLoading: false,
      isError: false,
      error: null,
      refetch: jest.fn(),
    });

    (useExplainMerge as jest.Mock).mockReturnValue({
      data: { explanation: mockMergeExplanation },
      isLoading: false,
      isError: false,
      error: null,
      refetch: jest.fn(),
    });

    (useExplainActivation as jest.Mock).mockReturnValue({
      data: { explanation: mockActivationExplanation },
      isLoading: false,
      isError: false,
      error: null,
      refetch: jest.fn(),
    });

    // Mock feature flags
    (useFeatures as jest.Mock).mockReturnValue({
      explainabilityEnabled: true,
    });
  });

  test('renders assembly details correctly', () => {
    render(<AssemblyDetail />);

    // Check basic assembly info
    expect(screen.getByText('Test Assembly')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument(); // member count
    
    // Check assembly members
    expect(screen.getByText('Test memory 1')).toBeInTheDocument();
    expect(screen.getByText('Test memory 2')).toBeInTheDocument();
  });

  test('renders loading state correctly', () => {
    (useAssembly as jest.Mock).mockReturnValue({
      isLoading: true,
      isError: false,
      error: null,
      refetch: jest.fn(),
    });

    render(<AssemblyDetail />);

    // Check for skeletons (simplistic check)
    const skeletons = document.querySelectorAll('.h-4, .h-6, .h-8');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  test('renders error state correctly', () => {
    const mockError = new Error('Failed to load assembly');
    
    (useAssembly as jest.Mock).mockReturnValue({
      isLoading: false,
      isError: true,
      error: mockError,
      refetch: jest.fn(),
    });

    render(<AssemblyDetail />);

    // Check for error message
    expect(screen.getByText('Assembly Not Found')).toBeInTheDocument();
    expect(toast).toHaveBeenCalledWith({
      title: 'Error loading assembly',
      description: 'Failed to load assembly',
      variant: 'destructive',
    });
  });

  test('hides explainability tabs when disabled', () => {
    // Mock explainability disabled
    (useFeatures as jest.Mock).mockReturnValue({
      explainabilityEnabled: false,
    });

    render(<AssemblyDetail />);

    // Check that base tabs exist
    expect(screen.getByText('Overview')).toBeInTheDocument();
    expect(screen.getByText('Members')).toBeInTheDocument();
    
    // Check that explainability tabs don't exist
    expect(screen.queryByText('Lineage')).not.toBeInTheDocument();
    expect(screen.queryByText('Merge')).not.toBeInTheDocument();
    expect(screen.queryByText('Activation')).not.toBeInTheDocument();
    
    // Check that the explainability disabled message is shown
    expect(screen.getByText('Explainability Features Disabled')).toBeInTheDocument();
  });

  test('refetches explainability data when tab changes', async () => {
    const lineageRefetch = jest.fn();
    const mergeRefetch = jest.fn();
    
    (useAssemblyLineage as jest.Mock).mockReturnValue({
      data: { lineage: mockLineage },
      isLoading: false,
      isError: false,
      error: null,
      refetch: lineageRefetch,
    });

    (useExplainMerge as jest.Mock).mockReturnValue({
      data: { explanation: mockMergeExplanation },
      isLoading: false,
      isError: false,
      error: null,
      refetch: mergeRefetch,
    });

    render(<AssemblyDetail />);

    // Click on Lineage tab
    fireEvent.click(screen.getByText('Lineage'));
    
    // Lineage data should be fetched
    expect(lineageRefetch).toHaveBeenCalled();
    
    // Click on Merge tab
    fireEvent.click(screen.getByText('Merge'));
    
    // Merge data should be fetched
    expect(mergeRefetch).toHaveBeenCalled();
  });

  test('refetches activation data when memory is selected', async () => {
    const activationRefetch = jest.fn();
    
    (useExplainActivation as jest.Mock).mockReturnValue({
      data: { explanation: mockActivationExplanation },
      isLoading: false,
      isError: false,
      error: null,
      refetch: activationRefetch,
    });

    render(<AssemblyDetail />);

    // Click on Activation tab
    fireEvent.click(screen.getByText('Activation'));
    
    // Find and click a memory item
    const memoryItems = screen.getAllByText(/Test memory/);
    fireEvent.click(memoryItems[0]);
    
    // Wait for the refetch to be called (due to the setTimeout)
    await waitFor(() => {
      expect(activationRefetch).toHaveBeenCalled();
    });
  });
});

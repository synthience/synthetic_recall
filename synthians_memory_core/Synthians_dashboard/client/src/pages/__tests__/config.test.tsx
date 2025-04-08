import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { useRuntimeConfig } from '@/lib/api';
import { useFeatures } from '@/contexts/FeaturesContext';
import { usePollingStore } from '@/lib/store';
import Config from '../config';

// Mock the hooks
jest.mock('@/lib/api', () => ({
  useRuntimeConfig: jest.fn(),
}));

jest.mock('@/contexts/FeaturesContext', () => ({
  useFeatures: jest.fn(),
}));

jest.mock('@/lib/store', () => ({
  usePollingStore: jest.fn(),
}));

describe('Config', () => {
  // Mock response data for each service
  const mockMemoryConfig = {
    success: true,
    config: {
      memory_core: {
        vector_index: {
          enabled: true,
          dimensions: 768,
          update_interval: 60,
        },
        storage: {
          persistence_enabled: true,
          backup_interval: 3600,
        },
      },
    },
  };

  const mockNeuralConfig = {
    success: true,
    config: {
      neural_memory: {
        model: {
          type: 'transformer',
          embedding_size: 768,
          layers: 12,
        },
        training: {
          learning_rate: 0.001,
          batch_size: 64,
        },
      },
    },
  };

  const mockCCEConfig = {
    success: true,
    config: {
      cce: {
        variants: {
          enabled: ['MAG', 'MAC', 'MAL'],
          default: 'MAG',
        },
        selection: {
          method: 'adaptive',
          history_window: 10,
        },
      },
    },
  };

  // Mock the refetch function for all services
  const mockRefetch = jest.fn();

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();

    // Default mock for useFeatures
    (useFeatures as jest.Mock).mockReturnValue({
      explainabilityEnabled: true,
    });

    // Default mock for usePollingStore
    (usePollingStore as jest.Mock).mockReturnValue({
      refreshAllData: jest.fn(),
    });

    // Default implementation for useRuntimeConfig
    (useRuntimeConfig as jest.Mock).mockImplementation((service) => {
      if (service === 'memory-core') {
        return {
          data: mockMemoryConfig,
          isLoading: false,
          isError: false,
          refetch: mockRefetch,
        };
      } else if (service === 'neural-memory') {
        return {
          data: mockNeuralConfig,
          isLoading: false,
          isError: false,
          refetch: mockRefetch,
        };
      } else if (service === 'cce') {
        return {
          data: mockCCEConfig,
          isLoading: false,
          isError: false,
          refetch: mockRefetch,
        };
      }
      return {
        isLoading: false,
        isError: false,
        refetch: mockRefetch,
      };
    });
  });

  test('renders memory-core config correctly', () => {
    render(<Config />);

    // Memory Core should be selected by default
    expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('Memory Core');

    // Check that some memory core config values are displayed
    expect(screen.getByText('vector_index')).toBeInTheDocument();
    expect(screen.getByText('dimensions')).toBeInTheDocument();
    expect(screen.getByText('768')).toBeInTheDocument();
  });

  test('switches to neural-memory config on tab click', () => {
    render(<Config />);

    // Click on Neural Memory tab
    fireEvent.click(screen.getByRole('tab', { name: /neural memory/i }));

    // Neural Memory should now be selected
    expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('Neural Memory');

    // Check that some neural memory config values are displayed
    expect(screen.getByText('model')).toBeInTheDocument();
    expect(screen.getByText('embedding_size')).toBeInTheDocument();
    expect(screen.getByText('768')).toBeInTheDocument();
  });

  test('switches to cce config on tab click', () => {
    render(<Config />);

    // Click on CCE tab
    fireEvent.click(screen.getByRole('tab', { name: /context cascade/i }));

    // CCE should now be selected
    expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('Context Cascade');

    // Check that some CCE config values are displayed
    expect(screen.getByText('variants')).toBeInTheDocument();
    expect(screen.getByText('enabled')).toBeInTheDocument();
    expect(screen.getByText('["MAG","MAC","MAL"]')).toBeInTheDocument();
  });

  test('renders loading state correctly', () => {
    // Mock loading state for memory-core
    (useRuntimeConfig as jest.Mock).mockImplementation((service) => {
      if (service === 'memory-core') {
        return {
          isLoading: true,
          isError: false,
          refetch: mockRefetch,
        };
      }
      return {
        isLoading: false,
        isError: false,
        refetch: mockRefetch,
      };
    });

    render(<Config />);

    // Check for skeletons
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  test('renders error state correctly', () => {
    const mockError = new Error('Failed to load config');
    
    // Mock error state for memory-core
    (useRuntimeConfig as jest.Mock).mockImplementation((service) => {
      if (service === 'memory-core') {
        return {
          isLoading: false,
          isError: true,
          error: mockError,
          refetch: mockRefetch,
        };
      }
      return {
        isLoading: false,
        isError: false,
        refetch: mockRefetch,
      };
    });

    render(<Config />);

    // Check for error message
    expect(screen.getByText('Failed to load configuration')).toBeInTheDocument();
    expect(screen.getByText('Failed to load config')).toBeInTheDocument();
  });

  test('shows explainability banner when enabled', () => {
    (useFeatures as jest.Mock).mockReturnValue({
      explainabilityEnabled: true,
    });

    render(<Config />);

    expect(screen.getByText('Explainability Features: Enabled')).toBeInTheDocument();
  });

  test('shows explainability disabled message when disabled', () => {
    (useFeatures as jest.Mock).mockReturnValue({
      explainabilityEnabled: false,
    });

    render(<Config />);

    expect(screen.getByText('Explainability Features: Disabled')).toBeInTheDocument();
  });

  test('refreshes config data when refresh button is clicked', () => {
    render(<Config />);

    // Find and click the refresh button
    const refreshButton = screen.getByRole('button', { name: /refresh/i });
    fireEvent.click(refreshButton);

    // Check that refetch was called
    expect(mockRefetch).toHaveBeenCalled();
  });
});

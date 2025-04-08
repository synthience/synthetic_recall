import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { MetricsChart } from '../MetricsChart';

// Mock recharts to avoid rendering issues in test environment
jest.mock('recharts', () => {
  const OriginalModule = jest.requireActual('recharts');
  return {
    ...OriginalModule,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
    LineChart: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="line-chart">{children}</div>
    )
  };
});

describe('MetricsChart', () => {
  const mockData = [
    { timestamp: '2025-04-05T10:00:00Z', loss: 0.42, grad_norm: 0.15 },
    { timestamp: '2025-04-05T11:00:00Z', loss: 0.39, grad_norm: 0.12 },
    { timestamp: '2025-04-05T12:00:00Z', loss: 0.35, grad_norm: 0.10 }
  ];

  const mockDataKeys = [
    { key: 'loss', color: '#ff0000', name: 'Loss' },
    { key: 'grad_norm', color: '#0000ff', name: 'Gradient Norm' }
  ];

  const mockSummary = [
    { label: 'Min Loss', value: '0.35', color: 'text-green-500' },
    { label: 'Max Loss', value: '0.42', color: 'text-red-500' },
    { label: 'Avg. Gradient', value: '0.12', color: 'text-blue-500' }
  ];

  const mockError = new Error('Failed to fetch metrics');

  const mockTimeRangeChange = jest.fn();

  test('renders correctly with data', () => {
    render(
      <MetricsChart
        title="Neural Memory Metrics"
        data={mockData}
        dataKeys={mockDataKeys}
        isLoading={false}
        isError={false}
        timeRange="12h"
        onTimeRangeChange={mockTimeRangeChange}
        summary={mockSummary}
      />
    );

    // Check if title is rendered
    expect(screen.getByText('Neural Memory Metrics')).toBeInTheDocument();
    
    // Check if chart container is rendered
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    
    // Check if summary items are rendered
    expect(screen.getByText('Min Loss')).toBeInTheDocument();
    expect(screen.getByText('0.35')).toBeInTheDocument();
    expect(screen.getByText('Max Loss')).toBeInTheDocument();
    expect(screen.getByText('0.42')).toBeInTheDocument();
    expect(screen.getByText('Avg. Gradient')).toBeInTheDocument();
    expect(screen.getByText('0.12')).toBeInTheDocument();
  });

  test('renders loading state correctly', () => {
    render(
      <MetricsChart
        title="Neural Memory Metrics"
        data={[]}
        dataKeys={mockDataKeys}
        isLoading={true}
        isError={false}
        timeRange="12h"
        onTimeRangeChange={mockTimeRangeChange}
        summary={mockSummary}
      />
    );

    // Check if title is still rendered during loading
    expect(screen.getByText('Neural Memory Metrics')).toBeInTheDocument();
    
    // Check if skeletons are rendered
    // In a real implementation, add data-testid attributes to the Skeleton components
    const skeleton = document.querySelector('.h-\\[240px\\]');
    expect(skeleton).toBeInTheDocument();
  });

  test('renders error state correctly', () => {
    render(
      <MetricsChart
        title="Neural Memory Metrics"
        data={[]}
        dataKeys={mockDataKeys}
        isLoading={false}
        isError={true}
        error={mockError}
        timeRange="12h"
        onTimeRangeChange={mockTimeRangeChange}
        summary={mockSummary}
      />
    );

    // Check if title is still rendered during error
    expect(screen.getByText('Neural Memory Metrics')).toBeInTheDocument();
    
    // Check if error alert is rendered
    expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
    expect(screen.getByText('Failed to fetch metrics')).toBeInTheDocument();
  });

  test('handles time range changes correctly', () => {
    render(
      <MetricsChart
        title="Neural Memory Metrics"
        data={mockData}
        dataKeys={mockDataKeys}
        isLoading={false}
        isError={false}
        timeRange="12h"
        onTimeRangeChange={mockTimeRangeChange}
        summary={mockSummary}
      />
    );

    // Click on a different time range button
    fireEvent.click(screen.getByText('24h'));
    
    // Check if the onTimeRangeChange callback was called with the right value
    expect(mockTimeRangeChange).toHaveBeenCalledWith('24h');
  });
});

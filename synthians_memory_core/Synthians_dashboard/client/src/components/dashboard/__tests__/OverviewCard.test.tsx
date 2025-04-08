import React from 'react';
import { render, screen } from '@testing-library/react';
import { OverviewCard } from '../OverviewCard';
import { ServiceStatus } from '@shared/schema';

describe('OverviewCard', () => {
  const mockService: ServiceStatus = {
    name: 'Test Service',
    status: 'Healthy',
    uptime: '1d 2h 30m',
    version: '1.0.0',
    url: '/api/test-service/health'
  };

  const mockMetrics = {
    'Test Metric 1': '42',
    'Test Metric 2': '99.9%'
  };

  const mockError = new Error('Test error message');

  test('renders correctly with data', () => {
    render(
      <OverviewCard
        title="Test Service"
        icon="database"
        service={mockService}
        metrics={mockMetrics}
        isLoading={false}
        isError={false}
      />
    );

    // Check if title is rendered
    expect(screen.getByText('Test Service')).toBeInTheDocument();
    
    // Check if metrics are rendered
    expect(screen.getByText('Test Metric 1')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('Test Metric 2')).toBeInTheDocument();
    expect(screen.getByText('99.9%')).toBeInTheDocument();
    
    // Check if service info is rendered
    expect(screen.getByText('Uptime: 1d 2h 30m')).toBeInTheDocument();
    expect(screen.getByText('Version: 1.0.0')).toBeInTheDocument();
  });

  test('renders loading state correctly', () => {
    render(
      <OverviewCard
        title="Test Service"
        icon="database"
        service={null}
        metrics={null}
        isLoading={true}
        isError={false}
      />
    );

    // Check if title is still rendered during loading
    expect(screen.getByText('Test Service')).toBeInTheDocument();
    
    // Check if skeletons are rendered
    // Note: This is a simplistic check since we can't easily target skeletons by test ID
    // In a real implementation, you might want to add data-testid attributes to the Skeleton components
    const skeletons = document.querySelectorAll('.h-24');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  test('renders error state correctly', () => {
    render(
      <OverviewCard
        title="Test Service"
        icon="database"
        service={null}
        metrics={null}
        isLoading={false}
        isError={true}
        error={mockError}
      />
    );

    // Check if title is still rendered during error
    expect(screen.getByText('Test Service')).toBeInTheDocument();
    
    // Check if error alert is rendered
    expect(screen.getByText('Failed to load data')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  test('renders empty state when no metrics available', () => {
    render(
      <OverviewCard
        title="Test Service"
        icon="database"
        service={mockService}
        metrics={null}
        isLoading={false}
        isError={false}
      />
    );

    // Check if title is rendered
    expect(screen.getByText('Test Service')).toBeInTheDocument();
    
    // Check if empty state message is rendered
    expect(screen.getByText('No metrics available for this service')).toBeInTheDocument();
    
    // Service info should still be rendered
    expect(screen.getByText('Uptime: 1d 2h 30m')).toBeInTheDocument();
    expect(screen.getByText('Version: 1.0.0')).toBeInTheDocument();
  });
});

import { test, expect } from '@playwright/test';

/**
 * E2E tests for Synthians Dashboard critical functionality
 * Phase 5.9.3 - Validating core dashboard features are properly rendered
 */

test.describe('Dashboard Critical Paths', () => {
  // Set up - visit the dashboard before each test
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for the dashboard to load - look for the header
    await page.waitForSelector('h1:has-text("Synthians Dashboard")');
  });

  test('Overview page loads with service health cards', async ({ page }) => {
    // Navigate to overview page (should be default)
    await page.getByRole('link', { name: /overview/i }).click();
    
    // Wait for core cards to appear
    await page.waitForSelector('h3:has-text("Memory Core")'); 
    await page.waitForSelector('h3:has-text("Neural Memory")');
    await page.waitForSelector('h3:has-text("Context Cascade Engine")');
    
    // Verify component loading state ends
    const skeletons = await page.$$('.animate-pulse');
    expect(skeletons.length).toBeLessThan(3); // Some might still be loading, but not all
    
    // Verify at least one health status indicator is visible
    const statuses = await page.$$('.bg-green-500, .bg-red-500, .bg-yellow-500');
    expect(statuses.length).toBeGreaterThan(0);

    // Verify metrics chart renders
    const chart = await page.locator('div[data-testid="metrics-chart"]').first();
    expect(await chart.isVisible()).toBeTruthy();
  });

  test('Assemblies page loads with assembly table', async ({ page }) => {
    // Navigate to assemblies page
    await page.getByRole('link', { name: /assemblies/i }).click();
    
    // Verify the page title
    await expect(page.locator('h1')).toContainText('Memory Assemblies');
    
    // Verify search input exists
    const searchInput = await page.getByPlaceholder('Search assemblies...');
    expect(await searchInput.isVisible()).toBeTruthy();
    
    // Verify table headers
    const headers = await page.$$('th');
    expect(headers.length).toBeGreaterThan(3); // Should have multiple columns
    
    // Wait for table rows to load (may take time if fetching data)
    await page.waitForSelector('table tbody tr', { timeout: 10000 });
    
    // Verify at least one row loads or "No assemblies found" message appears
    const hasRows = await page.$$('table tbody tr');
    const noDataMessage = await page.getByText('No assemblies found');
    
    expect(hasRows.length > 0 || await noDataMessage.isVisible()).toBeTruthy();
  });

  test('Neural Memory page loads diagnostics', async ({ page }) => {
    // Navigate to neural memory page
    await page.getByRole('link', { name: /neural memory/i }).click();
    
    // Verify the page title
    await expect(page.locator('h1')).toContainText('Neural Memory');
    
    // Verify health status card renders
    await page.waitForSelector('div:has-text("Health Status")');
    
    // Verify diagnostics chart renders
    const diagnosticsChart = await page.locator('div[data-testid="nm-diagnostics-chart"]').first();
    await expect(diagnosticsChart).toBeVisible({ timeout: 10000 });
  });

  test('Logs page shows placeholder for Phase 5.9.3', async ({ page }) => {
    // Navigate to logs page
    await page.getByRole('link', { name: /logs/i }).click();
    
    // Verify the page title
    await expect(page.locator('h1')).toContainText('System Logs');
    
    // Verify the Phase 5.9.3 placeholder message
    await expect(page.getByText(/WebSocket integration is planned for Phase 5\.9\.3/)).toBeVisible();
  });

  test('Config page loads service configurations', async ({ page }) => {
    // Navigate to config page
    await page.getByRole('link', { name: /config/i }).click();
    
    // Verify the page title
    await expect(page.locator('h1')).toContainText('Configuration');
    
    // Verify service tabs exist
    await expect(page.getByRole('tab', { name: /memory core/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /neural memory/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /context cascade/i })).toBeVisible();
    
    // Verify config loads or shows error
    const configJson = await page.locator('pre');
    const errorAlert = await page.getByText('Failed to load configuration');
    
    // Either config JSON or error message should be visible
    expect(
      await configJson.isVisible() || await errorAlert.isVisible()
    ).toBeTruthy();
  });
});

// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
import '@testing-library/jest-dom';

// Extend Jest matchers with @testing-library/jest-dom
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeInTheDocument(): R;
      toHaveTextContent(content: string | RegExp): R;
      toBeVisible(): R;
      toBeDisabled(): R;
      toBeEnabled(): R;
      toHaveAttribute(attr: string, value?: string): R;
      toHaveClass(className: string): R;
      toHaveStyle(style: Record<string, any>): R;
      toHaveValue(value: any): R;
      toBeChecked(): R;
      toBeEmpty(): R;
      toBeInvalid(): R;
      toBeRequired(): R;
      toBeValid(): R;
      toContainElement(element: HTMLElement | null): R;
      toContainHTML(html: string): R;
      toHaveFocus(): R;
    }
  }
}

// Mock the matchMedia function for tests
// This is needed for components that use media queries
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // Deprecated
    removeListener: jest.fn(), // Deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = class MockIntersectionObserver implements globalThis.IntersectionObserver {
  readonly root: Element | Document | null = null;
  readonly rootMargin: string = '0px';
  readonly thresholds: readonly number[] = [0];

  constructor(private callback: IntersectionObserverCallback, private options?: IntersectionObserverInit) {
    // The callback is stored internally but not exposed publicly in the real API
  }

  disconnect(): void {
    // Mock implementation: Do nothing
  }

  observe(target: Element): void {
    // Mock implementation: Optionally trigger callback immediately for basic tests
    // For more complex scenarios, you might need a more sophisticated mock
    // this.callback([], this);
  }

  takeRecords(): IntersectionObserverEntry[] {
    // Mock implementation: Return empty array
    return [];
  }

  unobserve(target: Element): void {
    // Mock implementation: Do nothing
  }
};

// Set up process.env for tests
process.env.NEXT_PUBLIC_ENABLE_EXPLAINABILITY = 'true';

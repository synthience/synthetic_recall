import React, { useLayoutEffect, useRef, useState } from 'react';

interface ChartWrapperProps {
  width?: number | string;
  height?: number | string;
  children: React.ReactNode;
}

/**
 * A wrapper component that properly handles SVG rendering issues
 * by ensuring SVG elements are properly mounted and sized
 */
export function ChartWrapper({ 
  width = '100%', 
  height = '100%', 
  children 
}: ChartWrapperProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);

  useLayoutEffect(() => {
    // Force a re-render after the component is mounted
    // This helps ensure SVG elements render properly
    setMounted(true);
  }, []);

  return (
    <div 
      ref={containerRef}
      className="chart-wrapper" 
      style={{ 
        width, 
        height,
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      {mounted && children}
    </div>
  );
}

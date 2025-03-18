// src/components/memory/MemoryMetrics.tsx
import React, { useEffect, useRef } from 'react';
import { ConfigurationPanelItem } from '@/components/config/ConfigurationPanelItem';
import { createGlitchEffect } from '@/lib/animations';

type MemoryMetricsProps = {
  accentColor: string;
  quickrecal_score?: number[];
  surprise?: number[];
};

const MemoryMetrics: React.FC<MemoryMetricsProps> = ({ 
  accentColor,
  quickrecal_score = [0.7, 0.5, 0.6, 0.4, 0.8],
  surprise = [0.4, 0.2, 0.7, 0.3, 0.5]
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Apply cyberpunk effects and initialize canvas
  useEffect(() => {
    if (containerRef.current) {
      const cleanupGlitch = createGlitchEffect(containerRef.current, 0.2);
      
      return () => {
        cleanupGlitch();
      };
    }
  }, []);

  // Initialize and animate the neural network visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const setCanvasSize = () => {
      if (canvas.parentElement) {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 120;
      }
    };

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    // Create neural network nodes
    const nodeCount = 20;
    const nodes = Array.from({ length: nodeCount }, (_, i) => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      radius: Math.random() * 2 + 1,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      connections: [] as number[],
      synapseStrength: i < 5 ? quickrecal_score[i] : Math.random() * 0.8 + 0.2, // Use provided quickrecal_score for first 5 nodes
      quickrecal_score: i < 5 ? quickrecal_score[i] : Math.random(),
      surprise: i < 5 ? surprise[i] : Math.random()
    }));

    // Create neural connections
    nodes.forEach((node, i) => {
      const connectionCount = Math.floor(Math.random() * 3) + 1;
      for (let j = 0; j < connectionCount; j++) {
        const target = Math.floor(Math.random() * nodeCount);
        if (target !== i && !node.connections.includes(target)) {
          node.connections.push(target);
        }
      }
    });

    // Animation variables
    let pulse = 0;
    let pulseDirection = 0.01;

    // Animation function
    const animate = () => {
      if (!ctx || !canvas) return;

      // Clear canvas with semi-transparent background for trail effect
      ctx.fillStyle = 'rgba(0, 0, 31, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update pulse
      pulse += pulseDirection;
      if (pulse >= 1 || pulse <= 0) {
        pulseDirection *= -1;
      }

      // Draw connections
      nodes.forEach((node, i) => {
        node.connections.forEach(targetIndex => {
          const target = nodes[targetIndex];
          const dx = target.x - node.x;
          const dy = target.y - node.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          // Skip if too far
          if (distance > canvas.width / 3) return;

          // Calculate connection strength based on synapse strength and distance
          const strength = Math.max(0, 1 - distance / (canvas.width / 3));
          const synapseActivity = (node.synapseStrength + target.synapseStrength) / 2;
          
          // Draw connection line with color based on quickrecal_score or surprise
          const quickrecalScoreColor = `rgba(255, 191, 0, ${strength * synapseActivity * 0.5})`;
          const surpriseColor = `rgba(255, 0, 255, ${strength * synapseActivity * 0.5})`;
          
          // Alternate between quickrecal_score and surprise colors
          const useQuickrecalScore = (i + targetIndex) % 2 === 0;
          
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(target.x, target.y);
          ctx.strokeStyle = useQuickrecalScore ? quickrecalScoreColor : surpriseColor;
          ctx.lineWidth = strength * 2;
          ctx.stroke();

          // Draw pulse effect along the connection
          const pulsePosition = pulse;
          const pulsePosX = node.x + dx * pulsePosition;
          const pulsePosY = node.y + dy * pulsePosition;
          
          ctx.beginPath();
          ctx.arc(pulsePosX, pulsePosY, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = useQuickrecalScore ? 'rgba(255, 191, 0, 0.8)' : 'rgba(255, 0, 255, 0.8)';
          ctx.fill();
        });
      });

      // Update node positions
      nodes.forEach(node => {
        node.x += node.vx;
        node.y += node.vy;

        // Bounce off edges
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        
        // Color based on quickrecal_score/surprise blend
        const r = Math.floor(0 + node.surprise * 255);
        const g = Math.floor(node.quickrecal_score * 191);
        const b = Math.floor(node.surprise * 255);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
        ctx.fill();
      });

      requestAnimationFrame(animate);
    };

    // Start animation
    const animationId = requestAnimationFrame(animate);

    // Cleanup
    return () => {
      window.removeEventListener('resize', setCanvasSize);
      cancelAnimationFrame(animationId);
    };
  }, [quickrecal_score, surprise]);

  return (
    <ConfigurationPanelItem title="Neural Activity Visualization">
      <div 
        ref={containerRef}
        className={`w-full h-[150px] border border-${accentColor}-800/30 bg-black/40 rounded-sm p-2 relative overflow-hidden`}
      >
        <canvas 
          ref={canvasRef} 
          className="w-full h-full"
        />
        
        <div className="absolute bottom-2 right-2 text-xs font-mono text-gray-500">
          Memory Synapse Activity
        </div>
        
        {/* Legend */}
        <div className="absolute top-2 right-2 flex flex-col gap-1">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-amber-500"></div>
            <span className="text-xs text-gray-400">Quickrecal Score</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-fuchsia-500"></div>
            <span className="text-xs text-gray-400">Surprise</span>
          </div>
        </div>
        
        {/* Scan line effect */}
        <div className="absolute inset-0 scan-line pointer-events-none"></div>
      </div>
    </ConfigurationPanelItem>
  );
};

export default MemoryMetrics;
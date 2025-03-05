// src/lib/animations.ts
/**
 * Animation utilities for cyberpunk UI effects
 */

/**
 * Creates a neon glow trail effect that follows cursor movement on the element
 */
export const addGlowTrail = (element: HTMLElement) => {
  element.addEventListener('mousemove', (e) => {
    const rect = element.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const glow = document.createElement('div');
    glow.className = 'glow-trail';
    glow.style.position = 'absolute';
    glow.style.width = '8px';
    glow.style.height = '8px';
    glow.style.background = 'rgba(0, 255, 255, 0.5)';
    glow.style.borderRadius = '50%';
    glow.style.pointerEvents = 'none';
    glow.style.zIndex = '10';
    glow.style.left = `${x}px`;
    glow.style.top = `${y}px`;
    glow.style.transform = 'translate(-50%, -50%)';
    
    element.appendChild(glow);
    
    // Animate the glow trail
    setTimeout(() => {
      glow.style.transition = 'all 500ms ease-out';
      glow.style.opacity = '0';
      glow.style.width = '16px';
      glow.style.height = '16px';
    }, 10);
    
    // Remove the glow trail after animation
    setTimeout(() => {
      if (element.contains(glow)) {
        element.removeChild(glow);
      }
    }, 500);
  });
};

/**
 * Creates a distortion pulse effect when the element is clicked
 */
export const addDistortionPulse = (element: HTMLElement) => {
  element.addEventListener('click', () => {
    // Store original box-shadow
    const originalShadow = element.style.boxShadow;
    
    // Apply distortion effect
    element.style.transform = 'scale(0.95)';
    element.style.boxShadow = '0 0 15px rgba(0, 255, 255, 0.7)';
    
    // Add glitch effect
    const glitch = document.createElement('div');
    glitch.style.position = 'absolute';
    glitch.style.inset = '0';
    glitch.style.backgroundColor = 'rgba(0, 255, 255, 0.1)';
    glitch.style.zIndex = '1';
    glitch.style.pointerEvents = 'none';
    
    element.appendChild(glitch);
    
    // Reset after animation
    setTimeout(() => {
      element.style.transform = 'scale(1)';
      element.style.boxShadow = originalShadow;
      if (element.contains(glitch)) {
        element.removeChild(glitch);
      }
    }, 150);
  });
};

/**
 * Creates a cursor glow effect that follows the mouse
 */
export const createCursorGlow = () => {
  // Create cursor glow element
  const glow = document.createElement('div');
  glow.className = 'cursor-glow';
  glow.style.position = 'fixed';
  glow.style.width = '24px';
  glow.style.height = '24px';
  glow.style.borderRadius = '50%';
  glow.style.background = 'radial-gradient(circle, rgba(0,255,255,0.2) 0%, rgba(0,255,255,0) 70%)';
  glow.style.pointerEvents = 'none';
  glow.style.zIndex = '9999';
  glow.style.transform = 'translate(-50%, -50%)';
  
  document.body.appendChild(glow);
  
  // Update cursor glow position
  const updatePosition = (e: MouseEvent) => {
    glow.style.left = `${e.clientX}px`;
    glow.style.top = `${e.clientY}px`;
  };
  
  // Add event listener
  document.addEventListener('mousemove', updatePosition);
  
  // Return cleanup function
  return () => {
    document.removeEventListener('mousemove', updatePosition);
    if (document.body.contains(glow)) {
      document.body.removeChild(glow);
    }
  };
};

/**
 * Creates a flickering text effect
 */
export const createTextFlicker = (element: HTMLElement) => {
  const text = element.textContent || '';
  element.textContent = '';
  
  // Create wrapper for animation
  const wrapper = document.createElement('span');
  wrapper.style.position = 'relative';
  
  // Split text into individual spans for letter-by-letter animation
  [...text].forEach(char => {
    const span = document.createElement('span');
    span.textContent = char;
    span.style.display = 'inline-block';
    
    // Random flicker animation
    if (Math.random() > 0.7) {
      span.style.animation = `flicker ${Math.random() * 5 + 2}s infinite`;
    }
    
    wrapper.appendChild(span);
  });
  
  element.appendChild(wrapper);
};

/**
 * Generates a neural pathway particle effect for the element
 */
export const createNeuralParticles = (element: HTMLElement, count: number = 20) => {
  const container = document.createElement('div');
  container.className = 'particles-container';
  container.style.position = 'absolute';
  container.style.top = '0';
  container.style.left = '0';
  container.style.width = '100%';
  container.style.height = '100%';
  container.style.overflow = 'hidden';
  container.style.pointerEvents = 'none';
  container.style.zIndex = '0';
  
  element.appendChild(container);
  
  // Create particles
  for (let i = 0; i < count; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.position = 'absolute';
    
    // Random position
    particle.style.left = `${Math.random() * 100}%`;
    particle.style.top = `${Math.random() * 100}%`;
    
    // Random size
    const size = Math.random() * 3 + 1;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    
    // Set style
    particle.style.background = 'rgba(0, 255, 255, 0.5)';
    particle.style.borderRadius = '50%';
    
    // Random animation delay
    particle.style.animationDelay = `${Math.random() * 10}s`;
    
    // Random animation duration
    particle.style.animationDuration = `${Math.random() * 8 + 7}s`;
    
    // Apply animation
    particle.style.animation = 'particleFlow 10s linear infinite';
    
    container.appendChild(particle);
  }
  
  // Return cleanup function
  return () => {
    if (element.contains(container)) {
      element.removeChild(container);
    }
  };
};

/**
 * Creates a scan line effect on the element
 */
export const addScanLineEffect = (element: HTMLElement) => {
  // Add scan line class if not already present
  if (!element.classList.contains('scan-line')) {
    element.classList.add('scan-line');
  }
  
  // Create scan line element if needed
  if (!element.querySelector('.scan-line-element')) {
    const scanLine = document.createElement('div');
    scanLine.className = 'scan-line-element';
    scanLine.style.position = 'absolute';
    scanLine.style.top = '0';
    scanLine.style.left = '0';
    scanLine.style.width = '100%';
    scanLine.style.height = '4px';
    scanLine.style.background = 'linear-gradient(90deg, transparent 0%, rgba(0, 255, 255, 0.2) 50%, transparent 100%)';
    scanLine.style.opacity = '0.5';
    scanLine.style.zIndex = '1';
    scanLine.style.pointerEvents = 'none';
    scanLine.style.animation = 'scanAnimation 3s linear infinite';
    
    element.appendChild(scanLine);
  }
  
  return () => {
    element.classList.remove('scan-line');
    const scanLine = element.querySelector('.scan-line-element');
    if (scanLine) {
      element.removeChild(scanLine);
    }
  };
};

/**
 * Creates a digital glitch effect
 */
export const createGlitchEffect = (element: HTMLElement, intensity: number = 1) => {
  const glitchInterval = setInterval(() => {
    if (Math.random() > 0.95) {
      const glitch = document.createElement('div');
      glitch.style.position = 'absolute';
      glitch.style.top = `${Math.random() * 100}%`;
      glitch.style.left = '0';
      glitch.style.right = '0';
      glitch.style.height = `${Math.random() * 5 + 1}px`;
      glitch.style.backgroundColor = 'rgba(0, 255, 255, 0.5)';
      glitch.style.zIndex = '5';
      glitch.style.transform = `translateX(${(Math.random() - 0.5) * 10}px)`;
      
      element.appendChild(glitch);
      
      setTimeout(() => {
        if (element.contains(glitch)) {
          element.removeChild(glitch);
        }
      }, 300 * intensity);
    }
  }, 2000 / intensity);
  
  return () => {
    clearInterval(glitchInterval);
  };
};
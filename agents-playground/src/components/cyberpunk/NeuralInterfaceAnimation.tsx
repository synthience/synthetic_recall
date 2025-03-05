import React from 'react';

export const NeuralInterfaceAnimation: React.FC = () => {
  return (
    <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" className="w-full h-full" style={{ backgroundColor: 'black' }}>
        {/* Definitions for reusable elements */}
        <defs>
          <linearGradient id="scanline" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(0,255,255,0)" />
            <stop offset="50%" stopColor="rgba(0,255,255,0.1)" />
            <stop offset="100%" stopColor="rgba(0,255,255,0)" />
          </linearGradient>
          
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
          
          <radialGradient id="pulseGradient" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.7)" />
            <stop offset="40%" stopColor="rgba(255,255,255,0.4)" />
            <stop offset="100%" stopColor="rgba(0,255,255,0)" />
            <animate attributeName="r" values="40%;60%;40%" dur="3s" repeatCount="indefinite" />
          </radialGradient>
          
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(0,255,255,0.1)" />
            <stop offset="50%" stopColor="rgba(0,255,255,0.8)" />
            <stop offset="100%" stopColor="rgba(0,255,255,0.1)" />
            <animate attributeName="x1" values="0%;100%;0%" dur="4s" repeatCount="indefinite" />
            <animate attributeName="x2" values="100%;200%;100%" dur="4s" repeatCount="indefinite" />
          </linearGradient>
          
          <symbol id="triangleMarker" viewBox="0 0 20 20">
            <path d="M10,2 L18,18 L2,18 Z" fill="white" />
          </symbol>
          
          <symbol id="circleMarker" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" stroke="white" strokeWidth="1" fill="none" />
            <circle cx="10" cy="10" r="2" fill="white" />
          </symbol>
          
          <symbol id="targetMarker" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" stroke="white" strokeWidth="1" fill="none" />
            <circle cx="10" cy="10" r="1" fill="white" />
            <line x1="10" y1="2" x2="10" y2="6" stroke="white" strokeWidth="1" />
            <line x1="10" y1="14" x2="10" y2="18" stroke="white" strokeWidth="1" />
            <line x1="2" y1="10" x2="6" y2="10" stroke="white" strokeWidth="1" />
            <line x1="14" y1="10" x2="18" y2="10" stroke="white" strokeWidth="1" />
          </symbol>
          
          <symbol id="dotMarker" viewBox="0 0 10 10">
            <circle cx="5" cy="5" r="2" fill="white" />
          </symbol>
        </defs>
        
        {/* Background grid lines */}
        <g id="gridLines" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
          {[...Array(9)].map((_, i) => (
            <line key={`v${i}`} x1={(i + 1) * 100} y1="0" x2={(i + 1) * 100} y2="600" />
          ))}
          {[...Array(5)].map((_, i) => (
            <line key={`h${i}`} x1="0" y1={(i + 1) * 100} x2="1000" y2={(i + 1) * 100} strokeOpacity="0.05" />
          ))}
        </g>
        
        {/* Coordinate lines */}
        <g id="diagonalLines" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
          <line x1="0" y1="150" x2="1000" y2="450" />
          <line x1="250" y1="0" x2="600" y2="600" stroke="url(#lineGradient)" />
          <line x1="900" y1="100" x2="400" y2="500" />
        </g>
        
        {/* Scan line effect - moving down */}
        <rect id="scanLine" x="0" y="0" width="1000" height="20" fill="url(#scanline)" opacity="0.1">
          <animate attributeName="y" values="0;600;0" dur="8s" repeatCount="indefinite" />
        </rect>
        
        {/* Japanese text labels */}
        <g fontFamily="monospace" fontSize="14" fill="rgba(255,255,255,0.6)">
          <text x="330" y="200" id="nonte1">ノンテ</text>
          <text x="550" y="330" id="nonte2">ノンテ</text>
          <text x="750" y="250" id="nonte3">ノンテ</text>
          <text x="180" y="460" id="nonte4">ノンテ</text>
          <text x="600" y="550" id="nonte5">ノンテ</text>
          <text x="850" y="500" id="nonte6">ノンテ</text>
          <text x="765" y="730" id="nonte7">ノンテ</text>
          
          <animate xlinkHref="#nonte2" attributeName="opacity" values="0.6;0.1;0.6" dur="3s" begin="1s" repeatCount="indefinite" />
          <animate xlinkHref="#nonte5" attributeName="opacity" values="0.6;0.1;0.6" dur="4s" begin="0s" repeatCount="indefinite" />
        </g>
        
        {/* Numerical indicators */}
        <g fontFamily="monospace" fontSize="12" fill="rgba(255,255,255,0.8)">
          <text x="30" y="105">761</text>
          <text x="960" y="105">761</text>
          <text x="30" y="482">796</text>
          <text x="960" y="482">796</text>
          <text x="30" y="600">716</text>
          <text x="960" y="600">716</text>
          
          {/* Coordinate numbers */}
          <text x="237" y="27" id="coord1">875-883/029</text>
          <text x="854" y="27" id="coord2">645-380/293</text>
          <text x="167" y="136" id="coord3">SIG-582/581</text>
          <text x="430" y="312" id="coord4">OPL-517/439</text>
          <text x="372" y="429" id="coord5">APP-673/582</text>
          <text x="652" y="440" id="coord6">875-883/029</text>
          
          <animate xlinkHref="#coord4" attributeName="textContent" values="OPL-517/439;OPL-518/440;OPL-517/439" dur="5s" begin="2s" repeatCount="indefinite" />
          <animate xlinkHref="#coord6" attributeName="textContent" values="875-883/029;875-884/030;875-883/029" dur="7s" begin="1s" repeatCount="indefinite" />
        </g>
        
        {/* Status indicators */}
        <g fontFamily="monospace" fontSize="12" fill="rgba(255,255,255,0.9)">
          <text x="866" y="708">STATUS ACTIVE</text>
          <text x="631" y="708">C7</text>
          <text x="585" y="708">1</text>
          
          <circle cx="840" cy="708" r="4" fill="#00ff00">
            <animate attributeName="opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite" />
          </circle>
        </g>
        
        {/* Central radar/tracking circle */}
        <g id="centralRadar" transform="translate(500, 360)">
          <circle cx="0" cy="0" r="100" stroke="white" strokeWidth="1" fill="none" />
          <circle cx="0" cy="0" r="100" stroke="rgba(0,255,255,0.3)" strokeWidth="1" fill="none">
            <animate attributeName="r" values="80;120;80" dur="4s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.3;0.1;0.3" dur="4s" repeatCount="indefinite" />
          </circle>
          
          {/* Radar sweep */}
          <line x1="0" y1="0" x2="0" y2="-100" stroke="rgba(0,255,255,0.6)" strokeWidth="1">
            <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="6s" repeatCount="indefinite" />
          </line>
          
          {/* Target triangles */}
          <use xlinkHref="#triangleMarker" x="-10" y="-75" width="20" height="20">
            <animate attributeName="y" values="-75;-72;-75" dur="3s" repeatCount="indefinite" />
          </use>
          <use xlinkHref="#triangleMarker" x="-35" y="-20" width="20" height="20" transform="scale(0.7)">
            <animate attributeName="y" values="-20;-25;-20" dur="4s" repeatCount="indefinite" />
          </use>
          <use xlinkHref="#triangleMarker" x="20" y="-50" width="20" height="20" transform="scale(0.5)">
            <animate attributeName="y" values="-50;-45;-50" dur="5s" repeatCount="indefinite" />
          </use>
        </g>
        
        {/* Marker circles */}
        <g id="markers">
          <use xlinkHref="#circleMarker" x="166" y="27" width="20" height="20" filter="url(#glow)" />
          <use xlinkHref="#circleMarker" x="817" y="27" width="20" height="20" filter="url(#glow)" />
          <use xlinkHref="#targetMarker" x="167" y="136" width="20" height="20" />
          <use xlinkHref="#circleMarker" x="332" y="165" width="20" height="20" filter="url(#glow)">
            <animate attributeName="width" values="20;24;20" dur="3s" repeatCount="indefinite" />
            <animate attributeName="height" values="20;24;20" dur="3s" repeatCount="indefinite" />
          </use>
          {[...Array(7)].map((_, i) => {
            const positions = [
              { x: 393, y: 312 },
              { x: 322, y: 429 },
              { x: 606, y: 440 },
              { x: 181, y: 508 },
              { x: 322, y: 513 },
              { x: 542, y: 624 },
              { x: 231, y: 746 }
            ];
            return (
              <use
                key={`marker${i}`}
                xlinkHref="#circleMarker"
                x={positions[i].x}
                y={positions[i].y}
                width="16"
                height="16"
              />
            );
          })}
        </g>
        
        {/* Small dot markers */}
        <g id="dots">
          {[
            { x: 170, y: 258 }, { x: 170, y: 307 }, { x: 30, y: 341 },
            { x: 170, y: 402 }, { x: 413, y: 502 }, { x: 538, y: 146 },
            { x: 960, y: 258 }, { x: 960, y: 402 }, { x: 538, y: 682 },
            { x: 688, y: 682 }, { x: 960, y: 960 }
          ].map((pos, i) => (
            <use
              key={`dot${i}`}
              xlinkHref="#dotMarker"
              x={pos.x}
              y={pos.y}
              width="10"
              height="10"
            />
          ))}
          <animate attributeName="opacity" values="1;0.4;1" dur="3s" repeatCount="indefinite" />
        </g>
        
        {/* Connection lines */}
        <g id="connectionLines" stroke="rgba(255,255,255,0.2)" strokeWidth="1">
          <line x1="176" y1="37" x2="332" y2="175" />
          <line x1="332" y1="175" x2="393" y2="312" stroke="url(#lineGradient)" />
          <line x1="322" y1="429" x2="393" y2="312" />
          <line x1="322" y1="429" x2="181" y2="508" />
          <line x1="181" y1="508" x2="322" y2="513" stroke="url(#lineGradient)" />
          <line x1="322" y1="513" x2="542" y2="624" />
          <line x1="393" y1="312" x2="606" y2="440" />
          <line x1="606" y1="440" x2="542" y2="624" stroke="url(#lineGradient)" />
        </g>
        
        {/* Random particle effects */}
        <g id="particles">
          {[...Array(6)].map((_, i) => {
            const x = 200 + i * 200;
            const y = 150 + (i % 3) * 100;
            const duration = 5 + i;
            return (
              <circle key={`particle${i}`} cx={x} cy={y} r="1" fill="white" opacity="0.6">
                <animate
                  attributeName="cy"
                  values={`${y};${y + 20};${y}`}
                  dur={`${duration}s`}
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="opacity"
                  values="0.6;0.2;0.6"
                  dur={`${duration}s`}
                  repeatCount="indefinite"
                />
              </circle>
            );
          })}
        </g>
        
        {/* Glitch effects */}
        <g id="glitchEffects">
          <rect id="glitch1" x="300" y="200" width="100" height="5" fill="rgba(0,255,255,0.3)" opacity="0">
            <animate attributeName="opacity" values="0;0.7;0" dur="0.2s" begin="3s;8s;15s;23s" />
          </rect>
          <rect id="glitch2" x="500" y="350" width="150" height="3" fill="rgba(255,255,255,0.5)" opacity="0">
            <animate attributeName="opacity" values="0;0.5;0" dur="0.1s" begin="5s;12s;18s;25s" />
          </rect>
          <rect id="glitch3" x="200" y="450" width="200" height="4" fill="rgba(255,0,0,0.2)" opacity="0">
            <animate attributeName="opacity" values="0;0.3;0" dur="0.15s" begin="7s;14s;21s;27s" />
          </rect>
        </g>
      </svg>
    </div>
  );
};

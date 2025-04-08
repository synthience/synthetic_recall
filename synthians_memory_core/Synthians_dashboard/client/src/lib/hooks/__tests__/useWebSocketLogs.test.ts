import { renderHook, act } from '@testing-library/react-hooks';
import { useWebSocketLogs } from '../useWebSocketLogs';
import { getWebSocketManager, LogMessage, LogLevel } from '../../websocket';

// Mock the WebSocket manager
jest.mock('../../websocket', () => {
  // Create mock implementation
  const mockOnConnectionStatus = jest.fn();
  const mockOnMessage = jest.fn();
  const mockConnect = jest.fn().mockResolvedValue(undefined);
  const mockDisconnect = jest.fn();
  const mockGetWebSocketManager = jest.fn(() => ({
    onConnectionStatus: mockOnConnectionStatus,
    onMessage: mockOnMessage,
    connect: mockConnect,
    disconnect: mockDisconnect,
    getStatus: jest.fn().mockReturnValue('disconnected')
  }));
  
  // Return all exports with mocked versions
  return {
    getWebSocketManager: mockGetWebSocketManager,
    LogLevel: {
      DEBUG: 'debug',
      INFO: 'info',
      WARNING: 'warning',
      ERROR: 'error'
    }
  };
});

describe('useWebSocketLogs', () => {
  // Mock log messages for testing
  const mockLogs: LogMessage[] = [
    { 
      id: '1', 
      timestamp: '2025-04-05T10:00:00.000Z', 
      service: 'memory-core', 
      level: 'info' as LogLevel, 
      message: 'Memory core service started' 
    },
    { 
      id: '2', 
      timestamp: '2025-04-05T10:01:00.000Z', 
      service: 'neural-memory', 
      level: 'warning' as LogLevel, 
      message: 'High memory usage detected' 
    },
    { 
      id: '3', 
      timestamp: '2025-04-05T10:02:00.000Z', 
      service: 'cce', 
      level: 'error' as LogLevel, 
      message: 'Failed to process request' 
    },
    { 
      id: '4', 
      timestamp: '2025-04-05T10:03:00.000Z', 
      service: 'memory-core', 
      level: 'debug' as LogLevel, 
      message: 'Assembly reconciliation completed' 
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('initializes with default state', () => {
    const { result } = renderHook(() => useWebSocketLogs());
    
    expect(result.current.logs).toEqual([]);
    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.isConnecting).toBe(false);
    expect(result.current.error).toBe(null);
    expect(result.current.filter).toEqual({
      service: 'all',
      level: 'all',
      search: ''
    });
  });

  test('connects to WebSocket when connect is called', async () => {
    const { result } = renderHook(() => useWebSocketLogs());
    
    // Mock the WebSocket connection
    const mockWSManager = getWebSocketManager();
    
    // Call connect
    await act(async () => {
      await result.current.connect();
    });
    
    // Verify the connect method was called
    expect(mockWSManager.connect).toHaveBeenCalled();
    
    // Verify isConnecting was set to true during connection
    expect(result.current.isConnecting).toBe(false); // After connection completes
  });

  test('disconnects from WebSocket when disconnect is called', () => {
    const { result } = renderHook(() => useWebSocketLogs());
    
    // Mock the WebSocket connection
    const mockWSManager = getWebSocketManager();
    
    // Call disconnect
    act(() => {
      result.current.disconnect();
    });
    
    // Verify the disconnect method was called
    expect(mockWSManager.disconnect).toHaveBeenCalled();
  });

  test('adds logs when messages are received', () => {
    const { result } = renderHook(() => useWebSocketLogs());
    
    // Get the message handler
    const mockWSManager = getWebSocketManager();
    const onConnectionStatusHandler = mockWSManager.onConnectionStatus.mock.calls[0][0];
    const onMessageHandler = mockWSManager.onMessage.mock.calls[0][0];
    
    // Simulate connection status change
    act(() => {
      onConnectionStatusHandler('connected');
    });
    
    // Verify connection status was updated
    expect(result.current.connectionStatus).toBe('connected');
    
    // Simulate receiving log messages
    act(() => {
      onMessageHandler(mockLogs[0]);
      onMessageHandler(mockLogs[1]);
    });
    
    // Verify logs were added (newest first)
    expect(result.current.logs).toHaveLength(2);
    expect(result.current.logs[0]).toEqual(mockLogs[1]);
    expect(result.current.logs[1]).toEqual(mockLogs[0]);
  });

  test('filters logs by service', () => {
    const { result } = renderHook(() => useWebSocketLogs());
    
    // Get the message handler
    const mockWSManager = getWebSocketManager();
    const onMessageHandler = mockWSManager.onMessage.mock.calls[0][0];
    
    // Add all mock logs
    act(() => {
      mockLogs.forEach(log => onMessageHandler(log));
    });
    
    // Apply service filter
    act(() => {
      result.current.setFilter({ service: 'memory-core' });
    });
    
    // Verify only memory-core logs are included
    expect(result.current.logs).toHaveLength(2);
    expect(result.current.logs.every(log => log.service === 'memory-core')).toBe(true);
  });

  test('filters logs by level', () => {
    const { result } = renderHook(() => useWebSocketLogs());
    
    // Get the message handler
    const mockWSManager = getWebSocketManager();
    const onMessageHandler = mockWSManager.onMessage.mock.calls[0][0];
    
    // Add all mock logs
    act(() => {
      mockLogs.forEach(log => onMessageHandler(log));
    });
    
    // Apply level filter
    act(() => {
      result.current.setFilter({ level: 'error' });
    });
    
    // Verify only error logs are included
    expect(result.current.logs).toHaveLength(1);
    expect(result.current.logs[0].level).toBe('error');
  });

  test('filters logs by search term', () => {
    const { result } = renderHook(() => useWebSocketLogs());
    
    // Get the message handler
    const mockWSManager = getWebSocketManager();
    const onMessageHandler = mockWSManager.onMessage.mock.calls[0][0];
    
    // Add all mock logs
    act(() => {
      mockLogs.forEach(log => onMessageHandler(log));
    });
    
    // Apply search filter
    act(() => {
      result.current.setFilter({ search: 'assembly' });
    });
    
    // Verify only logs containing 'assembly' are included
    expect(result.current.logs).toHaveLength(1);
    expect(result.current.logs[0].message).toContain('Assembly');
  });

  test('clears logs when clearLogs is called', () => {
    const { result } = renderHook(() => useWebSocketLogs());
    
    // Get the message handler
    const mockWSManager = getWebSocketManager();
    const onMessageHandler = mockWSManager.onMessage.mock.calls[0][0];
    
    // Add all mock logs
    act(() => {
      mockLogs.forEach(log => onMessageHandler(log));
    });
    
    // Verify logs were added
    expect(result.current.logs).toHaveLength(mockLogs.length);
    
    // Clear logs
    act(() => {
      result.current.clearLogs();
    });
    
    // Verify logs were cleared
    expect(result.current.logs).toHaveLength(0);
  });

  test('respects maxLogs limit', () => {
    const maxLogs = 2;
    const { result } = renderHook(() => useWebSocketLogs({ maxLogs }));
    
    // Get the message handler
    const mockWSManager = getWebSocketManager();
    const onMessageHandler = mockWSManager.onMessage.mock.calls[0][0];
    
    // Add more logs than the limit
    act(() => {
      mockLogs.forEach(log => onMessageHandler(log));
    });
    
    // Verify only maxLogs logs are kept (newest first)
    expect(result.current.logs).toHaveLength(maxLogs);
    expect(result.current.logs[0]).toEqual(mockLogs[3]); // Last log (newest)
    expect(result.current.logs[1]).toEqual(mockLogs[2]); // Second to last log
  });

  test('auto-connects when autoConnect is true', () => {
    renderHook(() => useWebSocketLogs({ autoConnect: true }));
    
    // Verify connect was called automatically
    const mockWSManager = getWebSocketManager();
    expect(mockWSManager.connect).toHaveBeenCalled();
  });
});

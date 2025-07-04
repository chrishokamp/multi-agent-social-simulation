import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatStream from './ChatStream';

// Mock EventSource
class MockEventSource {
  constructor(url) {
    this.url = url;
    this.readyState = EventSource.CONNECTING;
    this.onmessage = null;
    this.onerror = null;
    this.onopen = null;
    MockEventSource.instances.push(this);
  }

  close() {
    this.readyState = EventSource.CLOSED;
  }

  // Helper method to simulate receiving a message
  simulateMessage(data) {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }

  // Helper method to simulate an error
  simulateError() {
    this.readyState = EventSource.CLOSED;
    if (this.onerror) {
      this.onerror({ type: 'error' });
    }
  }
}

// Static array to track all instances
MockEventSource.instances = [];

// Replace global EventSource
global.EventSource = MockEventSource;
MockEventSource.CONNECTING = 0;
MockEventSource.OPEN = 1;
MockEventSource.CLOSED = 2;

describe('ChatStream Component', () => {
  beforeEach(() => {
    // Clear all EventSource instances before each test
    MockEventSource.instances = [];
    jest.clearAllMocks();
  });

  afterEach(() => {
    // Close all EventSource instances after each test
    MockEventSource.instances.forEach(instance => instance.close());
  });

  test('renders initial loading state', () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    expect(screen.getByText('Live Conversation')).toBeInTheDocument();
    expect(screen.getByText('Connecting to simulation...')).toBeInTheDocument();
  });

  test('connects to EventSource with correct URL', () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    expect(MockEventSource.instances).toHaveLength(1);
    expect(MockEventSource.instances[0].url).toBe('http://localhost:5000/sim/stream?id=test-sim-123');
  });

  test('handles connected event and updates status', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    
    act(() => {
      eventSource.simulateMessage({ type: 'connected' });
    });

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument();
    });
  });

  test('displays messages in real-time as they arrive', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    
    // Simulate connection
    act(() => {
      eventSource.simulateMessage({ type: 'connected' });
    });

    // Simulate first message
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Buyer',
        content: 'I want to buy a bike',
        timestamp: new Date().toISOString(),
        run_id: 'run-1'
      });
    });

    await waitFor(() => {
      expect(screen.getByText('I want to buy a bike')).toBeInTheDocument();
      expect(screen.getByText('Buyer')).toBeInTheDocument();
    });

    // Simulate second message
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Seller',
        content: 'I have a bike for sale',
        timestamp: new Date().toISOString(),
        run_id: 'run-1'
      });
    });

    await waitFor(() => {
      expect(screen.getByText('I have a bike for sale')).toBeInTheDocument();
      expect(screen.getByText('Seller')).toBeInTheDocument();
    });
  });

  test('handles multiple runs with separators', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    
    // First run message
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Buyer',
        content: 'Run 1 message',
        timestamp: new Date().toISOString(),
        run_id: 'run-1'
      });
    });

    // Second run message
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Seller',
        content: 'Run 2 message',
        timestamp: new Date().toISOString(),
        run_id: 'run-2'
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Run 1 message')).toBeInTheDocument();
      expect(screen.getByText('Run 2 message')).toBeInTheDocument();
      expect(screen.getByText('Run 2')).toBeInTheDocument(); // Run separator
    });
  });

  test('pause functionality stops processing new messages', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    const user = userEvent.setup();
    
    // First add a message to verify messages work
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Buyer',
        content: 'Initial message',
        timestamp: new Date().toISOString()
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Initial message')).toBeInTheDocument();
    });
    
    // Get pause button
    const pauseButton = screen.getByTitle('Pause');
    
    // Click pause and wait for state update
    await act(async () => {
      await user.click(pauseButton);
    });
    
    // Verify button changed to Resume
    expect(screen.getByTitle('Resume')).toBeInTheDocument();
    
    // Send message while paused
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Buyer',
        content: 'This should not appear',
        timestamp: new Date().toISOString()
      });
    });

    // Wait a bit to ensure message processing would have occurred
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Message should not appear
    expect(screen.queryByText('This should not appear')).not.toBeInTheDocument();
    
    // Resume
    const resumeButton = screen.getByTitle('Resume');
    await act(async () => {
      await user.click(resumeButton);
    });
    
    // Send another message
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Buyer',
        content: 'This should appear',
        timestamp: new Date().toISOString()
      });
    });

    // Message should now appear
    await waitFor(() => {
      expect(screen.getByText('This should appear')).toBeInTheDocument();
    });
  });

  test('handles status events correctly', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    
    // Test various status events
    act(() => {
      eventSource.simulateMessage({ 
        type: 'status', 
        status: 'pending',
        message: 'Simulation is starting'
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Simulation is starting. Messages will appear shortly...')).toBeInTheDocument();
    });

    act(() => {
      eventSource.simulateMessage({ 
        type: 'status', 
        status: 'live_streaming',
        mode: 'live'
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Connected to live simulation. Messages will appear as agents interact.')).toBeInTheDocument();
    });
  });

  test('handles complete event', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    
    act(() => {
      eventSource.simulateMessage({ type: 'connected' });
    });

    act(() => {
      eventSource.simulateMessage({ type: 'complete' });
    });

    await waitFor(() => {
      expect(screen.getByText('Complete')).toBeInTheDocument();
    });
  });

  test('handles error events', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    
    act(() => {
      eventSource.simulateMessage({ 
        type: 'error', 
        message: 'Failed to connect to simulation' 
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Failed to connect to simulation')).toBeInTheDocument();
    });
  });

  test('reconnects after connection error', async () => {
    jest.useFakeTimers();
    
    render(<ChatStream simulationId="test-sim-123" />);
    
    const firstEventSource = MockEventSource.instances[0];
    
    // Simulate connection error
    act(() => {
      firstEventSource.simulateError();
    });

    // Fast-forward 5 seconds to trigger reconnection
    act(() => {
      jest.advanceTimersByTime(5000);
    });

    await waitFor(() => {
      // Should create a new EventSource instance
      expect(MockEventSource.instances).toHaveLength(2);
    });

    jest.useRealTimers();
  });

  test('does not reconnect if simulation is complete', async () => {
    jest.useFakeTimers();
    
    render(<ChatStream simulationId="test-sim-123" isSimulationComplete={true} />);
    
    const eventSource = MockEventSource.instances[0];
    
    // Simulate completion
    act(() => {
      eventSource.simulateMessage({ type: 'complete' });
    });

    // Simulate connection error
    act(() => {
      eventSource.simulateError();
    });

    // Fast-forward 5 seconds
    act(() => {
      jest.advanceTimersByTime(5000);
    });

    // Should NOT create a new EventSource instance
    expect(MockEventSource.instances).toHaveLength(1);

    jest.useRealTimers();
  });

  test('handles optimization events', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    
    // Send messages for run 1
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Buyer',
        content: 'Run 1 message',
        timestamp: new Date().toISOString(),
        run_id: 'run-1'
      });
    });

    // Send optimization event
    act(() => {
      eventSource.simulateMessage({
        type: 'optimization',
        agent: 'Buyer',
        run_id: 'run-1',
        optimization_data: {
          improved: true,
          utility_delta: 0.5
        }
      });
    });

    // Send message for run 2 to trigger display of optimization
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Buyer',
        content: 'Run 2 message',
        timestamp: new Date().toISOString(),
        run_id: 'run-2'
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/Agent Optimizations After Run 1/)).toBeInTheDocument();
      expect(screen.getByText(/✅ Improved prompt/)).toBeInTheDocument();
      expect(screen.getByText(/Utility: \+0.50/)).toBeInTheDocument();
    });
  });

  test('restart functionality clears messages', async () => {
    render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    const user = userEvent.setup();
    
    // Add a message
    act(() => {
      eventSource.simulateMessage({
        type: 'message',
        agent: 'Buyer',
        content: 'Test message',
        timestamp: new Date().toISOString()
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Test message')).toBeInTheDocument();
    });

    // Click restart
    const restartButton = screen.getByTitle('Restart');
    await user.click(restartButton);

    // Messages should be cleared
    expect(screen.queryByText('Test message')).not.toBeInTheDocument();
    
    // Should create a new connection
    expect(MockEventSource.instances).toHaveLength(2);
  });

  test('cleans up EventSource on unmount', () => {
    const { unmount } = render(<ChatStream simulationId="test-sim-123" />);
    
    const eventSource = MockEventSource.instances[0];
    const closeSpy = jest.spyOn(eventSource, 'close');
    
    unmount();
    
    expect(closeSpy).toHaveBeenCalled();
  });
});
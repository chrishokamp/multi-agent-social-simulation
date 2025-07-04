import React, { useState, useEffect, useRef, Fragment } from 'react';
import { 
  PlayIcon,
  PauseIcon,
  ArrowPathIcon,
  ChatBubbleLeftRightIcon
} from '@heroicons/react/24/outline';
import './ChatStream.css';

const ChatStream = ({ simulationId, isSimulationComplete }) => {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('loading');
  const [currentRunId, setCurrentRunId] = useState(null);
  const [runIds, setRunIds] = useState(new Set());
  const [optimizationEvents, setOptimizationEvents] = useState({});
  const messagesEndRef = useRef(null);
  const eventSourceRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const isPausedRef = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const connectToStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource(
      `http://localhost:5000/sim/stream?id=${simulationId}`
    );

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'connected':
            setStatus('connected');
            setIsStreaming(true);
            setError(''); // Clear any previous errors
            break;
            
          case 'message':
            if (!isPausedRef.current) {
              const runId = data.run_id || 'default';
              setCurrentRunId(runId);
              setRunIds(prev => new Set([...prev, runId]));
              
              setMessages(prev => [...prev, {
                id: `${data.timestamp}-${data.agent}-${Math.random()}`,
                role: data.agent,
                content: data.content,
                timestamp: data.timestamp,
                runId: runId
              }]);
            }
            break;
            
          case 'optimization':
            // Handle agent optimization events
            if (!isPausedRef.current && data.agent && data.optimization_data) {
              const runId = data.run_id || 'default';
              setOptimizationEvents(prev => ({
                ...prev,
                [runId]: {
                  ...(prev[runId] || {}),
                  [data.agent]: data.optimization_data
                }
              }));
            }
            break;
            
          case 'status':
            setStatus(data.status);
            if (data.mode) {
              console.log(`Streaming mode: ${data.mode}`);
            }
            if (data.message) {
              console.log(`Status message: ${data.message}`);
            }
            break;
            
          case 'debug':
            console.log(`Debug: ${data.message}`);
            break;
            
          case 'complete':
            setStatus('complete');
            setIsStreaming(false);
            eventSource.close();
            break;
            
          case 'error':
            setError(data.message);
            setIsStreaming(false);
            break;
        }
      } catch (err) {
        console.error('Error parsing SSE data:', err);
      }
    };

    eventSource.onerror = (err) => {
      // SSE errors are normal when the connection closes
      if (eventSource.readyState === EventSource.CONNECTING) {
        console.log('SSE connecting...');
        return;
      }
      
      console.log('SSE connection closed');
      
      // Don't show error for pending simulations or normal connection closes
      if (status !== 'pending' && status !== 'no_data' && status !== 'complete') {
        setError('Connection interrupted. Reconnecting...');
      }
      
      setIsStreaming(false);
      
      // Clear any existing reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      // Only retry if simulation is not complete and we haven't received a complete message
      if (!isSimulationComplete && status !== 'complete') {
        reconnectTimeoutRef.current = setTimeout(() => {
          if (!eventSourceRef.current || eventSourceRef.current.readyState === EventSource.CLOSED) {
            console.log('Attempting to reconnect...');
            connectToStream();
          }
        }, 5000); // Increased to 5 seconds to reduce reconnection frequency
      }
    };

    eventSourceRef.current = eventSource;
  };

  useEffect(() => {
    if (simulationId) {
      // Always try to connect, regardless of completion status
      // The streaming endpoint will handle both live and completed simulations
      connectToStream();
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [simulationId]);

  const handlePauseResume = () => {
    const newPausedState = !isPaused;
    setIsPaused(newPausedState);
    isPausedRef.current = newPausedState;
  };

  const handleRestart = () => {
    setMessages([]);
    if (!isSimulationComplete) {
      connectToStream();
    }
  };

  const getAgentColor = (agent) => {
    const colors = {
      'Buyer': 'bg-blue-100 text-blue-900 border-blue-200',
      'Seller': 'bg-green-100 text-green-900 border-green-200',
      'InformationReturnAgent': 'bg-gray-100 text-gray-700 border-gray-200',
    };
    return colors[agent] || 'bg-gray-100 text-gray-900 border-gray-200';
  };

  const getAgentAvatar = (agent) => {
    const avatars = {
      'Buyer': '🛒',
      'Seller': '💰',
      'InformationReturnAgent': '📊',
    };
    return avatars[agent] || '👤';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <ChatBubbleLeftRightIcon className="w-6 h-6 text-white" />
            <h3 className="text-lg font-semibold text-white">Live Conversation</h3>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Status indicator */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  isStreaming ? 'bg-green-400 animate-pulse' : 
                  status === 'complete' ? 'bg-blue-400' : 'bg-gray-300'
                }`} />
                <span className="text-sm text-white">
                  {isStreaming ? 'Live' : status === 'complete' ? 'Complete' : 'Waiting'}
                </span>
              </div>
              
              {/* Run indicator */}
              {runIds.size > 0 && (
                <div className="text-xs text-white/80">
                  Run {Array.from(runIds).indexOf(currentRunId) + 1} of {runIds.size}
                </div>
              )}
            </div>
            
            {/* Controls */}
            {!isSimulationComplete && (
              <div className="flex items-center space-x-1">
                <button
                  onClick={handlePauseResume}
                  className="p-1.5 rounded hover:bg-white/20 transition-colors"
                  title={isPaused ? 'Resume' : 'Pause'}
                >
                  {isPaused ? 
                    <PlayIcon className="w-4 h-4 text-white" /> : 
                    <PauseIcon className="w-4 h-4 text-white" />
                  }
                </button>
                
                <button
                  onClick={handleRestart}
                  className="p-1.5 rounded hover:bg-white/20 transition-colors"
                  title="Restart"
                >
                  <ArrowPathIcon className="w-4 h-4 text-white" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="h-96 overflow-y-auto p-4 bg-gray-50">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <ChatBubbleLeftRightIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <div className="text-gray-500 text-sm">
                {isStreaming ? (
                  <div className="space-y-2">
                    <div>Waiting for messages...</div>
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-400"></div>
                    </div>
                  </div>
                ) : status === 'live_streaming' ? (
                  'Connected to live simulation. Messages will appear as agents interact.'
                ) : status === 'no_data' || status === 'pending' ? (
                  'Simulation is starting. Messages will appear shortly...'
                ) : isSimulationComplete ? (
                  'No conversation data available'
                ) : (
                  'Connecting to simulation...'
                )}
              </div>
              {error && messages.length === 0 && (
                <div className="text-red-500 text-xs mt-2">{error}</div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {messages.map((message, index) => {
              // Check if this is the start of a new run
              const isNewRun = index > 0 && message.runId !== messages[index - 1].runId;
              
              return (
                <React.Fragment key={message.id}>
                  {/* Run divider with optimization info */}
                  {isNewRun && runIds.size > 1 && (
                    <div className="my-4">
                      <div className="flex items-center space-x-2 py-2">
                        <div className="flex-1 h-px bg-gray-300"></div>
                        <span className="text-xs text-gray-500 px-2">
                          Run {Array.from(runIds).indexOf(message.runId) + 1}
                        </span>
                        <div className="flex-1 h-px bg-gray-300"></div>
                      </div>
                      
                      {/* Show optimization events for previous run */}
                      {index > 0 && messages[index - 1].runId && 
                       optimizationEvents[messages[index - 1].runId] && (
                        <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 mt-2 mb-3">
                          <div className="text-xs font-medium text-purple-700 mb-2">
                            🔧 Agent Optimizations After Run {Array.from(runIds).indexOf(messages[index - 1].runId) + 1}
                          </div>
                          {Object.entries(optimizationEvents[messages[index - 1].runId]).map(([agent, data]) => (
                            <div key={agent} className="text-xs text-purple-600 mb-1">
                              <span className="font-medium">{agent}:</span>
                              <span className="ml-1">
                                {data.improved ? '✅ Improved prompt' : '❌ No improvement'}
                                {data.utility_delta && ` (Utility: ${data.utility_delta > 0 ? '+' : ''}${data.utility_delta.toFixed(2)})`}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  
                  <div className="flex items-start space-x-3 animate-fadeIn">
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm flex-shrink-0 ${getAgentColor(message.role)}`}>
                  {getAgentAvatar(message.role)}
                </div>
                
                {/* Message content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline space-x-2 mb-1">
                    <span className="font-medium text-sm text-gray-900">
                      {message.role}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className={`rounded-lg p-3 border ${getAgentColor(message.role)}`}>
                    {message.role === 'InformationReturnAgent' ? (
                      <pre className="text-xs whitespace-pre-wrap font-mono overflow-x-auto">
                        {message.content}
                      </pre>
                    ) : (
                      <p className="text-sm leading-relaxed">{message.content}</p>
                    )}
                  </div>
                </div>
              </div>
              </React.Fragment>
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatStream;
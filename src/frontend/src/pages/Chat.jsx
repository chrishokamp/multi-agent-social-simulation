import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useChat } from 'ai/react';
import Navbar from './components/Navbar';
import { 
  ChevronLeftIcon, 
  ChevronRightIcon,
  PlayIcon,
  PauseIcon,
  ArrowPathIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

const Chat = () => {
  const { simulationId } = useParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('loading');
  const [selectedRun, setSelectedRun] = useState(0);
  const [runs, setRuns] = useState([]);
  const messagesEndRef = useRef(null);
  const eventSourceRef = useRef(null);

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
            break;
            
          case 'message':
            if (!isPaused) {
              setMessages(prev => [...prev, {
                id: `${data.timestamp}-${data.agent}`,
                role: data.agent,
                content: data.content,
                timestamp: data.timestamp,
                runId: data.run_id
              }]);
            }
            break;
            
          case 'status':
            setStatus(data.status);
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
      console.error('SSE error:', err);
      setError('Connection lost. Retrying...');
      setIsStreaming(false);
      
      // Retry connection after 3 seconds
      setTimeout(() => {
        if (!eventSourceRef.current || eventSourceRef.current.readyState === EventSource.CLOSED) {
          connectToStream();
        }
      }, 3000);
    };

    eventSourceRef.current = eventSource;
  };

  useEffect(() => {
    if (simulationId) {
      connectToStream();
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [simulationId]);

  const handlePauseResume = () => {
    setIsPaused(!isPaused);
  };

  const handleRestart = () => {
    setMessages([]);
    connectToStream();
  };

  const getAgentColor = (agent) => {
    const colors = {
      'Buyer': 'bg-blue-100 text-blue-900',
      'Seller': 'bg-green-100 text-green-900',
      'InformationReturnAgent': 'bg-gray-100 text-gray-700',
    };
    return colors[agent] || 'bg-gray-100 text-gray-900';
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
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="bg-white rounded-lg shadow mb-6 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Simulation Chat</h1>
              <p className="text-gray-600 mt-1">ID: {simulationId}</p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Status indicator */}
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  isStreaming ? 'bg-green-500 animate-pulse' : 
                  status === 'complete' ? 'bg-blue-500' : 'bg-gray-400'
                }`} />
                <span className="text-sm text-gray-600">
                  {isStreaming ? 'Streaming' : status}
                </span>
              </div>
              
              {/* Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={handlePauseResume}
                  className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                  title={isPaused ? 'Resume' : 'Pause'}
                >
                  {isPaused ? 
                    <PlayIcon className="w-5 h-5 text-gray-700" /> : 
                    <PauseIcon className="w-5 h-5 text-gray-700" />
                  }
                </button>
                
                <button
                  onClick={handleRestart}
                  className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                  title="Restart"
                >
                  <ArrowPathIcon className="w-5 h-5 text-gray-700" />
                </button>
                
                <button
                  onClick={() => navigate(`/dashboard/${simulationId}`)}
                  className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                  title="View Dashboard"
                >
                  <ChartBarIcon className="w-5 h-5 text-gray-700" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Chat messages */}
        <div className="bg-white rounded-lg shadow">
          <div className="h-[600px] overflow-y-auto p-6">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="text-gray-400 text-lg mb-2">
                    {isStreaming ? 'Waiting for messages...' : 'No messages yet'}
                  </div>
                  {error && (
                    <div className="text-red-500 text-sm mt-2">{error}</div>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message) => (
                  <div key={message.id} className="flex items-start space-x-3">
                    {/* Avatar */}
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg ${getAgentColor(message.role)}`}>
                      {getAgentAvatar(message.role)}
                    </div>
                    
                    {/* Message content */}
                    <div className="flex-1">
                      <div className="flex items-baseline space-x-2 mb-1">
                        <span className="font-semibold text-gray-900">
                          {message.role}
                        </span>
                        <span className="text-xs text-gray-500">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className={`rounded-lg p-3 ${getAgentColor(message.role)}`}>
                        {message.role === 'InformationReturnAgent' ? (
                          <pre className="text-sm whitespace-pre-wrap font-mono">
                            {message.content}
                          </pre>
                        ) : (
                          <p className="text-sm">{message.content}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
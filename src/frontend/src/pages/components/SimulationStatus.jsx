import React, { useState, useEffect } from 'react';
import { 
  ClockIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

const SimulationStatus = ({ simulationId, catalogData, onStatusChange }) => {
  const [status, setStatus] = useState('loading');
  const [progress, setProgress] = useState(0);
  const [expectedRuns, setExpectedRuns] = useState(0);
  const [completedRuns, setCompletedRuns] = useState(0);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    // If catalog data is provided, use it
    if (catalogData) {
      const sim = catalogData.find(s => s.simulation_id === simulationId);
      if (sim) {
        setExpectedRuns(sim.expected_runs || 0);
        setProgress(sim.progress_percentage || 0);
        
        // Determine status based on progress
        if (sim.progress_percentage === 0) {
          setStatus('pending');
        } else if (sim.progress_percentage === 100) {
          setStatus('complete');
          setCompletedRuns(sim.expected_runs);
        } else {
          setStatus('running');
          setCompletedRuns(Math.floor((sim.progress_percentage / 100) * sim.expected_runs));
        }
        
        setLastUpdate(new Date());
      }
    }
  }, [catalogData, simulationId]);

  useEffect(() => {
    if (onStatusChange) {
      onStatusChange({
        status,
        progress,
        expectedRuns,
        completedRuns,
        error
      });
    }
  }, [status, progress, expectedRuns, completedRuns, error, onStatusChange]);

  const getStatusIcon = () => {
    switch (status) {
      case 'pending':
        return <ClockIcon className="w-5 h-5 text-yellow-500" />;
      case 'running':
        return <ArrowPathIcon className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'complete':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'error':
        return <ExclamationCircleIcon className="w-5 h-5 text-red-500" />;
      default:
        return <ClockIcon className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'pending':
        return 'Queued';
      case 'running':
        return `Running (${completedRuns}/${expectedRuns} runs)`;
      case 'complete':
        return `Complete (${completedRuns} runs)`;
      case 'error':
        return 'Error';
      default:
        return 'Loading...';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-50 text-yellow-800 border-yellow-200';
      case 'running':
        return 'bg-blue-50 text-blue-800 border-blue-200';
      case 'complete':
        return 'bg-green-50 text-green-800 border-green-200';
      case 'error':
        return 'bg-red-50 text-red-800 border-red-200';
      default:
        return 'bg-gray-50 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className={`rounded-lg border p-4 ${getStatusColor()}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon()}
          <div>
            <h3 className="font-medium text-sm">{getStatusText()}</h3>
            {error && (
              <p className="text-xs mt-1 opacity-75">{error}</p>
            )}
          </div>
        </div>
        
        {status === 'running' && (
          <div className="flex items-center space-x-2">
            <div className="w-32 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="text-xs font-medium">{progress}%</span>
          </div>
        )}
        
        {lastUpdate && (
          <div className="text-xs opacity-50">
            Updated {lastUpdate.toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
};

export default SimulationStatus;
import { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import apiService from '../services/apiService.js';
import Navbar from './components/Navbar';

import { FaPlus } from 'react-icons/fa';
import { MdDelete } from 'react-icons/md';
import { FaChartColumn } from 'react-icons/fa6';
import { FaImage } from 'react-icons/fa6';

const backendUri = 'http://127.0.0.1:5000/sim';

const StatusBadge = ({ progress }) => {
  let status, bgColor, textColor;

  if (progress === 100) {
    status = 'Completed';
    bgColor = '#10b981'; // Green
    textColor = 'white';
  } else if (progress === 0) {
    status = 'Pending';
    bgColor = '#6b7280'; // Gray
    textColor = 'white';
  } else {
    status = 'Running';
    bgColor = '#3b82f6'; // Blue
    textColor = 'white';
  }

  return (
    <span 
      className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide"
      style={{ 
        background: bgColor,
        color: textColor,
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)'
      }}
    >
      {status}
    </span>
  );
};

const SimulationItem = ({ simulation, onViewRenderer, onViewDashboard, onDelete }) => {
  // Format the creation date if available
  const formatDate = (dateString) => {
    if (!dateString) return null;
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString();
  };
  
  // Calculate estimated completion time based on progress
  const getEstimatedCompletion = (progress) => {
    if (progress === 100) return null;
    if (progress === 0) return 'Waiting to start';

    // Simple estimation - just for demo purposes
    const remainingPercentage = 100 - progress;
    let timeEstimate;

    if (remainingPercentage > 75) {
      timeEstimate = 'About 1 hour';
    } else if (remainingPercentage > 50) {
      timeEstimate = 'About 30 minutes';
    } else if (remainingPercentage > 25) {
      timeEstimate = 'About 15 minutes';
    } else {
      timeEstimate = 'Less than 5 minutes';
    }

    return timeEstimate;
  };

  const estimatedCompletion = getEstimatedCompletion(simulation.progress_percentage);
  const isComplete = simulation.progress_percentage === 100;

  return (
    <div className="p-6 mb-4 rounded-lg border border-gray-200 bg-white shadow-sm hover:shadow-md transition-all">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-xl font-bold mb-1 text-gray-900">{simulation.name}</h2>
          <p className="text-sm mb-2 text-gray-500">ID: {simulation.simulation_id}</p>
          <div className="flex items-center gap-4">
            <StatusBadge progress={simulation.progress_percentage} />
            <span className="text-sm text-gray-600">{simulation.expected_runs} runs total</span>
            {simulation.created_at && (
              <span className="text-sm text-gray-600">• {formatDate(simulation.created_at)}</span>
            )}
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <button
            onClick={() => onViewDashboard(simulation.simulation_id)}
            className="px-4 py-2 rounded transition-all hover:shadow-lg hover:scale-105 font-medium cursor-pointer"
            style={{ 
              background: isComplete 
                ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' // Purple gradient for completed
                : 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', // Pink gradient for in-progress
              color: 'white',
              border: '1px solid transparent',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.15)',
              transform: 'translateZ(0)' // For hardware acceleration
            }}
          >
            <FaChartColumn className="inline-block mb-0.5 mr-2" />
            {isComplete ? 'View Dashboard' : 'View Progress'}
          </button>
          <button
            onClick={() => onDelete(simulation.simulation_id, isComplete)}
            className="px-4 py-2 rounded transition-all hover:shadow-md hover:scale-105 font-medium cursor-pointer"
            style={{
              background: 'transparent',
              color: '#dc2626',
              border: '1px solid #dc2626',
              boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
              transform: 'translateZ(0)' // For hardware acceleration
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#dc2626';
              e.target.style.color = 'white';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'transparent';
              e.target.style.color = '#dc2626';
            }}
          >
            <MdDelete className="h-4 w-4 inline-block mb-0.5 mr-1" />
            Delete
          </button>
        </div>
      </div>
      {simulation.progress_percentage > 0 && simulation.progress_percentage < 100 && (
        <div className="mt-3 text-sm" style={{ color: 'hsl(var(--text-400))' }}>
          <p>Estimated completion: {estimatedCompletion}</p>
          <div className="w-full rounded-full h-3 mt-2 bg-gray-200">
            <div
              className="h-3 rounded-full transition-all duration-500 ease-out"
              style={{ 
                background: 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)',
                width: `${simulation.progress_percentage}%`,
                boxShadow: '0 1px 3px rgba(59, 130, 246, 0.4)'
              }}
            ></div>
          </div>
          <p className="mt-1 text-right text-xs" style={{ color: 'hsl(var(--text-300))' }}>{simulation.progress_percentage}% complete</p>
        </div>
      )}
    </div>
  );
};

const SimulationsList = () => {
  const navigate = useNavigate();
  const [simulations, setSimulations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const scrollPosition = useRef(0);

  // Fetch simulations from API
  const fetchSimulations = async () => {
    setLoading(true);
    try {
      scrollPosition.current = window.pageYOffset; // Preserve scroll position

      const data = await apiService.getSimulationsCatalog();
      setSimulations(data);
      setLoading(false);
    } catch (err) {
      setError(`Failed to fetch simulations: ${err.message}`);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSimulations();

    // Set up polling to refresh data periodically
    const intervalId = setInterval(fetchSimulations, 30000); // Refresh every 30 seconds

    return () => clearInterval(intervalId); // Clean up on unmount
  }, []);

  useEffect(() => {
    // Restore scroll position when simulations are updated
    window.scrollTo({
      top: scrollPosition.current,
      left: 0,
      behavior: 'instant',
    });
  }, [simulations]);

  const handleViewRenderer = (simulationId) => {
    // Navigate to the renderer view for this simulation
    navigate(`/renderer/${simulationId}`);
  };

  const handleViewDashboard = (simulationId) => {
    // Navigate to the dashboard view for this simulation
    navigate(`/dashboard/${simulationId}`);
  };

  const handleDelete = (simulationId, isComplete) => {
    // Delete results and delete catalog share request formats
    const request = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: JSON.stringify({ id: simulationId }),
    };

    // Delete from result if it exists
    if (isComplete)
      fetch(`${backendUri}/del_results`, request)
        .then((response) => response.json())
        .then((json) => {
          console.log(json);
        });

    // Delete from catalog
    fetch(`${backendUri}/del_catalog`, request)
      .then((response) => response.json())
      .then((json) => {
        console.log(json);
        fetchSimulations();
      });
  };

  return (
    <div className="flex justify-center py-8 h-auto mt-40">
      <Navbar />
      <div className="w-full max-w-3xl px-4 container rounded mt-20 pt-10">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl p-2 font-bold">Simulations</h1>
          <div className="flex">
            <Link
              to="/Configuration"
              className="px-4 py-2 rounded transition-colors"
            >
              <FaPlus className="inline-block mr-2 mb-0.75" />
              Create New Simulation
            </Link>
          </div>
        </div>
        {error && (
          <div className="danger p-2 rounded">
            {error}
          </div>
        )}
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div style={{ color: 'hsl(var(--oncolor-100))' }}>Loading simulations...</div>
          </div>
        ) : (
          <>
            {simulations.length === 0 ? (
              <div className="rounded p-8 text-center">
                <p className="mb-4" style={{ color: 'hsl(var(--text-400))' }}>No simulations found</p>
                <button
                  onClick={() => navigate('/Configuration')}
                >
                  Create Your First Simulation
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                {simulations.map((sim) => (
                  <SimulationItem
                    key={sim.simulation_id}
                    simulation={sim}
                    onViewRenderer={handleViewRenderer}
                    onViewDashboard={handleViewDashboard}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default SimulationsList;

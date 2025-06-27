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
  let status, bgStyle;

  if (progress === 100) {
    status = 'Completed';
    bgStyle = { background: 'hsl(var(--accent-main-200))' };
  } else if (progress === 0) {
    status = 'Pending';
    bgStyle = { background: 'hsl(var(--accent-secondary-100))' };
  } else {
    status = 'Running';
    bgStyle = { background: 'hsl(var(--accent-pro-100))' };
  }

  return (
    <span className="px-2 py-1 rounded-full text-xs font-semibold" style={{ color: 'hsl(var(--oncolor-100))', ...bgStyle }}>
      {status}
    </span>
  );
};

const SimulationItem = ({ simulation, onViewRenderer, onViewDashboard, onDelete }) => {
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
    <div className="p-4 mb-3 rounded-lg  border border-light transition-colors">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-lg font-semibold mb-1" style={{ color: 'hsl(var(--text-000))' }}>{simulation.name}</h2>
          <p className="text-sm mb-2" style={{ color: 'hsl(var(--text-200))' }}>ID: {simulation.simulation_id}</p>
          <div className="flex items-center gap-4">
            <StatusBadge progress={simulation.progress_percentage} />
            <span className="text-sm" style={{ color: 'hsl(var(--text-400))' }}>{simulation.expected_runs} runs total</span>
          </div>
        </div>
        <div className="flex flex-col gap-2">
          {isComplete && (
            <>
              <button
                onClick={() => onViewRenderer(simulation.simulation_id)}
                className="px-3 py-1 rounded transition-colors cursor-pointer"
                style={{ background: 'hsl(var(--accent-pro-000))', color: 'hsl(var(--oncolor-100))' }}
              >
                <FaImage className="inline-block mb-0.75 mr-2" />
                View Render
              </button>
              <button
                onClick={() => onViewDashboard(simulation.simulation_id)}
                className="px-3 py-1 rounded transition-colors cursor-pointer"
                style={{ background: 'hsl(var(--accent-pro-200))', color: 'hsl(var(--oncolor-100))' }}
              >
                <FaChartColumn className="inline-block mb-0.75 mr-2" />
                View Dashboard
              </button>
            </>
          )}
          <button
            onClick={() => onDelete(simulation.simulation_id, isComplete)}
            className="px-3 py-1 rounded transition-colors cursor-pointer"
            style={{ background: 'hsl(var(--danger-100))', color: 'hsl(var(--oncolor-100))' }}
          >
            <MdDelete className="h-5 w-5 inline-block mb-0.75 mr-1" />
            Delete
          </button>
        </div>
      </div>
      {simulation.progress_percentage > 0 && simulation.progress_percentage < 100 && (
        <div className="mt-3 text-sm" style={{ color: 'hsl(var(--text-400))' }}>
          <p>Estimated completion: {estimatedCompletion}</p>
          <div className="w-full rounded-full h-2 mt-1" style={{ background: 'hsl(var(--border-200))' }}>
            <div
              className="h-2 rounded-full"
              style={{ background: 'hsl(var(--accent-pro-100))', width: `${simulation.progress_percentage}%` }}
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
    <div className="flex justify-center min-h-screen py-8">
      <Navbar />
      <div className="w-full max-w-4xl px-4">
        <div className="flex justify-between items-center mt-18 mb-6 pt-40">
          <h1 className="text-2xl font-bold" style={{ color: 'hsl(var(--text-000))' }}>Simulations</h1>
          <div className="flex space-x-4">
            <Link
              to="/Configuration"
              className="px-4 py-2 rounded-lg transition-colors"
            >
              <FaPlus className="inline-block mr-2 mb-0.75" />
              Create New Simulation
            </Link>
          </div>
        </div>
        {error && (
          <div className="p-3 mb-4 rounded-lg" style={{ background: 'hsl(var(--danger-000))', border: '1px solid hsl(var(--danger-100))', color: 'hsl(var(--oncolor-100))' }}>
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
              <div className="rounded-lg p-8 text-center" style={{ background: 'hsl(var(--bg-200))' }}>
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

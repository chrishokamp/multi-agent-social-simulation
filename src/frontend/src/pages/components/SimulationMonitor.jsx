import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import SimulationStatus from './SimulationStatus';
import ChatStream from './ChatStream';
import api from '../../services/apiService';

const SimulationMonitor = () => {
  const { simulationId } = useParams();
  const [simulationData, setSimulationData] = useState(null);
  const [catalogData, setCatalogData] = useState(null);
  const [simulationStatus, setSimulationStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pollInterval, setPollInterval] = useState(null);

  // Fetch simulation results
  const fetchSimulationData = useCallback(async () => {
    try {
      const data = await api.getSimulationOutput(simulationId);
      setSimulationData(data);
      
      // If we have runs, we might be complete
      if (data.runs && data.runs.length > 0) {
        // Check if we should stop polling
        if (simulationStatus?.status === 'complete') {
          if (pollInterval) {
            clearInterval(pollInterval);
            setPollInterval(null);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching simulation data:', err);
      setError('Failed to load simulation data');
    }
  }, [simulationId, simulationStatus, pollInterval]);

  // Fetch catalog for status
  const fetchCatalog = useCallback(async () => {
    try {
      const catalog = await api.getSimulationsCatalog();
      setCatalogData(catalog);
    } catch (err) {
      console.error('Error fetching catalog:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchSimulationData(),
        fetchCatalog()
      ]);
      setLoading(false);
    };
    
    loadData();
  }, [fetchSimulationData, fetchCatalog]);

  // Set up polling based on status
  useEffect(() => {
    // Clear existing interval
    if (pollInterval) {
      clearInterval(pollInterval);
    }

    // Only poll if simulation is pending or running
    if (simulationStatus?.status === 'pending' || simulationStatus?.status === 'running') {
      const interval = setInterval(() => {
        fetchSimulationData();
        fetchCatalog();
      }, 2000); // Poll every 2 seconds
      
      setPollInterval(interval);
    }

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [simulationStatus?.status]);

  const handleStatusChange = useCallback((status) => {
    setSimulationStatus(status);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading simulation...</p>
        </div>
      </div>
    );
  }

  if (error && !simulationData) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center text-red-600">
          <p className="text-xl mb-2">Error</p>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  const isPending = simulationStatus?.status === 'pending';
  const isRunning = simulationStatus?.status === 'running';
  const isComplete = simulationStatus?.status === 'complete';

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Simulation Dashboard
        </h1>
        <p className="text-gray-600">ID: {simulationId}</p>
      </div>

      {/* Status Card */}
      <div className="mb-6">
        <SimulationStatus 
          simulationId={simulationId}
          catalogData={catalogData}
          onStatusChange={handleStatusChange}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live Chat Stream */}
        <div className="lg:col-span-2">
          <ChatStream 
            simulationId={simulationId}
            isSimulationComplete={isComplete}
          />
        </div>

        {/* Simulation Info - Only show when we have data */}
        {simulationData && simulationData.runs && simulationData.runs.length > 0 && (
          <>
            {/* Run Statistics */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Run Statistics</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Runs:</span>
                  <span className="font-medium">{simulationData.runs.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Messages:</span>
                  <span className="font-medium">
                    {Math.round(
                      simulationData.runs.reduce((sum, run) => sum + (run.num_messages || 0), 0) / 
                      simulationData.runs.length
                    )}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Agents:</span>
                  <span className="font-medium">
                    {Array.from(new Set(
                      simulationData.runs
                        .flatMap(run => run.messages?.map(m => m.agent) || [])
                        .filter(agent => agent !== 'InformationReturnAgent')
                    )).join(', ') || 'None'}
                  </span>
                </div>
              </div>
            </div>

            {/* Output Variables */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Output Variables</h2>
              <div className="space-y-2">
                {simulationData.runs[0]?.output_variables?.map((variable, idx) => (
                  <div key={idx} className="flex justify-between py-2 border-b last:border-0">
                    <span className="text-gray-600">{variable.name}:</span>
                    <span className="font-medium">
                      {typeof variable.value === 'boolean' 
                        ? variable.value ? 'Yes' : 'No'
                        : variable.value}
                    </span>
                  </div>
                )) || (
                  <p className="text-gray-500">No output variables</p>
                )}
              </div>
            </div>
          </>
        )}

        {/* Pending/Running State Info */}
        {(isPending || isRunning) && !simulationData?.runs?.length && (
          <div className="lg:col-span-2 bg-blue-50 rounded-lg p-6 text-center">
            <div className="text-blue-800">
              {isPending && (
                <>
                  <p className="text-lg font-medium mb-2">Simulation Queued</p>
                  <p className="text-sm">Your simulation is waiting to be processed...</p>
                </>
              )}
              {isRunning && (
                <>
                  <p className="text-lg font-medium mb-2">Simulation Running</p>
                  <p className="text-sm">
                    Processing {simulationStatus.completedRuns} of {simulationStatus.expectedRuns} runs...
                  </p>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SimulationMonitor;
import React from 'react';
import { screen, waitFor } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import '@testing-library/jest-dom';
import Dashboard from './Dashboard';
import * as echarts from 'echarts';
import { renderWithRouter } from '../test-utils';

// Mock ECharts
const mockChartInstance = {
  setOption: jest.fn(),
  dispose: jest.fn(),
  resize: jest.fn()
};

jest.mock('echarts', () => ({
  init: jest.fn(() => mockChartInstance),
  getInstanceByDom: jest.fn(() => mockChartInstance)
}));

// Mock ECharts-for-React to use our mock chart instance
jest.mock('echarts-for-react', () => {
  const React = require('react');
  return React.forwardRef((props, ref) => {
    React.useEffect(() => {
      if (props.option) {
        mockChartInstance.setOption(props.option);
      }
    }, [props.option]);
    return React.createElement('div', { ref });
  });
});

// Mock fetch
global.fetch = jest.fn();

// Mock EventSource for ChatStream component
global.EventSource = jest.fn(() => ({
  addEventListener: jest.fn(),
  close: jest.fn(),
  onerror: null,
  onmessage: null,
  onopen: null,
  readyState: 0
}));

// Mock the API service
jest.mock('../services/apiService');
import api from '../services/apiService';

describe('Dashboard Component - Utility Evolution', () => {
  beforeEach(() => {
    fetch.mockClear();
    // Set up default mocks
    api.getSimulationsCatalog = jest.fn();
    api.getSimulationOutput = jest.fn();
    // Clear mock chart instance calls
    mockChartInstance.setOption.mockClear();
    mockChartInstance.dispose.mockClear();
    mockChartInstance.resize.mockClear();
  });

  test('renders utility evolution chart with correct data', async () => {
    // Mock simulation data with utility values
    const mockSimulation = {
      simulation_id: 'sim1',
      config: {
        name: 'Test Simulation',
        agents: [
          { name: 'Buyer', utility_class: 'BuyerAgent' },
          { name: 'Seller', utility_class: 'SellerAgent' }
        ]
      },
      runs: [
        {
          output_variables: [
            { name: 'deal_reached', value: true },
            { name: 'final_price', value: 350 },
            { name: 'utility', value: { Buyer: 0.125, Seller: 0.875 } }
          ]
        }
      ],
      status: 'completed'
    };
    
    // Mock the simulation details API call
    api.getSimulationOutput.mockResolvedValueOnce(mockSimulation);

    renderWithRouter(
      <Routes>
        <Route path="/dashboard/:simulationId" element={<Dashboard />} />
      </Routes>,
      { route: '/dashboard/sim1' }
    );

    // Wait for data to load
    await waitFor(() => {
      expect(api.getSimulationOutput).toHaveBeenCalledWith('sim1');
    });

    // Check that ECharts was initialized
    expect(echarts.init).toHaveBeenCalled();

    // Get the chart options that were set
    const setOptionCalls = mockChartInstance.setOption.mock.calls;

    // Find the utility evolution chart (should have utility data)
    const utilityChart = setOptionCalls.find(call => {
      const options = call[0];
      return options.series && options.series.some(s => 
        s.data && s.data.some(d => 
          d.name === 'Buyer' || d.name === 'Seller'
        )
      );
    });

    expect(utilityChart).toBeDefined();

    // Verify utility values are present
    const utilityOptions = utilityChart[0];
    const buyerSeries = utilityOptions.series.find(s => s.name === 'Buyer');
    const sellerSeries = utilityOptions.series.find(s => s.name === 'Seller');

    expect(buyerSeries).toBeDefined();
    expect(sellerSeries).toBeDefined();
    expect(buyerSeries.data).toContain(0.125);
    expect(sellerSeries.data).toContain(0.875);
  });

  test('handles multiple runs with utility evolution', async () => {
    // Mock simulation data with multiple runs
    const mockSimulation = {
      simulation_id: 'sim1',
      config: {
        name: 'Multi-Run Simulation',
        agents: [
          { name: 'Buyer', utility_class: 'BuyerAgent' },
          { name: 'Seller', utility_class: 'SellerAgent' }
        ]
      },
      runs: [
        {
          output_variables: [
            { name: 'utility', value: { Buyer: 0.1, Seller: 0.9 } }
          ]
        },
        {
          output_variables: [
            { name: 'utility', value: { Buyer: 0.15, Seller: 0.85 } }
          ]
        },
        {
          output_variables: [
            { name: 'utility', value: { Buyer: 0.2, Seller: 0.8 } }
          ]
        }
      ],
      status: 'completed'
    };
    
    // Mock the simulation details API call
    api.getSimulationOutput.mockResolvedValueOnce(mockSimulation);

    renderWithRouter(
      <Routes>
        <Route path="/dashboard/:simulationId" element={<Dashboard />} />
      </Routes>,
      { route: '/dashboard/sim1' }
    );

    await waitFor(() => {
      expect(api.getSimulationOutput).toHaveBeenCalledWith('sim1');
    });

    // Get chart options
    const setOptionCalls = mockChartInstance.setOption.mock.calls;

    // Find utility evolution chart
    const utilityChart = setOptionCalls.find(call => {
      const options = call[0];
      return options.series && options.series.some(s => 
        s.data && Array.isArray(s.data) && s.data.length === 3
      );
    });

    expect(utilityChart).toBeDefined();

    const utilityOptions = utilityChart[0];
    const buyerSeries = utilityOptions.series.find(s => s.name === 'Buyer');
    
    // Verify all three utility values are present
    expect(buyerSeries.data).toEqual([0.1, 0.15, 0.2]);
  });

  test('handles missing utility data gracefully', async () => {
    // Mock simulation data without utility values
    const mockSimulation = {
      simulation_id: 'sim1',
      config: {
        name: 'No Utility Simulation',
        agents: [{ name: 'Agent1' }]
      },
      runs: [
        {
          output_variables: [
            { name: 'deal_reached', value: true },
            { name: 'final_price', value: 300 }
          ]
        }
      ],
      status: 'completed'
    };
    
    // Mock the simulation details API call
    api.getSimulationOutput.mockResolvedValueOnce(mockSimulation);

    renderWithRouter(
      <Routes>
        <Route path="/dashboard/:simulationId" element={<Dashboard />} />
      </Routes>,
      { route: '/dashboard/sim1' }
    );

    await waitFor(() => {
      expect(api.getSimulationOutput).toHaveBeenCalledWith('sim1');
    });

    // Should still render without errors
    expect(echarts.init).toHaveBeenCalled();
  });
});
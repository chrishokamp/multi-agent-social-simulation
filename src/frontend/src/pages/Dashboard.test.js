import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import Dashboard from './Dashboard';
import * as echarts from 'echarts';

// Mock ECharts
jest.mock('echarts', () => ({
  init: jest.fn(() => ({
    setOption: jest.fn(),
    dispose: jest.fn(),
    resize: jest.fn()
  }))
}));

// Mock fetch
global.fetch = jest.fn();

// Mock the API service
jest.mock('../../services/apiService', () => ({
  default: {
    getSimulationsCatalog: jest.fn(),
    getSimulationOutput: jest.fn()
  }
}));

import api from '../../services/apiService';

describe('Dashboard Component - Utility Evolution', () => {
  beforeEach(() => {
    fetch.mockClear();
    api.getSimulationsCatalog.mockClear();
    api.getSimulationOutput.mockClear();
  });

  test('renders utility evolution chart with correct data', async () => {
    // Mock simulation data with utility values
    const mockData = [
      {
        simulation_id: 'sim1',
        config: {
          name: 'Test Simulation',
          agents: [
            { name: 'Buyer', utility_class: 'BuyerAgent' },
            { name: 'Seller', utility_class: 'SellerAgent' }
          ]
        },
        results: [
          {
            output_variables: [
              { name: 'deal_reached', value: true },
              { name: 'final_price', value: 350 },
              { name: 'utility', value: { Buyer: 0.125, Seller: 0.875 } }
            ]
          }
        ],
        status: 'completed'
      }
    ];

    // Mock the catalog API call
    api.getSimulationsCatalog.mockResolvedValueOnce(mockData);
    
    // Mock the simulation details API call
    api.getSimulationOutput.mockResolvedValueOnce(mockData[0]);

    render(
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText(/Test Simulation/)).toBeInTheDocument();
    });

    // Check that ECharts was initialized
    expect(echarts.init).toHaveBeenCalled();

    // Get the chart options that were set
    const chartInstance = echarts.init.mock.results[0].value;
    const setOptionCalls = chartInstance.setOption.mock.calls;

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
    const mockData = [
      {
        simulation_id: 'sim1',
        config: {
          name: 'Multi-Run Simulation',
          agents: [
            { name: 'Buyer', utility_class: 'BuyerAgent' },
            { name: 'Seller', utility_class: 'SellerAgent' }
          ]
        },
        results: [
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
      }
    ];

    // Mock the catalog API call
    api.getSimulationsCatalog.mockResolvedValueOnce(mockData);
    
    // Mock the simulation details API call
    api.getSimulationOutput.mockResolvedValueOnce(mockData[0]);

    render(
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Multi-Run Simulation/)).toBeInTheDocument();
    });

    // Get chart options
    const chartInstance = echarts.init.mock.results[0].value;
    const setOptionCalls = chartInstance.setOption.mock.calls;

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
    const mockData = [
      {
        simulation_id: 'sim1',
        config: {
          name: 'No Utility Simulation',
          agents: [{ name: 'Agent1' }]
        },
        results: [
          {
            output_variables: [
              { name: 'deal_reached', value: true },
              { name: 'final_price', value: 300 }
            ]
          }
        ],
        status: 'completed'
      }
    ];

    // Mock the catalog API call
    api.getSimulationsCatalog.mockResolvedValueOnce(mockData);
    
    // Mock the simulation details API call
    api.getSimulationOutput.mockResolvedValueOnce(mockData[0]);

    render(
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/No Utility Simulation/)).toBeInTheDocument();
    });

    // Should still render without errors
    expect(echarts.init).toHaveBeenCalled();
  });
});
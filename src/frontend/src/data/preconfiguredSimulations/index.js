// Import all preconfigured simulations
import bikeNegotiation from './bikeNegotiation.json';
import carNegotiation from './carNegotiation.json';
import collaborativeNegotiation from './collaborativeNegotiation.json';

// Export as an array for easy iteration
export const preconfiguredSimulations = [
  bikeNegotiation,
  carNegotiation,
  collaborativeNegotiation
];

// Export as an object for easy lookup by ID
export const preconfiguredSimulationsById = {
  'bike-negotiation': bikeNegotiation,
  'car-negotiation': carNegotiation,
  'collaborative-negotiation': collaborativeNegotiation
};

// Export categories for filtering
export const simulationCategories = [
  { id: 'all', name: 'All Simulations' },
  { id: 'negotiation', name: 'Negotiation' }
];
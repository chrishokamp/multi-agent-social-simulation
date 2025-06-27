import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Dashboard from './pages/Dashboard';
import Configurator from './pages/Configurator';
import SimulationsList from './pages/SimulationsList';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Configurator />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/dashboard/:simulationId" element={<Dashboard />} />
        <Route path="/configurator" element={<Configurator />} />
        <Route path="/simulations" element={<SimulationsList />} />
      </Routes>
    </Router>
  );
};

export default App;

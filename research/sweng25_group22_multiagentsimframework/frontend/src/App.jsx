import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import LandingPage from './pages/LandingPage';
import Renderer from './pages/Renderer';
import Dashboard from './pages/Dashboard';
import Configurator from './pages/Configurator';
import SimulationsList from './pages/SimulationsList';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/renderer" element={<Renderer />} />
        <Route path="/renderer/:simulationId" element={<Renderer />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/dashboard/:simulationId" element={<Dashboard />} />
        <Route path="/configurator" element={<Configurator />} />
        <Route path="/simulations" element={<SimulationsList />} />
      </Routes>
    </Router>
  );
};

export default App;

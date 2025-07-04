import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Dashboard from './pages/Dashboard';
import Configuration from './pages/Configuration';
import SimulationsList from './pages/SimulationsList';
import Chat from './pages/Chat';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Configuration />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/dashboard/:simulationId" element={<Dashboard />} />
        <Route path="/Configuration" element={<Configuration />} />
        <Route path="/simulations" element={<SimulationsList />} />
        <Route path="/chat/:simulationId" element={<Chat />} />
      </Routes>
    </Router>
  );
};

export default App;

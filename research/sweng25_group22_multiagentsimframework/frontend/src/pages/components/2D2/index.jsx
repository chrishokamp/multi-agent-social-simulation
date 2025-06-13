import React, { useEffect } from 'react';
import './styles.css';
import { Overworld } from './Overworld'; // Import Overworld directly for better React integration

const Scene2D2 = () => {
  useEffect(() => {
    // Ensure the script runs after the component has mounted
    const overworld = new Overworld({
      element: document.querySelector('.game-container'),
    });
    overworld.init();

    // Cleanup function to reset or destroy Overworld
    return () => {
      overworld.destroy();
    };
  }, []); // Empty dependency array ensures this runs only once on mount

  return (
    <div className="game-container">
      <canvas className="game-canvas" width="352" height="198"></canvas>
    </div>
  );
};

export default Scene2D2;

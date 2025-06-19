import React, { useRef, useEffect } from "react";
import { Application, Assets, Sprite } from "pixi.js";
import simulation from "../../constants/simulationExample.json";
import * as PIXI from "pixi.js";



export default function Renderer({ simulationData }) {
  const { agents, conversation } = simulation;

  // Reference to the React object wrapping the PixiJS canvas
  const containerRef = useRef(null);

  // Reference to PixiJS application object
  const appRef = useRef(null);

  // Reference to an array of sprites currently in the scene
  const spritesRef = useRef([]);

  // Reference to ellapsed time of the render
  const timeRef = useRef(0);

  // Called once per frame
  const draw = (ticker) => {
    for (const sprite of spritesRef.current) {
      sprite.x += Math.cos(timeRef.current / 50);
    }
    timeRef.current += ticker.deltaTime;
  };

  // Create a new sprite from a builder object and add it to the scene
  // x, y - Initial position of the sprite
  // texture - URL to a texture asset
  const createSprite = async (x, y, texture) => {
    const spriteTexture = await Assets.load(texture);
    const sprite = Sprite.from(spriteTexture);

    sprite.anchor.set(0.5);
    sprite.x = x;
    sprite.y = y;
    sprite.width/=5;
    sprite.height/=5;

    appRef.current.stage.addChild(sprite);
    spritesRef.current.push(sprite);
  };

  const printText = (x, y, message) => {
    const text = new PIXI.Text(message, {
      fontSize: 15,
      fill: "white",
      background: "white",
      align: "center",
      wordWrap: true,          // Enable word wrapping
      wordWrapWidth: 200,      // Set maximum width before going to new line 
    });
  
    text.x = x - text.width / 2; // Center horizontally around the sprite position
    text.y = y - text.height / 2 - 60; // Position above the sprite
  
    appRef.current.stage.addChild(text);
  
    setTimeout(() => {
      appRef.current.stage.removeChild(text);
    }, 3000); // Remove after 3 seconds
  };
  

  const printConversation = () => {
    let index = 0;
  
    // Interval to show messages one by one
    const interval = setInterval(() => {
      if (index >= conversation.length) {
        clearInterval(interval); // Stop when all messages are displayed
        return;
      }
      const { speaker, message } = conversation[index]; // Get speaker and message
      const speakerIndex = agents.findIndex(agent => agent.name === speaker); // Find corresponding agent's index
  
      if (speakerIndex !== -1) {
        const sprite = spritesRef.current[speakerIndex]; 
        if (sprite) {
          // Call printText to display the message near sprite
          printText(sprite.x, sprite.y - 60, message); 
        }
      }
      index++; // Move to the next message
    }, 3000); // Display each message every 3 seconds
  };

  
  // Hook to initialize the renderer upon mounting of the React object
  useEffect(() => {
    const app = new Application();
    appRef.current = app;

    const promise = app.init();

    // Destroy the app upon unmount. An async function is required here
    // since the application may still be initializing.
    const clean = async () => {
      await promise;
      app.destroy(true);
    };

    // Wait for the application to be created and perform initialization.
    const init = async () => {
      await promise;

      containerRef.current.appendChild(app.canvas);

      console.log("App initialized, screen available:", app.screen);
        agents.forEach((agent, i) => {
        createSprite(
          (app.screen.width / (agents.length + 1)) * (i + 1), // Distribute sprites evenly
          app.screen.height / 2,  // Vertically center sprites
          agent.sprite,
          agent.name
        );
      });
      printText(app.screen.width/2, app.screen.height / 2-100, "Simulation starting"); 
      printConversation();
      
      
    };

    init();

    return () => clean();
  }, []);

  return <div ref={containerRef} />;
}


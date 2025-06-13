import React, { useRef, useEffect, useState } from 'react';
import { useGLTF, useAnimations, Html } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Loads a GLB with an animation, and displays text at certain times.
 */
export default function HeroScene3D({ url }) {
  const groupRef = useRef();
  const [currentMessage, setCurrentMessage] = useState('');

  // Load GLB & animations
  const { scene, animations } = useGLTF(url);
  const { actions, names } = useAnimations(animations, groupRef);

  // On mount, play the first animation
  useEffect(() => {
    if (!names.length) return;
    const mainClipName = names[0];
    const action = actions[mainClipName];

    // Loop the animation indefinitely
    action.setLoop(THREE.LoopRepeat);
    action.reset().play();
  }, [actions, names]);

  // Track animation time each frame, set text by intervals
  useFrame(() => {
    if (!names.length) return;
    const mainClipName = names[0];
    const action = actions[mainClipName];
    if (!action) return;

    const t = action.time; // current time (seconds) of the clip

    if (t < 4) {
      if (currentMessage !== "Hello there! I'm your AI simulation Assistant.") {
        setCurrentMessage("Hello there! I'm your AI simulation Assistant.");
      }
    } else if (t < 6) {
      if (currentMessage !== 'Welcome to our multi-agent playground.') {
        setCurrentMessage('Welcome to our multi-agent playground');
      }
    } else if (t < 9) {
      if (currentMessage !== 'Configure agents. Set Goals. Observe outcomes.') {
        setCurrentMessage('Configure agents. Set Goals. Observe outcomes.');
      }
    } else {
      if (currentMessage !== 'Ready to shape your simulation world?') {
        setCurrentMessage('Ready to shape your simulation world?');
      }
    }
  });

  return (
    <group ref={groupRef} scale={[2, 2, 2]} position={[0, -2, 0]}>
      <primitive object={scene} />

      {currentMessage && (
        <Html position={[0, 2.1, 0]} transform distanceFactor={4}>
          <div
            className={`
              transition-all duration-700 transform
              text-white px-4 py-2 rounded
              text-2xl font-bold 
            `}
            style={{
              opacity: 1,
            }}
          >
            {currentMessage}
          </div>
        </Html>
      )}
    </group>
  );
}

import React, { useRef, useEffect } from "react";
import * as THREE from "three";
import { useGLTF, Html } from "@react-three/drei";

const AgentMesh = ({ agent, agentRefs, state = {} }) => {
  const meshRef = useRef();
  const { speech = "" } = state;

  // Load the GLTF model.
  const { scene } = useGLTF(agent.model);

  useEffect(() => {
    // Register this mesh reference for parent usage.
    if (agentRefs && agentRefs.current) {
      agentRefs.current[agent.id] = meshRef;
    }
  }, [agent.id, agentRefs]);

  return (
    <group
      ref={meshRef}
      position={agent.position}
      rotation={agent.rotation ? new THREE.Euler(...agent.rotation) : undefined}
    >
      {/* Render the GLTF model */}
      <primitive object={scene} castShadow />

      {/* Render speech bubble if there is text */}
      {speech && (
        <Html position={[0, 2.3, 0]} transform occlude distanceFactor={5}>
          <div
            style={{
              background: "black",
              padding: "0.5rem 1rem",
              borderRadius: "0.5rem",
              border: "1px solid #ccc",
              whiteSpace: "nowrap",
              color: "white",
            }}
          >
            {speech}
          </div>
        </Html>
      )}
    </group>
  );
};

export default AgentMesh;

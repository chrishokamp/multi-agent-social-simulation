import React, { useRef, useEffect, useState, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import AgentMesh from "./AgentMesh";

const Scene3D = ({ simulationData }) => {
  const { agents, conversation } = simulationData;
  // Memoize the conversation so that its reference does not change on every render.
  const conv = useMemo(() => conversation, [conversation]);

  // State to hold each agent's current speech.
  const [agentStates, setAgentStates] = useState({});
  // Flag to indicate that agent states have been initialized.
  const [agentsInitialized, setAgentsInitialized] = useState(false);
  // State to track which conversation line we're on.
  const [conversationIndex, setConversationIndex] = useState(0);
  // References for each AgentMesh (if needed).
  const agentRefs = useRef({});

  // Initialize agentStates once agents are available.
  useEffect(() => {
    if (agents.length > 0) {
      const initialStates = {};
      agents.forEach((agent) => {
        initialStates[agent.id] = { speech: "" };
      });
      setAgentStates(initialStates);
      setAgentsInitialized(true);
      console.log("Agents initialized:", initialStates);
    }
  }, [agents]);

  // Use a single effect to step through the conversation.
  useEffect(() => {
    if (!agentsInitialized) return;
    if (conversationIndex >= conv.length) {
      console.log("Conversation ended");
      return;
    }

    const line = conv[conversationIndex];
    console.log(
      `Displaying conversation line ${conversationIndex}:`,
      line.text
    );
    // Set the speech text for the speaker.
    setAgentStates((prev) => ({
      ...prev,
      [line.speakerId]: { speech: line.text },
    }));

    // Set a timeout to clear the speech after 2 seconds...
    const timer1 = setTimeout(() => {
      console.log(`Clearing speech for speaker ${line.speakerId}`);
      setAgentStates((prev) => ({
        ...prev,
        [line.speakerId]: { speech: "" },
      }));
    }, 2000);

    // And then move to the next conversation line after an additional 0.5 seconds.
    const timer2 = setTimeout(() => {
      setConversationIndex((prev) => prev + 1);
    }, 2500); // 2000ms + 500ms

    // Cleanup both timers if the effect re‑runs or the component unmounts.
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [agentsInitialized, conversationIndex, conv]);

  return (
    <Canvas
      camera={{ position: [0, 5, 10], fov: 60 }}
      style={{ background: "#f0f0f0" }}
    >
      <OrbitControls makeDefault />

      {/* Basic lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} />

      {/* Simple floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[10, 10]} />
        <meshStandardMaterial color="white" />
      </mesh>

      {/* Render each agent */}
      {agents.map((agent) => (
        <AgentMesh
          key={agent.id}
          agent={agent}
          agentRefs={agentRefs}
          state={agentStates[agent.id] || { speech: "" }}
        />
      ))}
    </Canvas>
  );
};

export default Scene3D;

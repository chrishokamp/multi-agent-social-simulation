// import React, { useState, useEffect, useRef } from 'react';
// import { useParams, useNavigate } from 'react-router-dom';
// import { SIMULATION_DATA } from '../constants/simulationData';
// import Scene3D from './components/Scene3D';
// import Scene2D2 from './components/2D2/index';
// import Navbar from './components/Navbar';
// import conversationData from '../constants/conversation.json'; // Import JSON file
// import apiService from '../services/apiService';

// const Renderer = () => {
//   const data3d = SIMULATION_DATA;
//   const [simulationId, setSimulationId] = useState('');
//   const [context, setContext] = useState('2d');
//   const [conversation, setConversation] = useState([]);
//   const [isPaused, setIsPaused] = useState(true);
//   const [isSimulationOver, setIsSimulationOver] = useState(false);
//   const [currentMessageIndex, setCurrentMessageIndex] = useState(0);
//   const [isPanelVisible, setIsPanelVisible] = useState(true);
//   const [selectedRun, setSelectedRun] = useState(0); // State to track selected simulation run
//   const [isOpeningScreenVisible, setIsOpeningScreenVisible] = useState(true); // Default to showing the opening screen
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const [initialized, setInitialized] = useState(false);
//   const agentIndexRef = useRef(0);
//   const agentImagesRef = useRef({});
//   const [agentSpeaking, setAgentSpeaking] = useState('agent1');
//   const [data, setData] = useState(null); // Initialize state to store data

//   // State for simulation catalog
//   const [simulationCatalog, setSimulationCatalog] = useState([]);
//   const [catalogLoading, setCatalogLoading] = useState(false);

//   // Get simulationId from URL params
//   const params = useParams();
//   const navigate = useNavigate();

//   // Fetch simulation catalog
//   const fetchSimulationCatalog = async () => {
//     setCatalogLoading(true);
//     try {
//       const catalog = await apiService.getSimulationsCatalog();
//       setSimulationCatalog(catalog);
//       setCatalogLoading(false);
//     } catch (error) {
//       console.error('Error fetching simulation catalog:', error);
//       setError('Failed to load simulation catalog. Please try again later.');
//       setCatalogLoading(false);
//     }
//   };

//   // Handle direct simulation selection (when clicking a card)
//   const handleDirectSimulationSelect = (simulationId) => {
//     navigate(`/renderer/${simulationId}`);
//   };

//   // Check for URL parameters on component mount
//   useEffect(() => {
//     if (params.simulationId) {
//       // If URL has a simulation ID, use it and don't show the opening screen
//       setSimulationId(params.simulationId);
//       setIsOpeningScreenVisible(false);

//       // Automatically load simulation data for this ID
//       loadSimulationData(params.simulationId);
//     } else {
//       // If no simulationId in URL, show the opening screen
//       setIsOpeningScreenVisible(true);
//       fetchSimulationCatalog();
//       setLoading(false);
//     }
//     setInitialized(true);
//   }, [params.simulationId]);

//   // Function to load simulation data based on ID
//   const loadSimulationData = async (id) => {
//     if (!id) return;

//     setLoading(true);
//     setError('');

//     if (id === 'saved') {
//       // Use local saved simulation data
//       const formattedConversation = conversationData.runs[selectedRun].messages.map((msg) => ({
//         agent: msg.agent,
//         message: msg.message,
//       }));
//       setConversation(formattedConversation);
//       setLoading(false);
//     } else {
//       try {
//         // Make an API call to fetch the simulation data
//         const response = await fetch(
//           `http://localhost:5000/sim/results?id=${id}&show_messages=yes`
//         );

//         if (!response.ok) {
//           throw new Error('Simulation not found.');
//         }

//         const data = await response.json();
//         setData(data);
//         const formattedConversation = data.runs[selectedRun].messages.map((msg) => ({
//           agent: msg.agent,
//           message: msg.message,
//         }));
//         setConversation(formattedConversation);
//         setLoading(false);
//       } catch (error) {
//         setError('Failed to load simulation data. Please check the ID and try again.');
//         setLoading(false);
//       }
//     }
//   };

//   // Fetch conversation data when selectedRun changes for 'saved' simulations
//   useEffect(() => {
//     if (simulationId === 'saved') {
//       const formattedConversation = conversationData.runs[selectedRun].messages.map((msg) => ({
//         agent: msg.agent,
//         message: msg.message,
//       }));
//       setConversation(formattedConversation);
//     } else if (simulationId) {
//       loadSimulationData(simulationId);
//     }
//   }, [selectedRun, simulationId]);

//   const getUniqueAgentCount = (conversation) => {
//     const uniqueAgents = [
//       ...new Set(
//         conversation
//           .filter((msg) => msg.agent && msg.agent !== 'InformationReturnAgent')
//           .map((msg) => msg.agent)
//       ),
//     ];

//     return uniqueAgents;
//   };

//   // update agentSpeaking
//   useEffect(() => {
//     if (conversation.length > 0) {
//       let latestAgent = 'none';
//       const uniqueAgents = getUniqueAgentCount(conversation); // Get the list of unique agents
//       // Create a map to store agent names and their assigned number
//       let agentMap = {};
//       uniqueAgents.forEach((agent, index) => {
//         agentMap[agent] = (index + 1).toString(); // Map each agent to 1, 2, 3...
//       });
//       // Find the agent speaking at the current message index
//       for (let i = currentMessageIndex; i >= 0; i--) {
//         if (conversation[i].agent && conversation[i].agent !== 'InformationReturnAgent') {
//           latestAgent = agentMap[conversation[i].agent]; // Get the agent's number from the map
//           break;
//         }
//       }
//       setAgentSpeaking(latestAgent);
//       window.agentSpeaking = latestAgent; // Update global agent speaking
//       console.log('Agent speaking:', latestAgent);
//     } else {
//       window.agentSpeaking = 'none';
//       setAgentSpeaking('none');
//     }
//   }, [currentMessageIndex, conversation]);

//   // Handle Simulation ID input change
//   const handleSimulationIdChange = (event) => {
//     setSimulationId(event.target.value);
//   };

//   // Handle saved simulation selection
//   const handleSavedSimulation = () => {
//     setSimulationId('saved');
//     setIsOpeningScreenVisible(false);
//     loadSimulationData('saved');
//   };

//   // Hide the opening screen when the simulation ID is entered or a saved simulation is selected
//   const handleStartSimulation = async () => {
//     if (!simulationId) {
//       setError('Simulation ID cannot be empty.'); // Set error message
//       return;
//     }

//     setLoading(true);
//     try {
//       // Make an API call to check if the simulation exists
//       const response = await fetch(
//         `http://localhost:5000/sim/results?id=${simulationId}&show_messages=yes`
//       );

//       if (!response.ok) {
//         throw new Error('Simulation not found.'); // If the response is not ok, throw an error
//       }

//       // If the simulation is found, proceed with starting the simulation
//       setError(''); // Clear any previous errors
//       console.log('Starting simulation with ID:', simulationId);
//       setIsOpeningScreenVisible(false); // Hide the opening screen

//       const data = await response.json();
//       setData(data);
//       const formattedConversation = data.runs[selectedRun].messages.map((msg) => ({
//         agent: msg.agent,
//         message: msg.message,
//       }));
//       setConversation(formattedConversation);

//       setLoading(false);

//       // Update the URL without refreshing the page (for bookmarking purposes)
//       window.history.pushState({}, '', `/renderer/${simulationId}`);
//     } catch (error) {
//       // Catch any errors (e.g., simulation not found)
//       setError('Simulation ID does not exist. Please check the ID and try again.');
//       setLoading(false);
//     }
//   };

//   useEffect(() => {
//     let messageInterval;
//     if (!isPaused && !isSimulationOver && conversation.length > 0) {
//       messageInterval = setInterval(() => {
//         setCurrentMessageIndex((prevIndex) => {
//           if (prevIndex + 1 >= conversation.length) {
//             clearInterval(messageInterval);
//             setIsSimulationOver(true);
//             setIsPaused(true);
//             return prevIndex;
//           }
//           return prevIndex + 1;
//         });
//       }, 2000);
//     }
//     return () => clearInterval(messageInterval);
//   }, [isPaused, isSimulationOver, conversation]);

//   const handleRestart = () => {
//     setCurrentMessageIndex(0);
//     setIsSimulationOver(false);
//   };

//   const handleTogglePlayPause = () => {
//     if (isPaused && !isSimulationOver) {
//       // Start the game (unpause)
//       setIsPaused(false);
//       window.isSimPaused = false; // Set global variable to resume the game
//     } else {
//       // Pause the game
//       setIsPaused(true);
//       window.isSimPaused = true; // Set global variable to pause the game
//     }
//   };

//   // Toggle context (2D or 3D)
//   const toggleContext = () => {
//     setContext((prev) => (prev === '2d' ? '3d' : '2d'));
//   };

//   const togglePanel = () => {
//     setIsPanelVisible((prev) => !prev);
//   };

//   // Function to render the appropriate scene
//   const getScene = () => {
//     switch (context) {
//       case '2d':
//         return <Scene2D2 key={`2d-${Date.now()}`} />;
//       case '3d':
//         return <Scene3D simulationData={data3d} />;
//       default:
//         return <></>;
//     }
//   };

//   const handleNextMessage = () => {
//     if (currentMessageIndex + 2 < conversation.length) {
//       setCurrentMessageIndex(currentMessageIndex + 1);
//     } else {
//       setIsSimulationOver(true);
//       setIsPaused(true);
//     }
//   };

//   const handlePrevMessage = () => {
//     if (currentMessageIndex > 0) {
//       setCurrentMessageIndex(currentMessageIndex - 1);
//     }
//   };

//   // Handle run selection from dropdown
//   const handleRunChange = (event) => {
//     const selectedRunIndex = parseInt(event.target.value);
//     setSelectedRun(selectedRunIndex);
//     setCurrentMessageIndex(0);
//     setIsSimulationOver(false);
//   };

//   function getAgentImage(agent, avatarOptions) {
//     if (agentImagesRef.current[agent]) {
//       return `/images/${agentImagesRef.current[agent]}`;
//     }
//     let assignedImage = null;

//     if (agentIndexRef.current < avatarOptions.length) {
//       // Bind the first distinct agent name to agent1.png, second to agent2.png, third to agent3.png
//       assignedImage = avatarOptions[agentIndexRef.current % avatarOptions.length];
//       agentImagesRef.current[agent] = assignedImage;
//       agentIndexRef.current++;
//     }
//     return `/images/${assignedImage || 'default.png'}`;
//   }

//   // Show a loading screen while initializing
//   if (!initialized) {
//     return (
//       <div className="w-full h-screen flex flex-col justify-center items-center bg-slate-900">
//         <Navbar />
//         <div className="text-white text-xl mt-20">Loading...</div>
//       </div>
//     );
//   }

//   return (
//     <div style={{ overflow: 'auto', height: '100vh' }}>
//       <Navbar />

//       {/* Opening Screen */}
//       {isOpeningScreenVisible && (
//         <div className="w-full min-h-screen">
//           <div className="fixed inset-0 flex items-center justify-center">
//             <div className="bg-violet-900/5 backdrop-blur-3xl p-10 w-96 md:w-1/2 lg:w-1/3 min-h-[300px] flex flex-col justify-center rounded-xl shadow-violet-600/60 shadow-card text-center">
//               <h2 className="text-2xl md:text-3xl font-bold text-violet-400">
//                 Select a Simulation
//               </h2>
//               <p className="text-lg text-gray-400 mt-4">
//                 Choose the simulation that you would like to render.
//               </p>

//               {/* Simulation List as Cards */}
//               <div className="mt-6">
//                 {simulationCatalog.map((sim) => (
//                   <div
//                     key={sim.simulation_id}
//                     onClick={() => setSimulationId(sim.simulation_id)}
//                     className={`p-3 my-2 flex justify-between items-center bg-violet-900/15 border rounded-lg text-left cursor-pointer transition-colors duration-200 ${
//                       simulationId === sim.simulation_id
//                         ? 'bg-violet-900/30 border border-violet-500'
//                         : 'border border-violet-400/50 hover:bg-violet-900/30'
//                     }`}
//                   >
//                     <p className="text-white">{sim.name || `Simulation ${sim.simulation_id}`}</p>
//                     {simulationId === sim.simulation_id && (
//                       <span className="text-emerald-500 text-xl">&#x2713;</span> // Checkmark icon
//                     )}
//                   </div>
//                 ))}
//               </div>

//               {/* Error Message */}
//               {error && <p className="text-red-500 mt-4">{error}</p>}

//               {/* View Render Button */}
//               <div className="mt-6">
//                 <button
//                   onClick={() => handleDirectSimulationSelect(simulationId)}
//                   disabled={!simulationId}
//                   className={`${
//                     !simulationId
//                       ? 'bg-violet-900/50 cursor-not-allowed'
//                       : 'bg-violet-800 hover:shadow-button'
//                   } text-white px-6 py-3 rounded-full transition-colors duration-200 cursor-pointer`}
//                 >
//                   View Render
//                 </button>
//               </div>
//             </div>
//           </div>
//         </div>
//       )}

//       {/* Show loading indicator while fetching data */}
//       {loading && !isOpeningScreenVisible && (
//         <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-30 z-40">
//           <div className="bg-white p-6 rounded-lg shadow-lg">
//             <p className="text-lg">Loading simulation data...</p>
//           </div>
//         </div>
//       )}

//       {/* Main content - only visible when not showing opening screen and not loading */}
//       {!isOpeningScreenVisible && !loading && (
//         <>
//           {/* Dropdown for selecting simulation run */}

//           <div className="absolute top-20 left-1/4 transform -translate-x-1/2 z-10">
//             <select
//               value={selectedRun}
//               onChange={handleRunChange}
//               className="bg-white text-gray-800 font-bold py-2 px-4 rounded border"
//             >
//               {conversationData.runs.map((run, index) => (
//                 <option key={index} value={index}>
//                   {`Run ${index + 1}`}
//                 </option>
//               ))}
//             </select>
//           </div>

//           {/* Scene and side Panel */}
//           <div className="flex flex-row flex-1 w-full mt-16 overflow:auto">
//             {/* Scene Container (takes remaining space) */}
//             <div
//               className="flex flex-1 flex-col justify-center items-center relative min-h-screen"
//               style={{ marginTop: '-20px', overflow: 'hidden' }}
//             >
//               {getScene()}
//             </div>

//             {/* Playback Controls */}
//             {context === '2d' && (
//               <div className="absolute bottom-5 left-1/2 transform -translate-x-1/2 flex space-x-6">
//                 <button
//                   onClick={handleRestart}
//                   className="bg-violet-600 text-white px-4 py-2 rounded shadow-md hover:bg-violet-700"
//                 >
//                   ↺
//                 </button>

//                 {/* Step buttons */}
//                 <button
//                   onClick={handlePrevMessage}
//                   className="bg-violet-600 text-white px-4 py-2 rounded shadow-md hover:bg-violet-700"
//                 >
//                   «
//                 </button>

//                 <button
//                   onClick={handleTogglePlayPause}
//                   className="bg-violet-600 text-white px-4 py-2 rounded shadow-md hover:bg-violet-700"
//                 >
//                   {isPaused ? '▶' : '⏸'}
//                 </button>

//                 <button
//                   onClick={handleNextMessage}
//                   className="bg-violet-600 text-white px-4 py-2 rounded shadow-md hover:bg-violet-700"
//                 >
//                   »
//                 </button>
//               </div>
//             )}

//             {/* Panel Toggle Button */}
//             <button
//               onClick={togglePanel}
//               className={`fixed top-20 ${isPanelVisible ? 'right-1/4' : 'right-5'}
//                 bg-white hover:bg-gray-300 font-bold px-4 py-2 rounded shadow-md z-20`}
//             >
//               ☰
//             </button>

//             {/* side Conversation Panel (only visible in 2D render) */}
//             {context === '2d' && isPanelVisible && (
//               <div
//                 className="w-1/4 max-h-screen bg-midnight p-4 overflow-auto border shadow-lg shadow-violet-600/60"
//                 style={{
//                   maxHeight: 'calc(100vh - 64px)', // Ensure the panel doesn't exceed screen height minus top margin
//                   position: 'fixed', // Fix the side panel in place
//                   top: '64px', // Adjust top offset for header
//                   right: 0, // Position the panel on the right
//                   zIndex: 10, // Make sure it appears above other content
//                 }}
//               >
//                 <h2 className="text-lg font-bold mt-2 mb-2 text-white">Conversation</h2>

//                 <div className="space-y-4">
//                   {conversation
//                     .slice(0, currentMessageIndex + 1)
//                     .filter(
//                       (msg) =>
//                         msg.agent !== 'InformationReturnAgent' || !msg.message.includes('TERMINATE')
//                     )
//                     .map((msg, index) => {
//                       // Filter out unique agents excluding "InformationReturnAgent"
//                       const uniqueAgents = [
//                         ...new Set(
//                           conversation
//                             .filter((msg) => msg.agent && msg.agent !== 'InformationReturnAgent')
//                             .map((msg) => msg.agent)
//                         ),
//                       ];

//                       // Decide on avatar options (2 or 3 images based on agent count)
//                       const avatarOptions =
//                         uniqueAgents.length >= 3
//                           ? ['agent1.png', 'agent2.png', 'agent3.png']
//                           : ['agent1.png', 'agent2.png'];

//                       // Render the message with avatar
//                       return (
//                         <div key={index} className="flex items-start space-x-4">
//                           {/* Avatar Circle */}
//                           <div className="min-w-10 w-10 h-10 flex-shrink-0 flex justify-center items-center bg-gray-500 rounded-full overflow-hidden">
//                             <img
//                               src={getAgentImage(msg.agent, avatarOptions)}
//                               alt={`${msg.agent} Avatar`}
//                               className="w-full h-full object-cover"
//                             />
//                           </div>

//                           {/* Message */}
//                           <div className="flex-grow p-2 bg-violet-950/20 border border-violet-400 rounded-lg">
//                             <p className="text-sm text-gray-200">
//                               <strong className="text-white">{msg.agent}:</strong> {msg.message}
//                             </p>
//                           </div>
//                         </div>
//                       );
//                     })}
//                 </div>
//               </div>
//             )}
//           </div>

//           {/* Simulation Over Modal */}
//           {isSimulationOver && (
//             <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
//               <div className="bg-white p-10 w-96 md:w-1/2 lg:w-1/3 min-h-[300px] flex flex-col justify-center rounded-xl shadow-2xl text-center">
//                 <h2 className="text-2xl md:text-3xl font-bold text-gray-800">Simulation Over</h2>
//                 <p className="text-lg text-gray-600 mt-4">The conversation has ended.</p>

//                 {/* Check for the selected run's output variables */}
//                 {simulationId === 'saved' ? (
//                   conversationData.runs[selectedRun]?.output_variables ? (
//                     <div className="mt-6 text-sm text-gray-700">
//                       <h3 className="font-bold text-gray-800">End result:</h3>
//                       {/* Loop through output variables and display them dynamically */}
//                       {conversationData.runs[selectedRun].output_variables.map((output, index) => (
//                         <div key={index}>
//                           <p>
//                             <strong>{output.name.replace('_', ' ').toUpperCase()}:</strong>{' '}
//                             {output.value}
//                           </p>
//                         </div>
//                       ))}
//                     </div>
//                   ) : null
//                 ) : data.runs[selectedRun]?.output_variables ? (
//                   <div className="mt-6 text-sm text-gray-700">
//                     <h3 className="font-bold text-gray-800">End result:</h3>
//                     {/* Loop through output variables and display them dynamically */}
//                     {console.log('Output Variables: ', data.runs[selectedRun].output_variables)}
//                     {data.runs[selectedRun].output_variables.map((output, index) => (
//                       <div key={index}>
//                         <p>
//                           <strong>{output.name.replace('_', ' ').toUpperCase()}:</strong>{' '}
//                           {output.value}
//                         </p>
//                       </div>
//                     ))}
//                   </div>
//                 ) : null}

//                 <button
//                   onClick={handleRestart}
//                   className="mt-6 bg-violet-600 text-white text-lg px-6 py-3 rounded-lg shadow-lg hover:bg-violet-700"
//                 >
//                   Restart
//                 </button>
//               </div>
//             </div>
//           )}
//         </>
//       )}
//     </div>
//   );
// };

// export default Renderer;

import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { SIMULATION_DATA } from '../constants/simulationData';
import Scene3D from './components/Scene3D';
import Scene2D2 from './components/2D2/index';
import Navbar from './components/Navbar';
import conversationData from '../constants/conversation.json'; // Import JSON file
import apiService from '../services/apiService';

const Renderer = () => {
  const data3d = SIMULATION_DATA;
  const [simulationId, setSimulationId] = useState('');
  const [context, setContext] = useState('2d');
  const [conversation, setConversation] = useState([]);
  const [isPaused, setIsPaused] = useState(true);
  const [isSimulationOver, setIsSimulationOver] = useState(false);
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);
  const [isPanelVisible, setIsPanelVisible] = useState(true);
  const [selectedRun, setSelectedRun] = useState(0); // State to track selected simulation run
  const [isOpeningScreenVisible, setIsOpeningScreenVisible] = useState(true); // Default to showing the opening screen
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [initialized, setInitialized] = useState(false);
  const agentIndexRef = useRef(0);
  const agentImagesRef = useRef({});
  const [agentSpeaking, setAgentSpeaking] = useState('agent1');
  const [data, setData] = useState(null); // Initialize state to store data
  const [showOutcomeModal, setShowOutcomeModal] = useState(false); // New state for outcome modal

  // State for simulation catalog
  const [simulationCatalog, setSimulationCatalog] = useState([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [availableRuns, setAvailableRuns] = useState([]);

  // Get simulationId from URL params
  const params = useParams();
  const navigate = useNavigate();

  // Fetch simulation catalog
  const fetchSimulationCatalog = async () => {
    setCatalogLoading(true);
    try {
      const catalog = await apiService.getSimulationsCatalog();
      setSimulationCatalog(catalog);
      setCatalogLoading(false);
    } catch (error) {
      console.error('Error fetching simulation catalog:', error);
      setError('Failed to load simulation catalog. Please try again later.');
      setCatalogLoading(false);
    }
  };

  // Handle direct simulation selection (when clicking a card)
  const handleDirectSimulationSelect = (simulationId) => {
    navigate(`/renderer/${simulationId}`);
  };

  // Navigate back to simulations catalog
  const handleBackToCatalog = () => {
    navigate('/simulations');
  };

  // Check for URL parameters on component mount
  useEffect(() => {
    if (params.simulationId) {
      // If URL has a simulation ID, use it and don't show the opening screen
      setSimulationId(params.simulationId);
      setIsOpeningScreenVisible(false);

      // Automatically load simulation data for this ID
      loadSimulationData(params.simulationId);
    } else {
      // If no simulationId in URL, show the opening screen
      setIsOpeningScreenVisible(true);
      fetchSimulationCatalog();
      setLoading(false);
    }
    setInitialized(true);
  }, [params.simulationId]);

  // Function to load simulation data based on ID
  const loadSimulationData = async (id) => {
    if (!id) return;

    setLoading(true);
    setError('');

    if (id === 'saved') {
      // Use local saved simulation data
      const formattedConversation = conversationData.runs[selectedRun].messages.map((msg) => ({
        agent: msg.agent,
        message: msg.message,
      }));
      setConversation(formattedConversation);

      // Set available runs based on the data
      const runsCount = conversationData.runs.length;
      setAvailableRuns(Array.from({ length: runsCount }, (_, i) => i));

      setLoading(false);
    } else {
      try {
        // Make an API call to fetch the simulation data
        const response = await fetch(
          `http://localhost:5000/sim/results?id=${id}&show_messages=yes`
        );

        if (!response.ok) {
          throw new Error('Simulation not found.');
        }

        const data = await response.json();
        setData(data);
        const formattedConversation = data.runs[selectedRun].messages.map((msg) => ({
          agent: msg.agent,
          message: msg.message,
        }));
        setConversation(formattedConversation);

        // Set available runs based on the data
        const runsCount = data.runs.length;
        setAvailableRuns(Array.from({ length: runsCount }, (_, i) => i));

        setLoading(false);
      } catch (error) {
        setError('Failed to load simulation data. Please check the ID and try again.');
        setLoading(false);
      }
    }
  };

  // Fetch conversation data when selectedRun changes for 'saved' simulations
  useEffect(() => {
    if (simulationId === 'saved' && conversationData.runs && conversationData.runs[selectedRun]) {
      const formattedConversation = conversationData.runs[selectedRun].messages.map((msg) => ({
        agent: msg.agent,
        message: msg.message,
      }));
      setConversation(formattedConversation);
      setCurrentMessageIndex(0); // Reset message index when changing run
      setIsSimulationOver(false);
    } else if (simulationId && data && data.runs && data.runs[selectedRun]) {
      const formattedConversation = data.runs[selectedRun].messages.map((msg) => ({
        agent: msg.agent,
        message: msg.message,
      }));
      setConversation(formattedConversation);
      setCurrentMessageIndex(0); // Reset message index when changing run
      setIsSimulationOver(false);
    }
  }, [selectedRun, simulationId, data]);

  const getUniqueAgentCount = (conversation) => {
    const uniqueAgents = [
      ...new Set(
        conversation
          .filter((msg) => msg.agent && msg.agent !== 'InformationReturnAgent')
          .map((msg) => msg.agent)
      ),
    ];

    return uniqueAgents;
  };

  // update agentSpeaking
  useEffect(() => {
    if (conversation.length > 0) {
      let latestAgent = 'none';
      const uniqueAgents = getUniqueAgentCount(conversation); // Get the list of unique agents
      // Create a map to store agent names and their assigned number
      let agentMap = {};
      uniqueAgents.forEach((agent, index) => {
        agentMap[agent] = (index + 1).toString(); // Map each agent to 1, 2, 3...
      });
      // Find the agent speaking at the current message index
      for (let i = currentMessageIndex; i >= 0; i--) {
        if (conversation[i].agent && conversation[i].agent !== 'InformationReturnAgent') {
          latestAgent = agentMap[conversation[i].agent]; // Get the agent's number from the map
          break;
        }
      }
      setAgentSpeaking(latestAgent);
      window.agentSpeaking = latestAgent; // Update global agent speaking
      console.log('Agent speaking:', latestAgent);
    } else {
      window.agentSpeaking = 'none';
      setAgentSpeaking('none');
    }
  }, [currentMessageIndex, conversation]);

  // Handle Simulation ID input change
  const handleSimulationIdChange = (event) => {
    setSimulationId(event.target.value);
  };

  // Handle saved simulation selection
  const handleSavedSimulation = () => {
    setSimulationId('saved');
    setIsOpeningScreenVisible(false);
    loadSimulationData('saved');
  };

  // Hide the opening screen when the simulation ID is entered or a saved simulation is selected
  const handleStartSimulation = async () => {
    if (!simulationId) {
      setError('Simulation ID cannot be empty.'); // Set error message
      return;
    }

    setLoading(true);
    try {
      // Make an API call to check if the simulation exists
      const response = await fetch(
        `http://localhost:5000/sim/results?id=${simulationId}&show_messages=yes`
      );

      if (!response.ok) {
        throw new Error('Simulation not found.'); // If the response is not ok, throw an error
      }

      // If the simulation is found, proceed with starting the simulation
      setError(''); // Clear any previous errors
      console.log('Starting simulation with ID:', simulationId);
      setIsOpeningScreenVisible(false); // Hide the opening screen

      const data = await response.json();
      setData(data);
      const formattedConversation = data.runs[selectedRun].messages.map((msg) => ({
        agent: msg.agent,
        message: msg.message,
      }));
      setConversation(formattedConversation);

      setLoading(false);

      // Update the URL without refreshing the page (for bookmarking purposes)
      window.history.pushState({}, '', `/renderer/${simulationId}`);
    } catch (error) {
      // Catch any errors (e.g., simulation not found)
      setError('Simulation ID does not exist. Please check the ID and try again.');
      setLoading(false);
    }
  };

  useEffect(() => {
    let messageInterval;
    if (!isPaused && !isSimulationOver && conversation.length > 0) {
      messageInterval = setInterval(() => {
        setCurrentMessageIndex((prevIndex) => {
          if (prevIndex + 1 >= conversation.length) {
            clearInterval(messageInterval);
            setIsSimulationOver(true);
            setIsPaused(true);
            return prevIndex;
          }
          return prevIndex + 1;
        });
      }, 2000);
    }
    return () => clearInterval(messageInterval);
  }, [isPaused, isSimulationOver, conversation]);

  const handleRestart = () => {
    setCurrentMessageIndex(0);
    setIsSimulationOver(false);
  };

  const handleTogglePlayPause = () => {
    if (isPaused && !isSimulationOver) {
      // Start the game (unpause)
      setIsPaused(false);
      window.isSimPaused = false; // Set global variable to resume the game
    } else {
      // Pause the game
      setIsPaused(true);
      window.isSimPaused = true; // Set global variable to pause the game
    }
  };

  // Toggle context (2D or 3D)
  const toggleContext = () => {
    setContext((prev) => (prev === '2d' ? '3d' : '2d'));
  };

  const togglePanel = () => {
    setIsPanelVisible((prev) => !prev);
  };

  // Function to show outcome modal
  const handleShowOutcome = () => {
    setShowOutcomeModal(true);
  };

  // Function to render the appropriate scene
  const getScene = () => {
    switch (context) {
      case '2d':
        return <Scene2D2 key={`2d-${Date.now()}`} />;
      case '3d':
        return <Scene3D simulationData={data3d} />;
      default:
        return <></>;
    }
  };

  const handleNextMessage = () => {
    if (currentMessageIndex + 2 < conversation.length) {
      setCurrentMessageIndex(currentMessageIndex + 1);
    } else {
      setIsSimulationOver(true);
      setIsPaused(true);
    }
  };

  const handlePrevMessage = () => {
    if (currentMessageIndex > 0) {
      setCurrentMessageIndex(currentMessageIndex - 1);
    }
  };

  // Handle run selection from dropdown
  const handleRunChange = (event) => {
    const selectedRunIndex = parseInt(event.target.value);
    setSelectedRun(selectedRunIndex);
    setCurrentMessageIndex(0);
    setIsSimulationOver(false);
    setShowOutcomeModal(false);
  };

  function getAgentImage(agent, avatarOptions) {
    if (agentImagesRef.current[agent]) {
      return `/images/${agentImagesRef.current[agent]}`;
    }
    let assignedImage = null;

    if (agentIndexRef.current < avatarOptions.length) {
      // Bind the first distinct agent name to agent1.png, second to agent2.png, third to agent3.png
      assignedImage = avatarOptions[agentIndexRef.current % avatarOptions.length];
      agentImagesRef.current[agent] = assignedImage;
      agentIndexRef.current++;
    }
    return `/images/${assignedImage || 'default.png'}`;
  }

  // Show a loading screen while initializing
  if (!initialized) {
    return (
      <div className="w-full h-screen flex flex-col justify-center items-center bg-slate-900">
        <Navbar />
        <div className="text-white text-xl mt-20">Loading...</div>
      </div>
    );
  }

  return (
    <div style={{ overflow: 'auto', height: '100vh' }}>
      <Navbar />

      {/* Opening Screen */}
      {isOpeningScreenVisible && (
        <div className="w-full min-h-screen">
          <div className="fixed inset-0 flex items-center justify-center">
            <div className="bg-violet-900/5 backdrop-blur-3xl p-10 w-96 md:w-1/2 lg:w-1/3 min-h-[300px] flex flex-col justify-center rounded-xl shadow-violet-600/60 shadow-card text-center">
              <h2 className="text-2xl md:text-3xl font-bold text-violet-400">
                Select a Simulation
              </h2>
              <p className="text-lg text-gray-400 mt-4">
                Choose the simulation that you would like to render.
              </p>

              {/* Simulation List as Cards */}
              <div className="mt-6">
                {simulationCatalog.map((sim) => (
                  <div
                    key={sim.simulation_id}
                    onClick={() => setSimulationId(sim.simulation_id)}
                    className={`p-3 my-2 flex justify-between items-center bg-violet-900/15 border rounded-lg text-left cursor-pointer transition-colors duration-200 ${
                      simulationId === sim.simulation_id
                        ? 'bg-violet-900/30 border border-violet-500'
                        : 'border border-violet-400/50 hover:bg-violet-900/30'
                    }`}
                  >
                    <p className="text-white">{sim.name || `Simulation ${sim.simulation_id}`}</p>
                    {simulationId === sim.simulation_id && (
                      <span className="text-emerald-500 text-xl">&#x2713;</span> // Checkmark icon
                    )}
                  </div>
                ))}
              </div>

              {/* Error Message */}
              {error && <p className="text-red-500 mt-4">{error}</p>}

              {/* View Render Button */}
              <div className="mt-6">
                <button
                  onClick={() => handleDirectSimulationSelect(simulationId)}
                  disabled={!simulationId}
                  className={`${
                    !simulationId
                      ? 'bg-violet-900/50 cursor-not-allowed'
                      : 'bg-violet-800 hover:shadow-button'
                  } text-white px-6 py-3 rounded-full transition-colors duration-200 cursor-pointer`}
                >
                  View Render
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Show loading indicator while fetching data */}
      {loading && !isOpeningScreenVisible && (
        <div className="absolute inset-0 flex items-center justify-center bg-midnight bg-opacity-30 z-40">
          <div className="bg-violet-900/10 p-6 rounded-lg shadow-lg border border-violet-400/40">
            <p className="text-lg text-white">Loading simulation data...</p>
          </div>
        </div>
      )}

      {/* Main content - only visible when not showing opening screen and not loading */}
      {!isOpeningScreenVisible && !loading && (
        <>
          {/* Control buttons in the top area */}
          <div className="fixed top-20 left-0 right-0 flex justify-center mt-10 space-x-6 z-20">
            {/* Back to Catalog button */}
            <button
              onClick={handleBackToCatalog}
              className="bg-violet-800 text-white font-bold py-2 px-4 rounded-lg shadow-md flex items-center hover:shadow-button"
            >
              <span>&#8592;</span>
              <span className="ml-2">Back to Catalog</span>
            </button>

            {/* Dropdown for selecting simulation run */}
            <select
              value={selectedRun}
              onChange={handleRunChange}
              className="custom-select bg-violet-900/20 text-white font-bold py-2 px-4 rounded-lg border border-violet-400 cursor-pointer transition-colors duration-200"
            >
              {availableRuns.map((runIndex) => (
                <option key={runIndex} value={runIndex}>
                  {`Run ${runIndex + 1}`}
                </option>
              ))}
            </select>
          </div>

          {/* Main content container */}
          <div className="flex justify-center w-full mt-28 overflow-auto">
            <div className="flex flex-row max-w-6xl mx-auto gap-6">
              {/* Scene Container */}
              <div className="flex-1 flex flex-col justify-center items-center relative min-h-[70vh]">
                {getScene()}

                {/* Playback Controls */}
                {context === '2d' && (
                  <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex space-x-4 bg-violet-900/20 p-3 rounded-full shadow-lg">
                    <button
                      onClick={handleRestart}
                      className="bg-violet-600 text-white w-12 h-12 rounded-full shadow-md hover:bg-violet-700 flex items-center justify-center transition-all duration-300 hover:shadow-lg hover:scale-105"
                      title="Restart"
                    >
                      <span className="text-xl">↺</span>
                    </button>

                    <button
                      onClick={handlePrevMessage}
                      className="bg-violet-600 text-white w-12 h-12 rounded-full shadow-md hover:bg-violet-700 flex items-center justify-center transition-all duration-300 hover:shadow-lg hover:scale-105"
                      title="Previous Message"
                    >
                      <span className="text-xl">«</span>
                    </button>

                    <button
                      onClick={handleTogglePlayPause}
                      className="bg-violet-600 text-white w-12 h-12 rounded-full shadow-md hover:bg-violet-700 flex items-center justify-center transition-all duration-300 hover:shadow-lg hover:scale-105"
                      title={isPaused ? 'Play' : 'Pause'}
                    >
                      <span className="text-xl">{isPaused ? '▶' : '⏸'}</span>
                    </button>

                    <button
                      onClick={handleNextMessage}
                      className="bg-violet-600 text-white w-12 h-12 rounded-full shadow-md hover:bg-violet-700 flex items-center justify-center transition-all duration-300 hover:shadow-lg hover:scale-105"
                      title="Next Message"
                    >
                      <span className="text-xl">»</span>
                    </button>

                    {/* New button to show outcome */}
                    <button
                      onClick={handleShowOutcome}
                      className="bg-violet-600 text-white w-12 h-12 rounded-full shadow-md hover:bg-violet-700 flex items-center justify-center transition-all duration-300 hover:shadow-lg hover:scale-105"
                      title="Show Outcome"
                    >
                      <span className="text-xl">⚑</span>
                    </button>
                  </div>
                )}
              </div>

              {/* Panel Toggle Button - now between the sections */}
              {context === '2d' && (
                <button
                  onClick={togglePanel}
                  className="self-center bg-violet-600 hover:bg-violet-700 text-white font-bold px-3 py-6 rounded-full shadow-md transition-all duration-300"
                >
                  {isPanelVisible ? '»' : '«'}
                </button>
              )}

              {/* Conversation Panel (only visible in 2D render) */}
              {context === '2d' && isPanelVisible && (
                <div
                  className="w-150 ml-20 bg-violet-900/10 backdrop-blur-md p-6 border border-violet-500/30 rounded-xl overflow-y-auto shadow-lg h-[70vh] mt-20 "
                  style={{
                    scrollbarWidth: 'thin',
                    scrollbarColor: '#7c3aed transparent',
                  }}
                >
                  <h2 className="text-xl font-bold mb-6 text-violet-300 border-b border-violet-500/30 pb-2">
                    Conversation
                  </h2>

                  <div className="space-y-6 pb-20">
                    {conversation
                      .slice(0, currentMessageIndex + 1)
                      .filter(
                        (msg) =>
                          msg.agent !== 'InformationReturnAgent' ||
                          !msg.message.includes('TERMINATE')
                      )
                      .map((msg, index) => {
                        // Filter out unique agents excluding "InformationReturnAgent"
                        const uniqueAgents = [
                          ...new Set(
                            conversation
                              .filter((msg) => msg.agent && msg.agent !== 'InformationReturnAgent')
                              .map((msg) => msg.agent)
                          ),
                        ];

                        // Decide on avatar options (2 or 3 images based on agent count)
                        const avatarOptions =
                          uniqueAgents.length >= 3
                            ? ['agent1.png', 'agent2.png', 'agent3.png']
                            : ['agent1.png', 'agent2.png'];

                        // Render the message with avatar
                        return (
                          <div
                            key={index}
                            className="flex items-start space-x-4 animate-fadeIn"
                            style={{
                              animation: `fadeIn 0.5s ease-in-out ${index * 0.1}s both`,
                            }}
                          >
                            {/* Avatar Circle */}
                            <div className="min-w-12 w-12 h-12 flex-shrink-0 flex justify-center items-center bg-violet-600 rounded-full overflow-hidden shadow-md">
                              <img
                                src={getAgentImage(msg.agent, avatarOptions)}
                                alt={`${msg.agent} Avatar`}
                                className="w-full h-full object-cover"
                              />
                            </div>

                            {/* Message */}
                            <div className="flex-grow p-4 bg-violet-900/20 border border-violet-500/50 rounded-lg shadow-inner hover:shadow-md transition-all duration-300">
                              <p className="font-bold text-white mb-1">{msg.agent}</p>
                              <p className="text-gray-200 leading-relaxed">{msg.message}</p>
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Simulation Over Modal */}
          {(isSimulationOver || showOutcomeModal) && (
            <div className="fixed inset-0 flex items-center justify-center bg-midnight/60 bg-opacity-70 z-50 backdrop-blur-lg transition-all duration-300">
              <div className="bg-violet-900/10 backdrop-blur-3xl p-10 w-96 md:w-1/2 lg:w-1/3 border border-violet-500/50 flex flex-col justify-center rounded-xl shadow-2xl text-center transition-all duration-300 transform">
                <h2 className="text-3xl font-bold text-violet-300 mb-4">Simulation Results</h2>
                <p className="text-xl text-gray-200 mb-8">The conversation has concluded.</p>

                {/* Check for the selected run's output variables */}
                {simulationId === 'saved' ? (
                  conversationData.runs[selectedRun]?.output_variables ? (
                    <div className="mt-6 bg-violet-950/30 p-6 rounded-lg border border-violet-400/30">
                      <h3 className="font-bold text-xl text-violet-300 mb-4">Outcome</h3>
                      {/* Loop through output variables and display them dynamically */}
                      {conversationData.runs[selectedRun].output_variables.map((output, index) => (
                        <div
                          key={index}
                          className="mb-3 last:mb-0 py-2 border-b border-violet-400/20 last:border-b-0"
                        >
                          <p className="text-lg">
                            <span className="text-gray-300 font-semibold">
                              {output.name.replace('_', ' ').toUpperCase()}:
                            </span>{' '}
                            <span className="text-white">{output.value}</span>
                          </p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="mt-6 bg-violet-950/30 p-6 rounded-lg border border-violet-400/30">
                      <p className="text-gray-300">No outcome data available for this run.</p>
                    </div>
                  )
                ) : data?.runs[selectedRun]?.output_variables ? (
                  <div className="mt-6 bg-violet-950/30 p-6 rounded-lg border border-violet-400/30">
                    <h3 className="font-bold text-xl text-violet-300 mb-4">Outcome</h3>
                    {/* Loop through output variables and display them dynamically */}
                    {data.runs[selectedRun].output_variables.map((output, index) => (
                      <div
                        key={index}
                        className="mb-3 last:mb-0 py-2 border-b border-violet-400/20 last:border-b-0"
                      >
                        <p className="text-lg">
                          <span className="text-gray-300 font-semibold">
                            {output.name.replace('_', ' ').toUpperCase()}:
                          </span>{' '}
                          <span className="text-white">{output.value}</span>
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-6 bg-violet-950/30 p-6 rounded-lg border border-violet-400/30">
                    <p className="text-gray-300">No outcome data available for this run.</p>
                  </div>
                )}

                <div className="flex justify-center mt-8 space-x-4">
                  <button
                    onClick={handleRestart}
                    className="bg-violet-800 text-white text-lg px-6 py-3 rounded-full shadow-lg hover:bg-violet-700 transition-all duration-300 hover:shadow-xl cursor-pointer"
                  >
                    Restart
                  </button>

                  <button
                    onClick={() => setShowOutcomeModal(false)}
                    className="bg-gray-600 text-white text-lg px-6 py-3 rounded-full shadow-lg hover:bg-gray-700 transition-all duration-300 hover:shadow-xl cursor-pointer"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Add CSS for animations */}
          <style jsx>{`
            @keyframes fadeIn {
              from {
                opacity: 0;
                transform: translateY(10px);
              }
              to {
                opacity: 1;
                transform: translateY(0);
              }
            }

            .animate-fadeIn {
              animation: fadeIn 0.5s ease-in-out forwards;
            }
          `}</style>
        </>
      )}
    </div>
  );
};

export default Renderer;

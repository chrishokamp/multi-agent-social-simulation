import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import apiService from '../services/apiService.js';
import Navbar from './components/Navbar.jsx';

import { FaLongArrowAltRight } from 'react-icons/fa';
import { FaRandom } from 'react-icons/fa';
import { RiAiGenerate } from 'react-icons/ri';
import { MdOutlineQueuePlayNext } from 'react-icons/md';
import { FaListUl } from 'react-icons/fa6';


const TextField = ({ label, description, value, onChange, placeholder }) => {
  return (
    <label className="form-label">
      <h1 className="form-label-title">{label}</h1>
      <p className="form-label-desc">{description}</p>
      <input
        className="form-input"
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </label>
  );
};

const TextArea = ({
  label,
  description,
  value,
  onChange,
  placeholder,
  height = 'min-h-24',
  container = '',
  textSize = '',
  textArea = '',
}) => {
  return (
    <label className={`form-label ${container}`}>
      <h1 className={`form-label-title ${textSize}`}>{label}</h1>
      <p className="form-label-desc">{description}</p>
      <textarea
        className={`form-input ${height} ${textArea}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </label>
  );
};

const Button = ({ children, onClick, color = 'green', disabled = false }) => {
  const baseClasses =
    'p-2 mt-3 rounded-lg font-bold transition-colors duration-200 cursor-pointer';
  const colorStyles = {
    
  };

  return (
    <button
      className={baseClasses}
      onClick={onClick}
      disabled={disabled}
      style={colorStyles[color]}
    >
      {children}
    </button>
  );
};

const Select = ({ label, description, options, value, onChange }) => {
  return (
    <label className="form-label">
      <h1 className="form-label-title">{label}</h1>
      <p className="form-label-desc">{description}</p>
      <select
        className="form-input custom-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
};

const AgentsList = ({ agents, setAgents }) => {
  const addAgent = () => {
    setAgents([
      ...agents,
      {
        name: '',
        description: '',
        prompt: '',
      },
    ]);
  };

  const updateAgent = (index, field, value) => {
    const updatedAgents = [...agents];
    updatedAgents[index] = {
      ...updatedAgents[index],
      [field]: value,
    };
    setAgents(updatedAgents);
  };

  const removeAgent = (index) => {
    const updatedAgents = [...agents];
    updatedAgents.splice(index, 1);
    setAgents(updatedAgents);
  };

  return (
    <div className="flex flex-col p-3 mt-3 border rounded-lg  bg-transparent">
      <h1 className="font-bold text-lg">Agents</h1>
      <p className="text-gray-300 text-sm">
        Define the agents that will participate in the simulation
      </p>
      <Button onClick={addAgent}>
        Add Agent
      </Button>

      <div className="mt-3 space-y-4">
        {agents.map((agent, index) => (
          <div key={index} className="p-3 border rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <h2 className="font-semibold">Agent #{index + 1}</h2>
              <Button color="red" onClick={() => removeAgent(index)}>
                Remove
              </Button>
            </div>

            <TextField
              label="Name"
              description="The name of this agent"
              value={agent.name}
              onChange={(value) => updateAgent(index, 'name', value)}
              placeholder="e.g., Judge, Prosecutor, DefenseLawyer"
            />

            <TextField
              label="Description"
              description="Brief description of this agent's role"
              value={agent.description}
              onChange={(value) => updateAgent(index, 'description', value)}
              placeholder="e.g., Presides over the court case"
            />

            <TextArea
              label="Prompt"
              description="The prompt that defines this agent's behavior"
              value={agent.prompt}
              onChange={(value) => updateAgent(index, 'prompt', value)}
              placeholder="e.g., You are a judge presiding over a court case..."
              container="bg-transparent border"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

const OutputVariablesList = ({ variables, setVariables }) => {
  const addVariable = () => {
    setVariables([
      ...variables,
      {
        name: '',
        type: 'String',
      },
    ]);
  };

  const updateVariable = (index, field, value) => {
    const updatedVariables = [...variables];
    updatedVariables[index] = {
      ...updatedVariables[index],
      [field]: value,
    };
    setVariables(updatedVariables);
  };

  const removeVariable = (index) => {
    const updatedVariables = [...variables];
    updatedVariables.splice(index, 1);
    setVariables(updatedVariables);
  };

  return (
    <div className="flex flex-col p-3 mt-3 border rounded-lg  bg-transparent">
      <h1 className="font-bold text-lg">Output Variables</h1>
      <p className="text-gray-300 text-sm">
        Define the variables to be extracted from the simulation
      </p>
      <Button color="darkpurple" onClick={addVariable}>
        Add Variable
      </Button>

      <div className="mt-3 space-y-4">
        {variables.map((variable, index) => (
          <div key={index} className="p-3 border rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <h2 className="font-semibold">Variable #{index + 1}</h2>
              <Button color="red" onClick={() => removeVariable(index)}>
                Remove
              </Button>
            </div>

            <TextField
              label="Name"
              description="The name of this output variable"
              value={variable.name}
              onChange={(value) => updateVariable(index, 'name', value)}
              placeholder="e.g., verdict, sentence_length"
            />

            <Select
              label="Type"
              description="The data type of this variable"
              options={[
                { label: 'String', value: 'String' },
                { label: 'Number', value: 'Number' },
                { label: 'Boolean', value: 'Boolean' },
              ]}
              value={variable.type}
              onChange={(value) => updateVariable(index, 'type', value)}
            />
          </div>
        ))}
      </div>
    </div>
);
};

const AIConfigGenerator = ({ onConfigGenerated, isGenerating, setIsGenerating }) => {
  const [prompt, setPrompt] = useState('');
  const [error, setError] = useState('');

  const demoPrompts = [
    'Simulate the final moments of the O.J. Simpson trial. Include a judge, prosecutor, and defense attorney. ' +
      'Focus on the closing statements, jury verdict, and potential sentencing. Keep the conversation short, direct, ' +
      'and logical, avoiding unnecessary emotion or lengthy arguments. Track the verdict (guilty or not guilty) and ' +
      'sentence length if applicable.Keep your messages short concise and logical.',

    'Design a business negotiation simulation with a buyer, seller, and mediator. Keep your messages short concise and logical.' +
      'They are buying a house and the seller is trying to sell it. Track the final price and whether a deal was reached.',

    'Simulate a hospital examination scenario involving one doctor, one patient presenting flu-like symptoms, and one ' +
      'assisting nurse. The doctor conducts a structured interview, reviews symptoms, orders basic tests (like temperature ' +
      'check and blood test), and discusses findings with the nurse. The simulation should focus on reaching an accurate ' +
      'diagnosis (e.g., influenza, bacterial infection, or other condition) and formulating a treatment plan (e.g., ' +
      'medication, rest, further testing). Track what the diagnosis is and what is the cost of treatment without insurance, ' +
      ' as well as the appropriateness of the treatment plan. Keep your messages short concise and logical.',

    'Simulate a live political debate featuring three candidates running for national office and one moderator. ' +
      'The debate covers three key topics: healthcare, taxation, and climate policy. Each candidate presents their ' +
      'stance and responds to questions posed by the moderator, as well as rebuttals from other candidates. Track shifts ' +
      "in public opinion after each topic based on candidates' performance, clarity, and persuasiveness. Also log key " +
      'discussion points, notable arguments, and any factual inaccuracies detected during the debate. ' +
      'Keep your messages short concise and logical.',
  ];

  const getRandomPrompt = () => {
    const randomIndex = Math.floor(Math.random() * demoPrompts.length);
    setPrompt(demoPrompts[randomIndex]);
  };

  const generateConfig = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt for the AI to generate a configuration');
      return;
    }

    try {
      setError('');
      setIsGenerating(true);
      const configData = await apiService.generateSimulationConfig({ desc: prompt });
      onConfigGenerated(configData);
    } catch (err) {
      setError(`Failed to generate configuration: ${err.message}`);
      setIsGenerating(false);
    }
  };

  return (
    <div className="container p-3 mt-0 border rounded-lg relative">
      {isGenerating && (
        <div className="absolute inset-0 backdrop-blur-md flex items-center justify-center z-10 rounded-lg">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-white mb-2"></div>
            <p className="">Generating Configuration...</p>
            <p className="text-gray-400 text-sm mt-2">This may take a few moments</p>
          </div>
        </div>
      )}

      <h1 className="font-bold text-lg">Automatic Configuration</h1>
      <ul className="list-disc list-inside mt-2 pl-3">
        <li className="text-sm">
          Describe your simulation in natural language, and let us generate a configuration for you.
        </li>
        <li className="text-sm">
          Remember to include the agents, their roles, termination condition and the expected
          outcomes.
        </li>
      </ul>

      {error && (
        <div className="p-3 mb-3  rounded-lg danger">
          {error}
        </div>
      )}

      <TextArea
        label="Simulation Description"
        value={prompt}
        onChange={setPrompt}
        placeholder="e.g., Create a court case simulation with a judge, prosecutor, and defense attorney. The simulation should track the verdict and sentence length..."
        height="min-h-48"
        container="container"
        textSize="text-md"
        textArea="border focus:outline-none focus:ring-2"
      />

      <div className="flex items-center justify-center gap-3 mb-2">
        <Button onClick={getRandomPrompt}>
          <FaRandom className="mr-2 inline-block mb-0.5" />
          Random Prompt
        </Button>
        <FaLongArrowAltRight className="mt-3  h-7 w-7" />
        <Button onClick={generateConfig} disabled={isGenerating}>
          <RiAiGenerate className="mr-2 inline-block mb-0.5 h-5 w-5" />
          {isGenerating ? 'Generating Configuration...' : 'Generate Configuration'}
        </Button>
      </div>
    </div>
  );
};

const JsonPreview = ({ data }) => {
  return data ? (
    <div className="mt-4 p-3 border rounded-lg overflow-auto max-h-60">
      <h2 className="font-bold  mb-2">Generated JSON Configuration:</h2>
      <pre className="text-sm">{JSON.stringify(data, null, 2)}</pre>
    </div>
  ) : null;
};

const Tab = ({ label, isActive, onClick }) => {
  return (
    <button
      className={`tab-btn ${isActive ? 'tab-active' : 'tab-inactive'}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
};

const Configuration = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('ai');
  // Manual configuration state
  const [name, setName] = useState('');
  const [numRuns, setNumRuns] = useState(1);
  const [agents, setAgents] = useState([]);
  const [terminationCondition, setTerminationCondition] = useState('');
  const [outputVariables, setOutputVariables] = useState([]);
  const [temperature, setTemperature] = useState(1);   // 0-2
  const [topP, setTopP]             = useState(1);   // 0-1

  const [rawJson, setRawJson] = useState('');

  // Shared state
  const [simulationConfig, setSimulationConfig] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState('');

  // Load raw JSON into form
  const loadJson = () => {
    try {
      const parsed = JSON.parse(rawJson);
      if (!parsed.config || !parsed.num_runs) {
        throw new Error('Must include top-level "config" and "num_runs"');
      }
      setName(parsed.config.name || '');
      setNumRuns(parsed.num_runs);
      setAgents(parsed.config.agents || []);
      setTerminationCondition(parsed.config.termination_condition || '');
      setOutputVariables(parsed.config.output_variables || []);
      setSimulationConfig(parsed);
      setError('');
    } catch (e) {
      setError(`Invalid JSON: ${e.message}`);
    }
  };

  const handleConfigGenerated = (config) => {
    setIsGenerating(false);

    if (!config || !config.config) {
      setError('Invalid configuration generated');
      return;
    }

    // Update form fields with the generated config
    const generatedConfig = config.config;
    setName(generatedConfig.name || '');
    setNumRuns(10); // Set to 10 runs by default
    setAgents(generatedConfig.agents || []);
    setTerminationCondition(generatedConfig.termination_condition || '');
    setOutputVariables(generatedConfig.output_variables || []);

    // Set the full configuration with 10 runs
    setSimulationConfig({
      ...config,
      num_runs: 10,
      temperature: parseFloat(temperature),
      top_p: parseFloat(topP)
    });

    // Log the simulation ID
    console.log(`Retrieved simulation: ${config.id}\n`);

    // Log the simulation data in a structured format
    console.log(`Simulation Name: ${generatedConfig.name}\n`);
    console.log(`Termination Condition: ${generatedConfig.termination_condition}\n`);
    console.log(`Agents:\n`);

    // Use console.table for better visualization of agents
    console.table(generatedConfig.agents);

    // Log output variables
    console.log(`Output Variables:\n`);
    console.table(generatedConfig.output_variables);
  };

  const validateForm = () => {
    if (!name.trim()) {
      setError('Please provide a simulation name');
      return false;
    }

    if (numRuns <= 0) {
      setError('Number of runs must be greater than 0');
      return false;
    }

    if (agents.length === 0) {
      setError('Please add at least one agent');
      return false;
    }

    // Validate agents
    for (const agent of agents) {
      if (!agent.name.trim() || !agent.description.trim() || !agent.prompt.trim()) {
        setError('All agent fields must be filled out');
        return false;
      }
    }

    if (!terminationCondition.trim()) {
      setError('Please provide a termination condition');
      return false;
    }

    if (outputVariables.length === 0) {
      setError('Please add at least one output variable');
      return false;
    }

    // Validate output variables
    for (const variable of outputVariables) {
      if (!variable.name.trim()) {
        setError('All output variables must have a name');
        return false;
      }
    }

    return true;
  };

  const createSimulationConfig = () => {
    if (!validateForm()) {
      return;
    }

    const config = {
      num_runs: parseInt(numRuns),
      temperature: parseFloat(temperature),
      top_p: parseFloat(topP),
      config: {
        name: name,
        agents: agents,
        termination_condition: terminationCondition,
        output_variables: outputVariables,
      },
    };

    setSimulationConfig(config);
    setError('');
  };

  const submitSimulation = async () => {
    if (!simulationConfig) {
      createSimulationConfig();
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      const response = await apiService.createSimulation(simulationConfig);
      console.log('Simulation created:', response);
      navigate('/simulations');
    } catch (err) {
      setError(`Failed to create simulation: ${err.message}`);
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex justify-center min-h-screen py-16">
      <Navbar />
      <div className="w-full max-w-3xl px-4 pt-40">
        {/* Added pt-10 for vertical padding after Navbar */}
        <div className="flex items-center justify-between mt-6">
        </div>

        {error && (
          <div className="danger">
            {error}
          </div>
        )}

        {/* Tab navigation */}
        <div className="flex justify-center items-center">
          <Tab
            label="Automatic Configuration"
            isActive={activeTab === 'ai'}
            onClick={() => setActiveTab('ai')}
          />
          <Tab
            label="Manual Configuration"
            isActive={activeTab === 'manual'}
            onClick={() => setActiveTab('manual')}
          />
        </div>

        {/* Tab content */}
        <div className="">
          {activeTab === 'manual' && (
            <div className="container p-3 mt-0 border rounded-lg tab-connected">
              {/* Raw JSON input */}
              <TextArea
                label="Raw JSON"
                description="Paste full config here to auto-populate fields"
                value={rawJson}
                onChange={setRawJson}
                placeholder='{"num_runs":10,"config":{...}}'
                height="min-h-24"
                container=""
                textSize="text-sm"
              />
              <Button color="green" onClick={loadJson}>
                Load JSON
              </Button>

              <TextField
                label="Simulation Name"
                description="Provide a descriptive name for this simulation"
                value={name}
                onChange={setName}
                placeholder="e.g., Criminal Trial Simulation"
              />

              <TextField
                label="Number of Runs"
                description="How many times should this simulation be executed"
                value={numRuns}
                onChange={(value) => setNumRuns(value)}
                placeholder="e.g., 10"
              />

              <TextField
                label="Temperature"
                description="0 = deterministic, 2 = very random"
                value={temperature}
                onChange={setTemperature}
                placeholder="e.g., 1"
              />

              <TextField
                label="Top-p"
                description="Nucleus sampling cutoff (0-1)"
                value={topP}
                onChange={setTopP}
                placeholder="e.g., 1"
              />

              <AgentsList agents={agents} setAgents={setAgents} />

              <TextArea
                label="Termination Condition"
                description="When should the simulation end"
                value={terminationCondition}
                onChange={setTerminationCondition}
                placeholder="e.g., The judge has delivered a verdict"
                container=""
              />

              <OutputVariablesList variables={outputVariables} setVariables={setOutputVariables} />
            </div>
          )}

          {activeTab === 'ai' && (
            <AIConfigGenerator
              onConfigGenerated={handleConfigGenerated}
              isGenerating={isGenerating}
              setIsGenerating={setIsGenerating}
            />
          )}

          <JsonPreview data={simulationConfig} />

          <div className="flex flex-col md:flex-row gap-3 mt-4 items-center justify-center">
            {activeTab === 'manual' && (
              <Button onClick={createSimulationConfig}>
                <RiAiGenerate className="mr-2 inline-block mb-0.5 h-5 w-5" />
                Generate Configuration
              </Button>
            )}

            {simulationConfig && (
              <Button onClick={submitSimulation} disabled={isSubmitting}>
                <MdOutlineQueuePlayNext className="mr-2 inline-block mb-0.5 h-5 w-5" />
                {isSubmitting ? 'Creating Simulation...' : 'Run Simulation'}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Configuration;

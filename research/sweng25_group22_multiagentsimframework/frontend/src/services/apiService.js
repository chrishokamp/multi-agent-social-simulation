// API endpoint constants
const API_BASE_URL = 'http://localhost:5000/sim';

// Helper function for handling API responses
const handleResponse = async (response) => {
    if (!response.ok) {
        // Try to get error details from the response if available
        try {
            const errorData = await response.json();
            throw new Error(errorData.message || `API error: ${response.status}`);
        } catch (e) {
            throw new Error(`API error: ${response.status}`);
        }
    }

    return response.json();
};

// Make a request body for a delete request
const deleteRequestBody = (simulationId) => ({
    method: 'POST',
    headers: {
        'Content-Type': 'application/json; charset=UTF-8',
    },
    body: JSON.stringify({ id: simulationId }),
});

// Simulation API methods
const apiService = {
    // Create a new simulation
    createSimulation: async (configData) => {
        const response = await fetch(`${API_BASE_URL}/create`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(configData),
        });

        return handleResponse(response);
    },

    // Generate simulation config using LLM
    generateSimulationConfig: async (promptData) => {
        const response = await fetch(`${API_BASE_URL}/gen_config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(promptData),
        });

        return handleResponse(response);
    },

    // Get catalog of all simulations
    getSimulationsCatalog: async () => {
        const response = await fetch(`${API_BASE_URL}/catalog`);
        return handleResponse(response);
    },

    // Get simulation output/results
    getSimulationOutput: async (simulationId, options = {}) => {
        let url = `${API_BASE_URL}/results?id=${simulationId}`;

        // Add optional parameters if provided
        if (options.index !== undefined) {
            url += `&i=${options.index}`;
        }

        if (options.showMessages !== undefined) {
            url += `&show_messages=${options.showMessages ? 'yes' : 'no'}`;
        }

        const response = await fetch(url);
        return handleResponse(response);
    },

    // Delete a simulation result
    deleteSimulationResult: async (simulationId) => {
        const response = await fetch(`${API_BASE_URL}/del_results`, deleteRequestBody(simulationId));
        return handleResponse(response);
    },

    // Delete a simulation catalog
    deleteSimulationCatalog: async (simulationId) => {
        const response = await fetch(`${API_BASE_URL}/del_catalog`, deleteRequestBody(simulationId));
        return handleResponse(response);
    }
};

export default apiService;
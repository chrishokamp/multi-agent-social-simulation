# Multi-Agent Social Simulation Framework

A framework for creating and running multi-agent social simulations powered by Large Language Models (LLMs). This system enables researchers and developers to create sophisticated agent-based simulations for studying social dynamics, negotiations, and complex interactions.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **MongoDB 6.0+**
- **UV** (recommended) or pip for Python package management

## Quick Start

### 1. Setup Environment

**Option A: Using UV (recommended)**
```bash
make uv-setup
source .venv/bin/activate
```

**Option B: Using pip with virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
make install
pip install -e src/backend
```

**Option C: Using make with automatic virtual environment setup**
```bash
make venv-install
source venv/bin/activate
```


#### Environment Variable Setup

```bash
export OPENAI_API_KEY="your-api-key"
export DB_CONNECTION_STRING="mongodb://localhost:27017"

# Optional: Azure OpenAI
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-key"

# Optional: Ollama
export OLLAMA_MODEL="qwen3:4b"

# Optional: if using user interface
export DB_CONNECTION_STRING="mongodb://localhost:27017"
```


### 2. Run Simulations using the API

The main entrypoint is the `make run-simulation` command.

```bash
make run-simulation
```

The system runs a sequence of episodes. After each episode each agent
evaluates its own utility and, if ``self_improve`` is enabled for that
agent, rewrites its system prompt using the full conversation history.
The updated prompts are saved back to a ``*_optimized.json`` file
alongside ``history.json`` containing run results.

In order to run with a specific config:

```bash
make run-simulation CONFIG="path/to/your/config"
```
### 3. Run and Explore Simulations using the User Interface

### MongoDB Setup

**macOS:**
```bash
brew tap mongodb/brew
brew install mongodb-community@6.0
brew services start mongodb/brew/mongodb-community@6.0
```

**Linux/Docker:**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:6.0
```

#### Run Backend

```bash
python src/backend/main.py
```

The API will be available at `http://localhost:5000`

#### Run Frontend

```bash
cd src/frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`

## Usage Examples

### Create a Car Sale Negotiation

```python
import requests

config = {
    "name": "Car Sale Negotiation",
    "agents": [
        {
            "name": "Buyer",
            "description": "Wants to buy a car for the best price",
            "prompt": "You are a car buyer. Negotiate for the lowest price possible.",
            "utility_class": "BuyerAgent",
            "strategy": {"max_price": 25000}
        },
        {
            "name": "Seller", 
            "description": "Wants to sell car for maximum profit",
            "prompt": "You are a car salesperson. Maximize your sale price.",
            "utility_class": "SellerAgent",
            "strategy": {"target_price": 20000}
        }
    ],
    "termination_condition": "Deal reached or negotiations break down",
    "output_variables": [
        {"name": "final_price", "type": "Number"},
        {"name": "deal_reached", "type": "Boolean"}
    ]
}

# Create simulation
response = requests.post("http://localhost:5000/sim/create", json=config)
sim_id = response.json()["simulation_id"]

# Check results
results = requests.get(f"http://localhost:5000/sim/results/{sim_id}")
```

### Dynamic Configuration

You can define dynamic, randomized parameters for your simulations using the 
`variables` section in your config. This allows you to generate a new scenario 
for every run.

Each variable can be defined as:

**range**

Uniformly sample a random value

```json
"asking_price": { "range": { "min": 900, "max": 1400, "step": 50 } }
```

**expr**

Evaluate an expression at runtime

```json
"floor": { "expr": "asking_price - randint(100, 300)" }
```

Variables with `expr` can use `randint` (random.randint), `choice` (random.choice), `min`, `max`, `abs` and `round`, and reference each other, provided that the referenced variables are defined earlier in the list

```json
"variables": {
  "asking_price": { "range": { "min": 900, "max": 1400, "step": 50 } },
  "floor": { "expr": "asking_price - randint(100, 300)" },
  "budget": { "expr": "randint(floor + 50, asking_price - 50)" }
}
```

Variables can be referenced in the prompt like

```json
"You are the SELLER of a laptop.\n• Asking price: {asking_price}"
```

### Generate Configuration with AI

```python
response = requests.post("http://localhost:5000/sim/gen_config", json={
    "desc": "Create a negotiation between a landlord and tenant about rent increase",
    "temperature": 0.7
})

config = response.json()["config"]
```

## Testing

Run all tests:
```bash
make test
```

Run specific test suites:
```bash
# Backend only
make test-backend

# Unit tests only
make test-unit

# UI tests (requires Playwright)
make install-playwright  # First time only
make test-ui

# Simple UI-backend integration test (no browser)
make test-ui-simple
```

## Project Structure

```
src/
    backend/          # Python backend
        api/          # Flask REST API
        agents.py     # Agent implementations
        engine/       # Simulation engine
        db/           # Database models
        orchestrator/ # Simulation orchestrator
        tests/        # Test suite
    frontend/         # React frontend
        src/pages/    # Main application pages
        src/services/ # API integration
    configs/              # Example configurations
research/             # Research notebooks and experiments
Makefile             # Build and development commands
```

## Configuration

### Agent Types

- **UtilityAgent**: Base agent with customizable utility computation
- **BuyerAgent**: Optimizes for lowest purchase price
- **SellerAgent**: Optimizes for highest sale price
- **Custom Agents**: Extend base classes for domain-specific behavior

### Simulation Parameters

- **max_messages**: Maximum conversation length
- **min_messages**: Minimum messages for valid results
- **termination_condition**: When to end the simulation
- **output_variables**: Structured data to extract
 - **self_improve**: Set this flag on an agent to allow it to rewrite its
   own prompt between runs using the information it observed.

### Supported Models

- **OpenAI**: GPT-3.5, GPT-4, GPT-4o
- **Azure OpenAI**: All supported Azure models
- **Ollama**: Local models (Llama, Qwen, Mistral, etc.)

## Visualization

The framework includes rich visualization capabilities:
- **Conversation Flow**: Real-time message visualization
- **Analytics Dashboard**: Performance metrics and trends
- **Utility Tracking**: Agent satisfaction over time

## Research Applications

This framework is designed for:

- **Social Science Research**: Study group dynamics and decision-making
- **Economic Modeling**: Analyze market behaviors and negotiations
- **AI Safety Research**: Test agent alignment and cooperation
- **Game Theory**: Explore strategic interactions and equilibria
- **Human-AI Interaction**: Study communication patterns

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `make test`
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

For detailed documentation, see:
- [API Documentation](src/backend/api/) - REST API reference
- [Frontend Guide](src/frontend/README.md) - UI development guide

---

**Happy Simulating!**

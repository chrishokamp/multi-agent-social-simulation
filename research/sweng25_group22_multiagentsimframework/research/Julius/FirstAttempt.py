from smolagents import CodeAgent, LiteLLMModel
from smolagents import tool
from smolagents.agents import ToolCallingAgent
from smolagents import DuckDuckGoSearchTool

model = LiteLLMModel(model_id="ollama_chat/llama3.2", api_key="ollama", api_base="http://127.0.0.1:11434")

agent = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True,
    additional_authorized_imports=['numpy', 'sys', 'wikipedia', 'scipy', 'requests', 'math']
)

agent.run("Say hello")

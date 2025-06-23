import json
from types import SimpleNamespace
import pathlib, sys, types, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src" / "backend"))
class DummyAssistantAgent:
    def __init__(self, *a, **k):
        pass

autogen_stub = types.SimpleNamespace(
    agentchat=types.SimpleNamespace(AssistantAgent=DummyAssistantAgent),
    GroupChat=DummyAssistantAgent,
    GroupChatManager=DummyAssistantAgent,
    ConversableAgent=DummyAssistantAgent,
    LLMConfig=DummyAssistantAgent,
)
sys.modules.setdefault("autogen", autogen_stub)
sys.modules.setdefault("autogen.agentchat", autogen_stub.agentchat)
class DummyOpenAIClient:
    def __init__(self, *a, **k):
        pass

sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=DummyOpenAIClient, AzureOpenAI=DummyOpenAIClient))
os.environ.setdefault("OPENAI_API_KEY", "test")
from src.backend.agents import BuyerAgent

class DummyResponse:
    def __init__(self, text):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]

class DummyClient:
    def __init__(self, text="updated prompt"):
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
    def _create(self, model=None, messages=None):
        return DummyResponse(self.text)

def test_learn_from_feedback_updates_prompt():
    agent = BuyerAgent(system_prompt="initial", strategy={"max_price": 100})
    agent._client = DummyClient("new prompt")
    env = {
        "runs": [(1, {"messages": [{"agent": "Seller", "message": "Offer 80"}]})],
        "outputs": {"final_price": 80, "deal_reached": True},
    }
    agent.learn_from_feedback(0.2, env)
    assert agent.system_prompt == "new prompt"

import types
import pathlib
import sys
import types as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src" / "backend"))

class DummyAssistantAgent:
    def __init__(self, *a, **k):
        pass

autogen_stub = t.SimpleNamespace(
    agentchat=t.SimpleNamespace(AssistantAgent=DummyAssistantAgent),
    GroupChat=DummyAssistantAgent,
    GroupChatManager=DummyAssistantAgent,
    ConversableAgent=DummyAssistantAgent,
    LLMConfig=DummyAssistantAgent,
)
sys.modules.setdefault("autogen", autogen_stub)
sys.modules.setdefault("autogen.agentchat", autogen_stub.agentchat)

from engine.simulation import SelectorGCSimulation


def test_process_result_extracts_json():
    sim = SelectorGCSimulation.__new__(SelectorGCSimulation)
    sim.min_messages = 1
    chat_history = [
        types.SimpleNamespace(source="Seller", content="hello"),
        types.SimpleNamespace(source="InformationReturnAgent", content='{"final_price": 365, "deal_reached": true}\nTERMINATE')
    ]
    chat_result = types.SimpleNamespace(chat_history=chat_history)

    result = sim._process_result(chat_result)
    assert result is not None
    assert result["output_variables"][0]["name"] == "final_price"

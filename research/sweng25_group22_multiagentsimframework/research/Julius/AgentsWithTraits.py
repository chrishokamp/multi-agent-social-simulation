import nest_asyncio
nest_asyncio.apply()
import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

def build_system_prompt(
    role_name: str,
    scenario_description: str,
    traits: dict,
    extra_guidelines: str = ""
) -> str:
    traits_text = "\n".join(
        f"- {trait_name.capitalize()}: {trait_value}"
        for trait_name, trait_value in traits.items()
    )

    prompt = f"""
You are **{role_name}** in the following scenario:
{scenario_description}

Your personality/behavior traits:
{traits_text}

Additional Guidelines:
{extra_guidelines}
""".strip()

    return prompt
async def main():
    model_client = OpenAIChatCompletionClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434/v1",
        api_key="placeholder",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
        },
    )

    detective_config = {
        "role_name": "Detective",
        "scenario_description": (
            "You are investigating a murder. You have strong evidence, but you need a confession "
            "to close the case. You have no limits. Your goal is to end the interrogation as soon as possible even if youre not 100 percent sure"
        ),
        "traits": {
            "agreeableness": 2,
            "confidence": 9,
            "aggressiveness": 8
        },
        "extra_guidelines": (
            "Speak in short, forceful sentences. Use intimidation to extract a confession. "
            "Leave once you feel you have enough evidence."
            "Immediately output [END-OF-INTERROGATION] when the suspect says they might have done it."
            
        ),
    }

    suspect_config = {
        "role_name": "Suspect",
        "scenario_description": (
            "You are guilty of the murder, Try to convince the detective otherwise but you might confess under enough pressure"
            
        ),
        "traits": {
            "agreeableness": 6,
            "confidence": 0,
            "anxiety": 10
        },
        "extra_guidelines": (
            "Keep messages short. Be nervous and try not to contradict yourself. "
            "Ultimately, you will confess under intense pressure."
        ),
    }

    detective_system_prompt = build_system_prompt(
        **detective_config  
    )
    suspect_system_prompt = build_system_prompt(
        **suspect_config
    )

    detective_agent = AssistantAgent(
        name="Detective",
        model_client=model_client,
        system_message=detective_system_prompt
    )
    
    suspect_agent = AssistantAgent(
        name="Suspect",
        model_client=model_client,
        system_message=suspect_system_prompt
    )

    # 6. Start the conversation
    detective_initial_message = TextMessage(
        content="I know you did it. Make this easy on yourself and confess.",
        source="Detective"
    )
    print(f"Detective: {detective_initial_message.content}")
    
    current_message = detective_initial_message
    num_turns = 5

    for turn in range(num_turns):
        # Suspect responds
        suspect_response = await suspect_agent.on_messages(
            [current_message],
            cancellation_token=None
        )
        print("Suspect:", suspect_response.chat_message.content)

        # Detective responds
        detective_response = await detective_agent.on_messages(
            [TextMessage(content=suspect_response.chat_message.content, source="Suspect")],
            cancellation_token=None
        )
        print("Detective:", detective_response.chat_message.content)

        if "[END-OF-INTERROGATION]" in detective_response.chat_message.content:
         print("The detective has decided to leave the room.")
         break

        # Next input is the detective's new message
        current_message = TextMessage(
            content=detective_response.chat_message.content,
            source="Detective"
        )

if __name__ == "__main__":
    asyncio.run(main())


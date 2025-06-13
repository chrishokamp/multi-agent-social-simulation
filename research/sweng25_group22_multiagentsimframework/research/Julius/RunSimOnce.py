import random
import asyncio
import nest_asyncio
nest_asyncio.apply()

from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

def build_system_prompt(role_name, scenario_description, traits, extra_guidelines=""):
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

async def run_dialogue_once(detective_traits, suspect_traits) -> bool:
    """Runs one detective-suspect dialogue with given traits.
       Returns True if suspect confesses, otherwise False.
    """
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

    detective_prompt = build_system_prompt(
        role_name="Detective",
        scenario_description="You are investigating a robbery. You have strong evidence. You want a confession.",
        traits=detective_traits,
        extra_guidelines=(
            "Use intimidation. Output [END-OF-INTERROGATION] when done. "
            "Keep answers short."
        ),
    )

    suspect_prompt = build_system_prompt(
        role_name="Suspect",
        scenario_description="You are guilty of the robbery but you try to hide it.",
        traits=suspect_traits,
        extra_guidelines=(
            "Keep messages short. You will eventually confess under heavy pressure. "
            "You might say 'I confess' if pressured enough."
        ),
    )

    detective_agent = AssistantAgent(
        name="Detective",
        model_client=model_client,
        system_message=detective_prompt,
    )

    suspect_agent = AssistantAgent(
        name="Suspect",
        model_client=model_client,
        system_message=suspect_prompt,
    )

    detective_initial_message = TextMessage(
        content="I know you did it. Make this easy on yourself and confess.",
        source="Detective"
    )
    print(f"Detective: {detective_initial_message.content}")
    current_message = detective_initial_message
    suspect_confessed = False
    max_turns = 5 
    for _ in range(max_turns):
        suspect_response = await suspect_agent.on_messages(
            [current_message], cancellation_token=None )
        suspect_text = suspect_response.chat_message.content
        print("Suspect:", suspect_text)

        if "i confess" in suspect_text.lower():
            suspect_confessed = True
            break

        detective_response = await detective_agent.on_messages(
            [TextMessage(content=suspect_text, source="Suspect")], cancellation_token=None )
        detective_text = detective_response.chat_message.content
        print("Detective:", detective_text)

        if "[END-OF-INTERROGATION]" in detective_text:
            break

        current_message = TextMessage(
            content=detective_text,
            source="Detective"
        )

    return suspect_confessed

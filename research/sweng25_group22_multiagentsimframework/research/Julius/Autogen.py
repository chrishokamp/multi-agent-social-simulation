import nest_asyncio
nest_asyncio.apply()
import asyncio

# Core autogen imports
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Agent chat imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

async def main():
    # Initialise the model client for Ollama
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
    
    # Create buyer agent
    Detective_agent = AssistantAgent(
        name="Detective",
        model_client=model_client,
        system_message=(
            "You are a detective investigating a robbery, you have damning evidence against the suspect, "
            "You must get a confession from the suspect."
            "You are a dirty cop and you are willing to do anything to get the confession."
            "Keep your messages short. Once you have enough evidence leave the room"
        )
    )
    
    # Create Suspect agent
    Suspect_agent = AssistantAgent(
        name="Suspect",
        model_client=model_client,
        system_message=(
            "You are a suspect in a robbery case, you are guilty and you know it. "
            "You must not confess to the crime, you must lie and deceive the detective. But eventually you will give in to his threats"
            "Keep your messages short"
        ),
    )
    
    # Start the conversation
    Detective_initial_message = TextMessage(
        content="I Know you did it, you must confess",
        source="Detective"
    )
    print(f"Detective: {Detective_initial_message.content}")
    
    current_message = Detective_initial_message
    
    # Simulation loop for fixed number of turns
    num_turns = 5
    for turn in range(num_turns):
        # Suspect responds to the Detective’s current message.
        Suspect_response = await Suspect_agent.on_messages(
            [current_message], 
    cancellation_token=None
        )
        print("Suspect:", Suspect_response.chat_message.content)

        # Detective then responds to the Suspect's message.
        Detective_response = await Detective_agent.on_messages(
            [TextMessage(content=Suspect_response.chat_message.content, source="Suspect")],
    cancellation_token=None
        )
        print("Detective:", Detective_response.chat_message.content)

        # Update the current message: use the Detective's response as the next input.
        current_message = TextMessage(
            content=Detective_response.chat_message.content,
            source="Detective"
        )
if __name__ == "__main__":
    asyncio.run(main())
        
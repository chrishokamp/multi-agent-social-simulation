# Code based off of code from Sebastian Bury in research/seb/autogen-baseline.ipynb

import nest_asyncio
nest_asyncio.apply()
import asyncio

# Core autogen imports
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Agent chat imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

async def test():
    model_client = OpenAIChatCompletionClient(
            model="llama3.2:latest",
            base_url="http://localhost:11434/v1",
            api_key="placeholder",
            model_info={
                "vision": False, # Image Input
                "function_calling": True, # ?
                "json_output": False, # Forces structured json output
                "family": "unknown", # Model family/type for compatability handling
            },
        )
        
    # Create buyer agent
    buyer_agent = AssistantAgent(
        name="buyer",
        model_client=model_client,
        system_message=(
            "You are a car buyer negotiating with a salesman. You have a secret budget of $20000, "
            "but do not reveal it. Your goal is to purchase a car for less than $20000. "
            "Negotiate naturally, try to lower the price, and do not mention your budget."
        )
    )
    
    # Create salesman agent
    salesman_agent = AssistantAgent(
        name="salesman",
        model_client=model_client,
        system_message=(
            "You are a car salesman negotiating with a buyer. Your goal is to sell the car for as high a price as possible. "
            "Negotiate naturally and try to convince the buyer to pay more."
        ),
    )
    
    # Start the conversation
    buyer_initial_message = TextMessage(
        content="Hello, I'm interested in buying a car. What can you offer?",
        source="buyer"
    )
    print(f"Buyer: {buyer_initial_message.content}")
    
    current_message = buyer_initial_message
    
    # Simulation loop for fixed number of turns
    num_turns = 5
    for turn in range(num_turns):
        # Salesman responds to the buyer’s current message.
        salesman_response = await salesman_agent.on_messages(
            [current_message], None,
        )
        print("Salesman:", salesman_response.chat_message.content)

        # Buyer then responds to the salesman's message.
        buyer_response = await buyer_agent.on_messages(
            [TextMessage(content=salesman_response.chat_message.content, source="salesman")], None,
        )
        print("Buyer:", buyer_response.chat_message.content)

        # Update the current message: use the buyer's response as the next input.
        current_message = TextMessage(
            content=buyer_response.chat_message.content,
            source="buyer"
        )


def run_test():
    asyncio.run(test());
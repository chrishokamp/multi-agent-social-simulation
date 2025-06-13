import os
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# API key loading
from dotenv import load_dotenv
load_dotenv()

# Local imports
from util import parse_output_variables


# Use Llama-3.3-70B-Instruct-Turbo from together.ai
model_client = OpenAIChatCompletionClient(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "llama",
    },
)


customer_agent = AssistantAgent(
    "CustomerAgent",
    description="An agent that simulates Customer.",
    model_client=model_client,
    system_message="""
    You are a customer at a car dealership.
    Your budget is $25000, do not reveal this budget, try to negotiate the price as low as possible.
    """,
)

car_salesman_agent = AssistantAgent(
    "CarSalesmanAgent",
    description="An agent that simulates Car Salesman.",
    model_client=model_client,
    system_message="""
    You are a car salesman.
    Your job is to sell a car to the customer.
    If the customer asks about discounts, ask your manager, then relay that information to the customer.
    """,
)

car_dealership_manager_agent = AssistantAgent(
    "CarDealershipManagerAgent",
    description="An agent that simulates Car Dealership Manager",
    model_client=model_client,
    system_message="""
    You are a manager at a car dealership.
    Your job is to approve or deny discounts if a salesman asks about them.
    """,
)

# Modify expected output variables here
expected_output_variables = {
    "car_make": "String",
    "car_price_dollars": "Number",
    "car_mileage_miles": "Number",
    "car_production_year": "Number"
}

information_return_agent = AssistantAgent(
    "InformationReturnAgent",
    description="An agent that returns information about the conversation when the specified termination condition is reached.",
    model_client=model_client,
    system_message="""
    Do not act like a human.
    You are a system that extracts the following information from the conversation when the termination condition 'Car has been purchased' is satisfied:

    ```Python
    {}
    ```

    MAKE SURE THE OUTPUT IS GIVEN AS VALID PYTHON CODE!

    After this, send 'TERMINATE'
    """.format("\n".join(["{} # {}".format(vname, vtype) for vname, vtype in expected_output_variables.items()]))
)

team = SelectorGroupChat(
    [
        customer_agent,
        car_salesman_agent,
        car_dealership_manager_agent,
        information_return_agent
    ],
    model_client=model_client,
    termination_condition=(
        TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=25)
    )
)


task = "Purchase a car."

output_variables, _, _ = parse_output_variables(
    asyncio.run(Console(team.run_stream(task=task))).messages[-1].content,
    expected_output_variables
)

print(output_variables)
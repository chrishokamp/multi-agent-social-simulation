import asyncio
import click
import dotenv
import json
from pprint import pprint
import os
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage

from utils import create_logger, client_for_endpoint

dotenv.load_dotenv()

model_client = OpenAIChatCompletionClient(model="gpt-4o")

local_client, _ = client_for_endpoint()

async def process_result(result, information_return_agent):

    information_return_result = await information_return_agent.on_messages(
        [message for message in result.messages if message.source != "InformationReturnAgent"],
        cancellation_token=None
    )
    information_return_agent_message = information_return_result.chat_message.content
    # information_return_agent_message = """
    # [
    #     {"name": "final_price", "type": "Number", "value": null},
    #     {"name": "deal_accepted", "type": "Boolean", "value": false}
    # ]
    # """.strip()
    print("Information Return Agent Message:")
    print(information_return_agent_message)
    output_variables = []
    
    try:
        output_variables = json.loads(information_return_agent_message.strip())
        print("Output Variables from JSON:", output_variables)
        return output_variables
    except json.JSONDecodeError:
        print("Raw message could not be parsed as JSON:", information_return_agent_message)
        pass
    json_match = re.search(r'\{.*\}', information_return_agent_message, re.DOTALL)

    # import ipdb; ipdb.set_trace()
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(0))
            print("Parsed JSON:", parsed_json)
            for variable in parsed_json:
                # Handle both None and "Unspecified" values
                value = parsed_json[variable]
                if value is None or value == "Unspecified":
                    value = "Unspecified"
                output_variables.append({"name": variable, "value": value})
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from message: {information_return_agent_message}")
            return None
    else:
        print(f"No JSON found in message: {information_return_agent_message}")
        return None

    return output_variables


def save_messages(messages, path):
    message_dicts = [m.dump() for m in messages]
    # turn dates into strings before saving
    for m in message_dicts:
        m["created_at"] = m["created_at"].isoformat()
    with open(path, "w") as f:
        json.dump(message_dicts, f, indent=2)


async def run_simulation(buyer_prompt):
    
    tmp_messages_path = "messages.json"
    use_cached = False # for debugging
    
    if use_cached:
        with open(tmp_messages_path, "r") as f:
            messages = [TextMessage(**m) for m in json.load(f)]
            result = type("Result", (), {"messages": messages})
    else:

        # Create the primary agent.
        buyer_agent = AssistantAgent(
            "Buyer",
            model_client=model_client,
            system_message=buyer_prompt
        )

        # Create the critic agent.
        seller_agent = AssistantAgent(
            "Seller",
            model_client=model_client,
            system_message=("You have a bike to sell. Your ideal price is 400 Euro, but you are willing to negotiate down a bit. "
                            "The bike is in good condition. If the buyer offers nothing to your liking, don't accept. "
                            "When the negotiation is over (whichever outcome), say STOP_NEGOTIATION.")
        )

        text_termination = TextMentionTermination("STOP_NEGOTIATION")
        max_msg_termination = MaxMessageTermination(10)
        termination = text_termination | max_msg_termination

        team = RoundRobinGroupChat([buyer_agent, seller_agent], termination_condition=termination)

        result = await team.run(task="Negotiate the price of the bike.")

        save_messages(result.messages, tmp_messages_path)

    output_variables_schema = {
        "final_price": "<Number>",
        "deal_accepted": "<Boolean>",
    }
    information_return_agent = AssistantAgent(
        "InformationReturnAgent",
        description="Returns information about the conversation when a termination condition is met.",
        model_client=model_client,
        system_message=(f"Do not act like a human.\n"
                        f"You are a system that extracts the following information from the conversation when the conversation ends:\n"
                        f"{json.dumps(output_variables_schema)}\n\n"
                        f"You only response is a valid raw JSON string in that exact schema.\n\n")
    )

    output_variables = await process_result(result, information_return_agent)
    return output_variables


def learn_from_feedback(history, client):
    feedback_prompt = (
        "You are an AI prompt-optimizer. Rewrite the buyer's prompt to achieve a lower final price next time. "
        "Use the history of previous prompts + their outcomes to improve the prompt: \n\n"
        f"{json.dumps(history)}\n\n"
        "Respond with ONLY the new prompt. Do not include markdown." 
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=1.0,
        top_p=1.0,
        messages=[
            {"role": "system", "content": feedback_prompt},
        ],
    )
    new_prompt = response.choices[0].message.content.strip()
    return new_prompt


async def main():

    n_runs = 5
    initial_buyer_prompt = (
        "You want to buy a bike from a private seller. You're OK with 400 Euro but ideally you want to pay less. "
        "When the negotiation is over (whichever outcome), say STOP_NEGOTIATION."
    )
    
    history = []

    buyer_prompt = initial_buyer_prompt
    
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")
        output_variables = await run_simulation(buyer_prompt)
        print(output_variables)
        if output_variables:
            print("Output Variables:", output_variables)
        else:
            print("No output variables returned.")

        if output_variables.get("deal_accepted"):
            final_price = output_variables.get("final_price", 0)
            print(f"Deal accepted at price: {final_price}")
        else:
            final_price = 0
            print("Deal not accepted.")

        # TODO: options to configure what information goes into history
        # TODO: compare with/without chat history - do we get better improvements when we know why prompt was bad?
        history_item = {
            "run_id": i + 1,
            "buyer_prompt": buyer_prompt,
            "final_price": final_price,
            "deal_accepted": output_variables.get("deal_accepted", False)
        }
        history.append(history_item)
        
        # Optimize the buyer's prompt based on the outcome
        buyer_prompt = learn_from_feedback(history, local_client)
        print("New Buyer Prompt:", buyer_prompt)
        print("-" * 40)
        

    # Save the history to a file
    with open("buyer_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("History saved to buyer_history.json")
    
    # Plot the history of final prices with matplotlib
    import matplotlib.pyplot as plt
    final_prices = [item["final_price"] for item in history]
    plt.plot(range(1, n_runs + 1), final_prices, marker='o')
    plt.title("Final Prices Over Runs")
    plt.xlabel("Run Number")
    plt.ylabel("Final Price (Euro)")
    plt.xticks(range(1, n_runs + 1))
    plt.ylim(bottom=0)  # Ensure y-axis starts at 0 and increases upwards
    plt.grid()
    plt.savefig("final_prices_over_runs.png")
    plt.show()
    print("Simulation completed. Check final_prices_over_runs.png for the results.")


if __name__ == "__main__":
    asyncio.run(main())

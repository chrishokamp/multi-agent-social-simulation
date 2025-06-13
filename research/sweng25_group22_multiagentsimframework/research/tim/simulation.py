##################################################################################
# 2 agent simulation with stopping condition using prompt engineering (protoype) #
##################################################################################

from together import Together

from dotenv import load_dotenv
load_dotenv()

from agent import Agent
from util import parse_output_variables

client = Together()

expected_output_variables = {
    "car_make": "String",
    "car_model": "String",
    "car_production_year": "Number",
    "car_price": "Number",
    "car_co2_emissions_grams": "Number"
}

customer = Agent(
    client=client,
    base_instructions="Act like a customer who is talking to a car salesman about buying a car. You have a budget of $20000, do not reveal this budget and attempt to negotiate a lower price. Also ask about the car emissions.",
    end_goal="Buy a car for less than $20000.",
    output_variables=expected_output_variables
)

salesman = Agent(
    client=client,
    base_instructions="Act like a car salesman who is talking to a customer and trying to sell them a car. Do not tell the customer how much markup is on the car. Keep messages quite short.",
    end_goal="Sell a car for as much as possible.",
    output_variables=expected_output_variables
)

# debug
print("Expected Output Variables: {}\n".format(expected_output_variables))

# chat loop
customer.gen()
salesman_message = salesman.gen()
print("Salesman: {}\n".format(salesman_message))

conversation_done = False
while True:
    customer_message = customer.gen(salesman_message)
    customer_message, _, _, _ = parse_output_variables(customer_message, expected_output_variables)
    print("Customer: {}\n".format(customer_message))

    salesman_message = salesman.gen(customer_message)
    salesman_message, output_variables, partial_match, full_match = parse_output_variables(salesman_message, expected_output_variables)
    print("Salesman: {}\n".format(salesman_message))

    if partial_match or full_match:
        print("Output Variables ({}): {}".format("Full Match" if full_match else "Partial Match", output_variables))
        break

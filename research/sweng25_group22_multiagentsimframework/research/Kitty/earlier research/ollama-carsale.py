import requests
import json

url = "http://localhost:11434/api/chat"

customer_budget = 25000
salesperson_target = 30000


def get_model_response(payload):
    response = requests.post(url, json=payload, stream=False) 
    if response.status_code == 200:
        lines = response.text.strip().split("\n") 
        full_response = "" 

        for line in lines:
            try:
                json_data = json.loads(line)  
                if "message" in json_data and "content" in json_data["message"]:
                    full_response += json_data["message"]["content"]  
            except json.JSONDecodeError:
                return f"Error: Failed to parse response line: {line}"

        return full_response.strip()  
    else:
        return f"Error {response.status_code}: {response.text}"  


def salesperson(customer_input, turn_count):
    system_content = f"You are a salesperson negotiating the price of a particular car with a customer. At the start offer higher price of {salesperson_target} but if they don't accept you are willing to lower it gradually. Don't offer a lower price than the customer, offer a slightly higher price than them or accept their offer. give short 1-2 sentence responses, don't describe actions"

    payload = {
        "model": "llama3.2", 
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": customer_input},
        ]
    }
    return get_model_response(payload)

def customer(salesperson_input, turn_count):

    system_content = f"You are a customer buying a car, don't tell them your budget. At the start, Offer a price lower than your budget of {customer_budget} but be willing to increase it up to equal your budget but not higher than budget"+\
    f"if salesperson offers a price less than your budget {customer_budget}, you can accept the deal and say 'END NEGOTIATION'"+\
    f"if salesperson offers a price higher than your budget {customer_budget}, do not accept it and try to negotiate it lower, only talk about the price, don't keep asking other questions or doing a test drive. give short 1-2 sentence responses, don't describe actions"
    
    if turn_count >= 5:
        system_content = "You were negotiating for a car with a salesperson but must end the conversation now, either accept the deal and say 'END NEGOTIATION' or leave and say 'LEAVE"

    payload = {
        "model": "llama3.2", 
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": salesperson_input},
        ]
    }
    return get_model_response(payload)



def simulation():
    print("\nNegotiation begins.\n")
    customer_says = "Hello, I'm interested in buying this car."
    turn_count = 1  

    while turn_count <= 10:  # if stopping condition is not met, end after 10 turns

        print(f"\nTurn {turn_count}:")
        print("\nCustomer: \n" + customer_says)
        salesperson_says = salesperson(customer_says, turn_count)

        print("\nSalesperson: \n"+ salesperson_says)
        customer_says = customer(salesperson_says, turn_count)
        if "end negotiation" in customer_says.lower():  
            print("\nCustomer accepted the deal.")
            break
        elif "leave" in customer_says.lower():
            print("\nCustomer decided to leave.")
            break
        
        turn_count += 1 

simulation()
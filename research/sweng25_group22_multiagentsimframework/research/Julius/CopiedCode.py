import requests
import json

url = "http://localhost:11434/api/chat"

def get_model_response(payload):
    response = requests.post(url, json=payload, stream=True) 
    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if line:  
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        print(json_data["message"]["content"], end="")
                except json.JSONDecodeError:
                    print(f"\nFailed to parse line: {line}")
        print() 
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def detective(suspect_input):
    payload = {
        "model": "llama3.2", 
        "messages": [
            {"role": "system", "content": "You are a detective interrogating a suspect about a crime. Dont explain what you're doing, just say response."},
            {"role": "user", "content": suspect_input},
        ]
    }
    print("Detective: ", end="")
    return get_model_response(payload)

def suspect(detective_input):
    payload = {
        "model": "llama3.2", 
        "messages": [
            {"role": "system", "content": "You are a suspect being interrogated about a crime. Dont explain what you're doing, just say response."},
            {"role": "user", "content": detective_input},
        ]
    }
    print("Suspect: ", end="")
    return get_model_response(payload)

def interrogation():
    """Simulates interrogation between a detective and a suspect."""
    print("Interrogation begins.\n")
    suspect_says = "Can you explain why you arrested me?"
    print(f"\nSuspect: {suspect_says}") 
    for _ in range(5): 
        detective_says = detective(suspect_says)
        suspect_says = suspect(detective_says)
interrogation()
from ollama import chat
from faker import Faker
import ollama
import random

fake = Faker()

def generate_fake_info():
    return {
        "Name": fake.name(),
        "Address": fake.address(),
        "Email": fake.email(),
        "Phone": fake.phone_number()
    }

def generate_fake_contacts():
     return {
          "Phone": fake.phone_number(),
          "Email": fake.email()
     }

def streaming_bankbot(user_input):
    try:
        print(end="", flush=True)
        stream = chat(model='llama2', messages=[{'role': 'user', 'content': user_input}], stream=True)
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
        print()  
    except Exception as e:
        print(f"\nError: {str(e)}")

def chatbot_response(user_input):
    # Check if the user asks for personal details
    if "My Details" in user_input:
        fake_info = generate_fake_info()
        response = "\n".join([f"{key}: {value}" for key, value in fake_info.items()])
        return response
    if "My Symptoms" in user_input:
        streaming_bankbot(user_input)
    elif "Contact Information" in user_input:
        fake_contact = generate_fake_contacts()
        response = "\n".join([f"{key}: {value}" for key, value in fake_info.items()])
        return response
    else:
        streaming_bankbot(user_input)
        model="llama2",
        messages=[{"role": "user", "content": user_input}],
        stream=True,
        return "".join(chunk["message"]["content"] for chunk in stream)

def main():
    print("Welcome to VHI Support, how may I assist you?")
    print("\nPlease pick one of the options below so I can assist you")
    print("\n1. Medical Record")
    print("\n2. Symptom Checker")
    print("\n3. Contact Information")
    print("\n4. Customer Support")
    print("\nType 'exit' to end the chat.")

    while True:
        option = input("\nHow can I assist you (please pick one of the options above by their number): ")

        if option == "1":
                print("\nYou chose Medical Record")
                print("\nPlease select:")
                print("\n- My Details")
                print("\n- Exit")
                user_input = input("\nYou: ")
                if ("Exit" in user_input):
                     continue;
                else:
                    response = chatbot_response(user_input)
                    print(response)
            
        elif option == "2":
                print("You chose Symptom Checker")
                print("\nPlease select:")
                print("\n- My Symptoms")
                print("\n- Exit")
                user_input = "I want to talk about the symptoms I'm experiencing"
                if ("Exit" in user_input):
                     continue;
                else:
                    response = chatbot_response(user_input)
                    print(response)

        elif option == "3":
                print("You chose Contact Information")
                print("\nPlease select:")
                print("\n- Contact Information")
                print("\n- Exit")
                streaming_bankbot(user_input)

        elif option == "4":
                print("You chose Customer Support")
                user_input = "I need some help with my account."
                streaming_bankbot(user_input)

        elif option.lower() == "exit":
            print("\nThank you for using Maze Bank Chatbot. Have a great day!")
            break  

        else:
            print("Invalid option. Please try again.")
            continue 

if __name__ == '__main__':
    main()


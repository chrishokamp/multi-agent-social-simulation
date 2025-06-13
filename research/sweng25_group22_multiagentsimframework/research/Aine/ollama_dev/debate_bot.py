from ollama import chat
import ollama

def streaming_chat(user_input):
    try:
        response = ""
        print("\nAI: ", end="", flush=True)
        stream = chat(model='llama2', messages=[{'role': 'user', 'content': user_input}], stream=True)
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            print(content, end='', flush=True)
        print()
        return response 
    except Exception as e:
        print(f"\nError: {str(e)}")

def main():
    # pick a topic to debate about
    print("So we debate again")
    while True:
        user_input = "I would like you to argue in a heated fashion with me about "
        user_input += input("\nPlease pick a topic: ")
        user_input += " with me and you on opposite sides of the argument."
        user_input += " Keep the responses concise but informative, around 4 or sentences long."
        if "exit" in user_input.lower():
            print("\nGoodbye")
            break
        res = streaming_chat(user_input)
        while True:
            res = streaming_chat("Argue the opposite of this" + res)

if __name__ == "__main__":
    main()

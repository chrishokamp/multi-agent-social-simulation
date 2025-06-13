import openai
import promptlayer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI and PromptLayer API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
promptlayer.api_key = os.getenv("PROMPTLAYER_API_KEY")

# Custom subclass to inject PromptLayer tags
class PromptLayerOpenAIClient:
    def __init__(self, model, api_key=None):
        print(f"PromptLayerOpenAIClient initialized with model: {model}")
        self.model = model
        self.api_key = api_key

    def generate(self, messages, **kwargs):
        print(f"Calling generate method for model: {self.model}")
        
        # OpenAI call
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        return response

    def set_api_key(self, api_key):
        self.api_key = api_key
        print(f"API key set to: {api_key}")

# Example usage
def example_usage():
    # Initialize the client with the desired model
    client = PromptLayerOpenAIClient(model="gpt-4")

    # Call the generate method with a sample message
    response = client.generate(messages=[{"role": "user", "content": "Hello, how are you?"}])
    print("Response from generate:", response)

    # Optionally, change the API key if needed
    client.set_api_key("new_api_key_here")

# Run the example usage function to test
example_usage()

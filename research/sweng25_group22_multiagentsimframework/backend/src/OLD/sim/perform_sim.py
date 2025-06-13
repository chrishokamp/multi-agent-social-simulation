import openai
import promptlayer
import os
from dotenv import load_dotenv
import openai        # Import the wrapped OpenAI client

#openai = promptlayer.openai

# Load environment variables
load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
promptlayer.api_key = os.getenv("PROMPTLAYER_API_KEY")

# Custom subclass to use PromptLayer OpenAI tracking
class PromptLayerOpenAIClient:
    def __init__(self, model, api_key=None):
        print(f"PromptLayerOpenAIClient initialized with model: {model}")
        self.model = model
        self.api_key = api_key or openai.api_key  # Use provided or default API key

    def generate(self, messages, **kwargs):
        print(f"Calling generate method for model: {self.model}")
        
        # Use promptlayer's OpenAI wrapper
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        # Log performance details
        usage = response.get("usage", {})
        print(f"Tokens Used - Prompt: {usage.get('prompt_tokens', 0)}, Completion: {usage.get('completion_tokens', 0)}, Total: {usage.get('total_tokens', 0)}")
        
        return response

    def set_api_key(self, api_key):
        self.api_key = api_key
        print(f"API key set to: {api_key}")

# Example usage
def example_usage():
    # Initialize the client with the desired model
    client = PromptLayerOpenAIClient(model="gpt-4")

    # Call the generate method with a sample message
    response = client.generate(messages=[{"role": "user", "content": "You're a car salesperson"}])
    print("Response from generate:", response["choices"][0]["message"]["content"])

# Run the example usage function
if __name__ == "__main__":
    example_usage()
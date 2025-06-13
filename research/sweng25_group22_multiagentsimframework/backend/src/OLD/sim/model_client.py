from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

def init_gpt_model_client():
    return OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"]
        )   

def init_together_model_client():
    return OpenAIChatCompletionClient(
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

def init_ollama_model_client():
    return OpenAIChatCompletionClient(
            model="llama3.2:latest",
            base_url="http://localhost:11434/v1",
            api_key="placeholder",
            model_info={
                "vision": False, 
                "function_calling": True, 
                "json_output": False, 
                "family": "llama", 
            },
        )

def select_model(type: str, gpt_model, together_model, ollama_model):
    if (type == "GPT"):
        return gpt_model
    elif (type == "Together"):
        return together_model
    elif (type == "Ollama"):
        return ollama_model
    raise Exception("Invalid model type: " + type)

def init_model_clients(sim):
    models = sim.config['models']
    model_list = [ models["agent"], models["info_return_agent"], models["group_chat"]]
    gpt_model_client = None
    together_model_client = None
    ollama_model_client = None

    if (model_list.__contains__("GPT")):
        gpt_model_client = init_gpt_model_client()
    if (model_list.__contains__("Together")):
        together_model_client = init_together_model_client()
    if (model_list.__contains__("Ollama")):
        ollama_model_client = init_ollama_model_client()

    sim.agent_model_client = select_model(models["agent"], gpt_model_client, together_model_client, ollama_model_client)
    sim.info_model_client = select_model(models["info_return_agent"], gpt_model_client, together_model_client, ollama_model_client)
    sim.gc_model_client = select_model(models["group_chat"], gpt_model_client, together_model_client, ollama_model_client)
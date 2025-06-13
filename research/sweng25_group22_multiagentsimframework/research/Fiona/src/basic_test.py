import ollama
import settings
import utils

def run_test():
    print("Running Basic Test");
    response = ollama.generate(model=settings.OLLAMA_MODEL, prompt="Explain the most famous RDNA3 instructions");
    utils.print_response(response);


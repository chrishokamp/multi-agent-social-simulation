import os
import multiprocessing

from dotenv import load_dotenv
load_dotenv()

if not "OPENAI_API_KEY" in os.environ:
    print("OPENAI_API_KEY required!")
    
if not "DB_CONNECTION_STRING" in os.environ:
    print("DB_CONNECTION_STRING required!")

from orchestrator.simulation_orchestrator import orchestrator
from api.app import app

def run_api():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    api_process = multiprocessing.Process(target=run_api)
    api_process.start()
    try:
        orchestrator()
    finally:
        api_process.terminate()
        api_process.join()

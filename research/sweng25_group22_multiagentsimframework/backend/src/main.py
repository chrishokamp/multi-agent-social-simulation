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

def run_simulation_orchestrator():
    orchestrator(max_threads=4)


if __name__ == "__main__":
    simulation_orchestrator_process = multiprocessing.Process(target=run_simulation_orchestrator)
    api_process = multiprocessing.Process(target=run_api)

    simulation_orchestrator_process.start()
    api_process.start()

    simulation_orchestrator_process.join()
    api_process.join()
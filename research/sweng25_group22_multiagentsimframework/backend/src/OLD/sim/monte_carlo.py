import asyncio
import numpy as np
import re

from OLD.sim.simulation import Simulation

class MonteCarloSimulator:
    def __init__(self, sim_config_file_name, num_runs=5):
        self.num_runs = num_runs
        self.sim_config_file_name = sim_config_file_name
        self.results = []

    async def run_single_simulation(self, run_id):
        print(f"\nRunning Simulation {run_id+1}/{self.num_runs}...\n")
        sim = Simulation(self.sim_config_file_name)
        return await sim.run()

    async def run_monte_carlo(self):
        tasks = [self.run_single_simulation(i) for i in range(self.num_runs)]
        self.results = await asyncio.gather(*tasks)
        return self.results
    
    # Is this zombie code below being used? Remove?

    # def analyze_results(self):
    #     success_rate = 100

    #     print(f"\nMonte Carlo Results: {self.num_runs} runs completed.")
    #     print(f"Termination Rate: {success_rate:.2f}%\n")

    #     return {"num_runs": self.num_runs, "success_rate": success_rate}

    # Deprecated? Remove?
    def parse_output_variables(self):
        results = {}
        
        lines = message.splitlines()
        filtered_lines = []
        
        for line in lines:
            if not any(re.search(rf"{vname}\s*=", line) for vname in output_variables):
                filtered_lines.append(line)
            
        full_match = True
        partial_match = False

        for vname, vtype in output_variables.items():
            str_pattern = rf"{vname}\s*=\s*\"(.*?)\""
            num_pattern = rf"{vname}\s*=\s*(\d+)"
            
            str_match = re.search(str_pattern, message)
            num_match = re.search(num_pattern, message)
            
            if vtype == "String":
                results[vname] = str_match.group(1) if str_match else None
            elif vtype == "Number":
                results[vname] = int(num_match.group(1)) if num_match else None

            if results[vname] == None:
                full_match = False
            else:
                partial_match = True
        
        return results, partial_match, full_match
    
# Is this zombie code below being used? Remove?

# async def main():
#     monte_carlo = MonteCarloSimulator("car_sale_simulation.json", num_runs=1)
#     results = await monte_carlo.run_monte_carlo()
#     stats = monte_carlo.analyze_results()
#     print(stats)

# if __name__ == "__main__":
#     asyncio.run(main())

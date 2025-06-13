import random
import asyncio
from RunSimOnce import run_dialogue_once 

async def run_monte_carlo(num_runs: int = 100):
    """Runs the detective-suspect interrogation multiple times with random traits.
       Returns how many times suspect confessed.
    """
    confessions = 0

    for i in range(num_runs):
       
        detective_traits = {
            "agreeableness": random.randint(1, 10),
            "confidence": random.randint(1, 10),
            "aggressiveness": random.randint(1, 10)
        }

       
        suspect_traits = {
            "agreeableness": random.randint(1, 10),
            "confidence": random.randint(1, 10),
            "anxiety": random.randint(1, 10)
        }

        print(f"\n--- Simulation #{i+1} ---")
        print(f"Detective Traits: {detective_traits}")
        print(f"Suspect Traits:   {suspect_traits}")
        
       
        confessed = await run_dialogue_once(detective_traits, suspect_traits)

        
        if confessed:
            confessions += 1
            print("Suspect confessed!")
        else:
            print("Suspect did NOT confess.")

    return confessions

if __name__ == "__main__":
    total_runs = 5
    result = asyncio.run(run_monte_carlo(total_runs))
    print(f"Out of {total_runs} runs, suspect confessed {result} times.")

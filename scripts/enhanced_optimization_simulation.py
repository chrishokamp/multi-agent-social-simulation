"""
Enhanced optimization simulation script using the new optimization framework.

This script integrates bandit algorithms, prompt templates, meta-learning,
and comprehensive tracking for advanced self-optimization of agent prompts.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from pprint import pprint

import click
import dotenv
import numpy as np

import sys

# Allow running without installation by adjusting PYTHONPATH
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src" / "backend"))

from engine.simulation import SelectorGCSimulation
from optimization.enhanced_optimizer import EnhancedOptimizer, OptimizationConfig
from optimization.meta_learner import SimulationContext

dotenv.load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimulationUtilityEvaluator:
    """Evaluates utility of prompts by running simulations."""
    
    def __init__(self, base_config: Dict[str, Any], target_agent: str, 
                 max_messages: int = 10, min_messages: int = 1, model: str = None):
        self.base_config = base_config.copy()
        self.target_agent = target_agent
        self.max_messages = max_messages
        self.min_messages = min_messages
        self.model = model
        self.environment = {"runs": [], "outputs": {}}
        self.run_count = 0
        
    async def evaluate_utility(self, prompt: str) -> float:
        """Evaluate utility of a prompt by running a simulation."""
        try:
            # Update target agent's prompt
            config = self._update_agent_prompt(self.base_config.copy(), prompt)
            
            # Run simulation
            sim = SelectorGCSimulation(
                config,
                environment=self.environment.copy(),
                max_messages=self.max_messages,
                min_messages=self.min_messages,
                model=self.model,
            )
            
            result = await sim.run()
            
            if not result:
                logger.warning("Simulation returned no result")
                return 0.0
            
            # Calculate utility based on agent performance
            utility = self._calculate_utility(result, sim)
            
            self.run_count += 1
            logger.info(f"Run {self.run_count}: Utility = {utility:.4f}")
            
            return utility
            
        except Exception as e:
            logger.error(f"Error evaluating utility: {e}")
            return 0.0
    
    def _update_agent_prompt(self, config: Dict[str, Any], new_prompt: str) -> Dict[str, Any]:
        """Update the target agent's prompt in the configuration."""
        for agent_config in config["agents"]:
            if agent_config["name"] == self.target_agent:
                agent_config["prompt"] = new_prompt
                break
        return config
    
    def _calculate_utility(self, result: Dict[str, Any], sim) -> float:
        """Calculate utility based on simulation results."""
        # Extract output variables
        outputs = {var["name"]: var["value"] for var in result["output_variables"]}
        
        # Find the target agent
        target_agent = None
        for agent in sim.agents:
            if agent.name == self.target_agent:
                target_agent = agent
                break
        
        if not target_agent:
            logger.warning(f"Target agent {self.target_agent} not found")
            return 0.0
        
        # Use agent's built-in utility function if available
        if hasattr(target_agent, 'compute_utility'):
            try:
                agent_utility = target_agent.compute_utility(outputs)
                return max(0.0, min(1.0, agent_utility))
            except Exception as e:
                logger.warning(f"Error computing agent utility: {e}")
        
        # Fallback utility calculation based on common metrics
        utility = 0.5  # Default neutral utility
        
        # Negotiation-specific metrics
        if "final_price" in outputs and "deal_reached" in outputs:
            if outputs.get("deal_reached", False):
                utility += 0.3  # Bonus for reaching a deal
                
                # Agent-specific utility based on role
                if hasattr(target_agent, 'strategy') and isinstance(target_agent.strategy, dict):
                    if "max_price" in target_agent.strategy:  # Buyer
                        max_price = target_agent.strategy["max_price"]
                        final_price = outputs.get("final_price", max_price)
                        if final_price <= max_price:
                            utility += 0.2 * (1 - final_price / max_price)
                    elif "target_price" in target_agent.strategy:  # Seller
                        target_price = target_agent.strategy["target_price"]
                        final_price = outputs.get("final_price", 0)
                        if final_price >= target_price:
                            utility += 0.2 * (final_price / target_price - 1)
        
        # Conversation quality metrics
        if "messages" in result:
            messages = result["messages"]
            agent_messages = [msg for msg in messages if msg.get("name") == self.target_agent]
            
            if agent_messages:
                # Reward appropriate message length
                avg_length = np.mean([len(msg.get("content", "")) for msg in agent_messages])
                if 50 <= avg_length <= 300:  # Reasonable message length
                    utility += 0.1
                
                # Reward varied vocabulary (simple heuristic)
                all_words = []
                for msg in agent_messages:
                    all_words.extend(msg.get("content", "").lower().split())
                
                if all_words:
                    unique_words = len(set(all_words))
                    total_words = len(all_words)
                    vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
                    utility += 0.1 * vocabulary_diversity
        
        return max(0.0, min(1.0, utility))


async def main(config_path: Path, target_agent: str, algorithm: str, max_iterations: int,
               exploration_factor: float, enable_meta_learning: bool, enable_templates: bool,
               max_messages: int, min_messages: int, results_dir: str, experiment_name: str):
    """Run enhanced optimization on agent prompts."""
    
    logger.info("Starting enhanced optimization simulation")
    
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    
    config = raw_config.get("config", raw_config)
    model = raw_config.get("model") or config.get("model")
    
    # Find target agent and get initial prompt
    target_agent_config = None
    initial_prompt = None
    
    for agent_config in config["agents"]:
        if agent_config["name"] == target_agent:
            target_agent_config = agent_config
            initial_prompt = agent_config.get("prompt", "You are a helpful assistant.")
            break
    
    if not target_agent_config:
        raise click.ClickException(f"Target agent '{target_agent}' not found in configuration")
    
    logger.info(f"Optimizing agent: {target_agent}")
    logger.info(f"Initial prompt length: {len(initial_prompt)} characters")
    
    # Create simulation context for meta-learning
    simulation_context = None
    if enable_meta_learning:
        simulation_context = SimulationContext(
            simulation_type="negotiation",  # Infer from config or make configurable
            domain="business",  # Could be extracted from config
            objectives=[],  # Could be extracted from agent strategy
            constraints=[],
            agent_types=[agent["name"] for agent in config["agents"]],
            output_variables=[var["name"] for var in config.get("output_variables", [])]
        )
    
    # Setup optimization configuration
    opt_config = OptimizationConfig(
        algorithm=algorithm,
        exploration_factor=exploration_factor,
        max_iterations=max_iterations,
        enable_meta_learning=enable_meta_learning,
        enable_prompt_templates=enable_templates,
        results_dir=results_dir
    )
    
    # Create optimizer
    if not experiment_name:
        experiment_name = f"optimize_{target_agent}_{algorithm}"
    
    optimizer = EnhancedOptimizer(opt_config, experiment_name)
    
    # Create utility evaluator
    evaluator = SimulationUtilityEvaluator(
        config, target_agent, max_messages, min_messages, model
    )
    
    # Run optimization
    logger.info("Starting optimization process...")
    
    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        utility_function=evaluator.evaluate_utility,
        context=simulation_context
    )
    
    # Display results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Algorithm: {algorithm}")
    print(f"Target Agent: {target_agent}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Converged: {result.converged} ({result.convergence_reason})")
    print(f"Best Utility: {result.best_utility:.4f}")
    print(f"Total Improvement: {result.final_statistics.get('total_improvement', 0):.4f}")
    
    print(f"\nBest Prompt ({len(result.best_prompt)} characters):")
    print("-" * 50)
    print(result.best_prompt)
    print("-" * 50)
    
    # Show learned patterns if meta-learning was enabled
    if result.learned_patterns:
        print(f"\nTop Learned Patterns ({len(result.learned_patterns)}):")
        for i, pattern in enumerate(result.learned_patterns[:5], 1):
            print(f"{i}. Success Rate: {pattern.success_rate:.3f}, "
                  f"Usage: {pattern.usage_count}, "
                  f"Domains: {list(pattern.domain_tags)}")
            print(f"   Pattern: {pattern.content[:100]}...")
    
    # Show final statistics
    print(f"\nOptimization Statistics:")
    pprint(result.final_statistics)
    
    # Update original config with best prompt
    print(f"\nUpdating configuration file with best prompt...")
    
    # Update the target agent's prompt in the original config
    for agent_config in config["agents"]:
        if agent_config["name"] == target_agent:
            agent_config["prompt"] = result.best_prompt
            break
    
    # Save updated configuration
    output_config_path = config_path.with_name(f"{config_path.stem}_optimized.json")
    raw_config["config"] = config
    
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(raw_config, f, indent=2)
    
    print(f"Updated configuration saved to: {output_config_path}")
    
    # Save optimization history
    history_path = Path(results_dir) / f"{experiment_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(result.optimization_history, f, indent=2)
    
    print(f"Optimization history saved to: {history_path}")
    print(f"Full results and plots saved to: {results_dir}/")
    
    logger.info("Enhanced optimization completed successfully!")


def sync_main():
    """Synchronous wrapper for the async main function."""
    import asyncio
    
    @click.command()
    @click.option(
        "--config",
        "config_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        required=True,
        help="Path to simulation configuration JSON",
    )
    @click.option("--target-agent", required=True, help="Name of agent to optimize")
    @click.option("--algorithm", default="ucb", type=click.Choice(["ucb", "thompson_sampling"]), 
                  help="Optimization algorithm")
    @click.option("--max-iterations", default=30, show_default=True, help="Maximum optimization iterations")
    @click.option("--exploration-factor", default=1.0, show_default=True, help="Exploration factor for bandit algorithm")
    @click.option("--enable-meta-learning", is_flag=True, default=True, help="Enable meta-learning")
    @click.option("--enable-templates", is_flag=True, default=True, help="Enable prompt templates")
    @click.option("--max-messages", default=10, show_default=True, help="Maximum conversation length")
    @click.option("--min-messages", default=1, show_default=True, help="Minimum messages for valid result")
    @click.option("--results-dir", default="optimization_results", help="Directory to save results")
    @click.option("--experiment-name", help="Custom experiment name")
    def cli_main(config_path: Path, target_agent: str, algorithm: str, max_iterations: int,
                 exploration_factor: float, enable_meta_learning: bool, enable_templates: bool,
                 max_messages: int, min_messages: int, results_dir: str, experiment_name: str):
        """CLI wrapper for the async main function."""
        return asyncio.run(main(
            config_path, target_agent, algorithm, max_iterations,
            exploration_factor, enable_meta_learning, enable_templates,
            max_messages, min_messages, results_dir, experiment_name
        ))
    
    cli_main()


if __name__ == "__main__":
    sync_main()
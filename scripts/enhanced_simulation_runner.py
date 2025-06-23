"""
Enhanced simulation runner with integrated optimization framework.

This script combines the rich logging framework with the enhanced optimization
to provide a comprehensive simulation experience via `make run-simulation`.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys

import click
import dotenv

# Setup path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src" / "backend"))

from engine.simulation import SelectorGCSimulation
from optimization.enhanced_optimizer import EnhancedOptimizer, OptimizationConfig
from optimization.meta_learner import SimulationContext

# Import logging framework if available
try:
    from logging_framework import (
        SimulationLogger, AgentLogger, MetricsCollector,
        HTMLReporter
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    logging.warning("Rich logging framework not available, using basic logging")

dotenv.load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedSimulationRunner:
    """Enhanced simulation runner with optimization and logging."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.optimization_enabled = self._check_optimization_enabled()
        self.logging_enabled = LOGGING_AVAILABLE and self.config.get('rich_logging', {}).get('enabled', True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load simulation configuration."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _check_optimization_enabled(self) -> bool:
        """Check if enhanced optimization is enabled."""
        return self.config.get('enhanced_optimization', {}).get('enabled', False)
    
    def _get_optimization_target_agent(self) -> Optional[str]:
        """Get the agent marked for optimization."""
        for agent in self.config['config']['agents']:
            if agent.get('optimization_target', False):
                return agent['name']
        
        # Fallback to first agent with self_improve
        for agent in self.config['config']['agents']:
            if agent.get('self_improve', False):
                return agent['name']
        
        return None
    
    def _create_simulation_context(self) -> Optional[SimulationContext]:
        """Create simulation context for meta-learning."""
        sim_context_config = self.config.get('simulation_context', {})
        
        if not sim_context_config:
            return None
        
        return SimulationContext(
            simulation_type=sim_context_config.get('type', 'negotiation'),
            domain=sim_context_config.get('domain', 'general'),
            objectives=sim_context_config.get('objectives', []),
            constraints=sim_context_config.get('constraints', []),
            agent_types=[agent['name'] for agent in self.config['config']['agents']],
            output_variables=[var['name'] for var in self.config['config'].get('output_variables', [])]
        )
    
    async def run_basic_simulation(self) -> Dict[str, Any]:
        """Run basic simulation without optimization."""
        logger.info("🎯 Running basic simulation...")
        
        config = self.config.get('config', self.config)
        model = self.config.get('model') or config.get('model')
        num_runs = self.config.get('num_runs', 5)
        
        environment = {"runs": [], "outputs": {}}
        all_results = []
        
        for run_idx in range(1, num_runs + 1):
            logger.info(f"📊 Run {run_idx}/{num_runs}")
            
            sim = SelectorGCSimulation(
                config,
                environment=environment,
                max_messages=10,
                min_messages=1,
                model=model,
            )
            
            result = await sim.run()
            
            if result:
                outputs = {var["name"]: var["value"] for var in result["output_variables"]}
                all_results.append({
                    "run_id": run_idx,
                    "result": result,
                    "outputs": outputs
                })
                
                environment["runs"].append((run_idx, {"messages": result["messages"]}))
                environment["outputs"] = outputs
                
                # Update agent prompts if self-improvement is enabled
                for agent in sim.agents:
                    for cfg in config["agents"]:
                        if cfg["name"] == agent.name and cfg.get("self_improve", False):
                            cfg["prompt"] = getattr(agent, "system_prompt", cfg.get("prompt"))
                
                logger.info(f"✅ Run {run_idx} completed: {outputs}")
            else:
                logger.warning(f"❌ Run {run_idx} failed")
        
        return {
            "config": config,
            "results": all_results,
            "total_runs": num_runs,
            "successful_runs": len(all_results)
        }
    
    async def run_enhanced_optimization(self) -> Dict[str, Any]:
        """Run enhanced optimization simulation."""
        logger.info("🚀 Running enhanced optimization simulation...")
        
        # Get optimization settings
        opt_config_data = self.config.get('enhanced_optimization', {})
        target_agent = self._get_optimization_target_agent()
        
        if not target_agent:
            raise ValueError("No optimization target agent found. Set 'optimization_target': true for an agent.")
        
        logger.info(f"🎯 Optimizing agent: {target_agent}")
        
        # Create optimization configuration
        opt_config = OptimizationConfig(
            algorithm=opt_config_data.get('algorithm', 'ucb'),
            exploration_factor=opt_config_data.get('exploration_factor', 1.2),
            max_iterations=opt_config_data.get('max_iterations', 15),
            convergence_patience=opt_config_data.get('convergence_patience', 3),
            utility_threshold=opt_config_data.get('utility_threshold', 0.85),
            enable_meta_learning=opt_config_data.get('enable_meta_learning', True),
            enable_prompt_templates=opt_config_data.get('enable_prompt_templates', True),
            save_results=opt_config_data.get('save_detailed_results', True),
            results_dir=opt_config_data.get('results_dir', 'optimization_results')
        )
        
        # Create experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"simulation_{target_agent}_{timestamp}"
        
        # Create optimizer
        optimizer = EnhancedOptimizer(opt_config, experiment_name)
        
        # Create simulation context
        context = self._create_simulation_context()
        
        # Get initial prompt for target agent
        initial_prompt = None
        for agent in self.config['config']['agents']:
            if agent['name'] == target_agent:
                initial_prompt = agent['prompt']
                break
        
        if not initial_prompt:
            raise ValueError(f"No prompt found for target agent: {target_agent}")
        
        # Create utility evaluator
        evaluator = SimulationUtilityEvaluator(
            self.config['config'], 
            target_agent,
            self.config.get('model')
        )
        
        logger.info(f"🧠 Algorithm: {opt_config.algorithm.upper()}")
        logger.info(f"🔍 Max iterations: {opt_config.max_iterations}")
        logger.info(f"🎯 Target utility: {opt_config.utility_threshold}")
        logger.info(f"🧬 Meta-learning: {'enabled' if opt_config.enable_meta_learning else 'disabled'}")
        logger.info(f"📝 Prompt templates: {'enabled' if opt_config.enable_prompt_templates else 'disabled'}")
        
        # Run optimization
        result = await optimizer.optimize(
            initial_prompt=initial_prompt,
            utility_function=evaluator.evaluate_utility,
            context=context
        )
        
        # Update configuration with optimized prompt
        for agent in self.config['config']['agents']:
            if agent['name'] == target_agent:
                agent['prompt'] = result.best_prompt
                break
        
        # Run final validation simulations
        logger.info("🔬 Running validation simulations with optimized prompt...")
        validation_results = await self.run_basic_simulation()
        
        return {
            "optimization_result": result,
            "validation_results": validation_results,
            "experiment_name": experiment_name,
            "target_agent": target_agent,
            "final_config": self.config
        }
    
    async def run_with_logging(self, results: Dict[str, Any]) -> str:
        """Run simulation with rich logging framework."""
        if not LOGGING_AVAILABLE:
            logger.warning("Rich logging framework not available")
            return ""
        
        # Setup logging directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path("simulation_logs") / f"enhanced_sim_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging components
        sim_logger = SimulationLogger(simulation_id=f"enhanced_sim_{timestamp}", log_dir=log_dir)
        metrics_collector = MetricsCollector()
        
        # Log optimization results if available
        if "optimization_result" in results:
            opt_result = results["optimization_result"]
            
            # Log optimization progress
            for step in opt_result.optimization_history:
                agent_logger = sim_logger.get_agent_logger(results["target_agent"])
                agent_logger.log_action(
                    action_type="prompt_optimization",
                    content={
                        "step": step["step"],
                        "utility": step["utility"],
                        "prompt_length": len(step["prompt"])
                    }
                )
                
                sim_logger.log_utility_update(
                    agent_name=results["target_agent"],
                    utility_value=step["utility"],
                    environment={"optimization_step": step["step"]}
                )
        
        # Log validation results
        if "validation_results" in results:
            val_results = results["validation_results"]
            
            for run_data in val_results["results"]:
                run_id = run_data["run_id"]
                result = run_data["result"]
                outputs = run_data["outputs"]
                
                # Log conversation
                for msg in result.get("messages", []):
                    sim_logger.log_message(
                        agent_name=msg.get("name", "Unknown"),
                        message=msg.get("content", ""),
                        metadata={"run_id": run_id}
                    )
                
                # Log utilities if agents have compute_utility method
                for agent_name in [agent["name"] for agent in self.config["config"]["agents"]]:
                    sim_logger.log_utility_update(
                        agent_name=agent_name,
                        utility_value=outputs.get(f"{agent_name.lower()}_satisfaction", 0.5),
                        environment={"run_id": run_id, "outputs": outputs}
                    )
        
        # Save logs
        sim_logger.save_logs()
        
        # Generate visualizations and reports
        report_generator = HTMLReporter(log_dir)
        
        # Generate reports (includes visualizations)
        report_path = report_generator.generate_report()
        
        return str(report_path)
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a comprehensive results summary."""
        print("\n" + "="*80)
        print("🎉 ENHANCED SIMULATION RESULTS")
        print("="*80)
        
        if "optimization_result" in results:
            opt_result = results["optimization_result"]
            print(f"🎯 Target Agent: {results['target_agent']}")
            print(f"🧠 Algorithm: {opt_result.optimization_history[0].get('metadata', {}).get('algorithm', 'UCB').upper()}")
            print(f"🔄 Optimization Iterations: {opt_result.total_iterations}")
            print(f"✅ Converged: {opt_result.converged} ({opt_result.convergence_reason})")
            print(f"📈 Best Utility: {opt_result.best_utility:.4f}")
            print(f"📊 Total Improvement: {opt_result.final_statistics.get('total_improvement', 0):.4f}")
            
            print(f"\n🎯 Optimized Prompt ({len(opt_result.best_prompt)} chars):")
            print("-" * 60)
            print(opt_result.best_prompt)
            print("-" * 60)
            
            if opt_result.learned_patterns:
                print(f"\n🧠 Meta-Learning: Discovered {len(opt_result.learned_patterns)} patterns")
                for i, pattern in enumerate(opt_result.learned_patterns[:3], 1):
                    print(f"  {i}. Success rate: {pattern.success_rate:.3f} | "
                          f"Usage: {pattern.usage_count} | "
                          f"Pattern: {pattern.content[:50]}...")
        
        if "validation_results" in results:
            val_results = results["validation_results"]
            print(f"\n📊 Validation Results:")
            print(f"   Total runs: {val_results['total_runs']}")
            print(f"   Successful runs: {val_results['successful_runs']}")
            print(f"   Success rate: {val_results['successful_runs']/val_results['total_runs']*100:.1f}%")
            
            # Show sample outputs
            if val_results["results"]:
                sample_outputs = val_results["results"][-1]["outputs"]
                print(f"   Sample outputs: {sample_outputs}")
        
        print(f"\n📁 Results saved to:")
        if "optimization_result" in results:
            print(f"   • optimization_results/{results['experiment_name']}_*")
        print(f"   • Updated config: {self.config_path}")


class SimulationUtilityEvaluator:
    """Evaluates utility of prompts by running simulations."""
    
    def __init__(self, base_config: Dict[str, Any], target_agent: str, model: str = None):
        self.base_config = base_config.copy()
        self.target_agent = target_agent
        self.model = model
        self.run_count = 0
        
    async def evaluate_utility(self, prompt: str) -> float:
        """Evaluate utility of a prompt by running a simulation."""
        try:
            # Update target agent's prompt
            config = self._update_agent_prompt(self.base_config.copy(), prompt)
            
            # Run simulation
            sim = SelectorGCSimulation(
                config,
                environment={"runs": [], "outputs": {}},
                max_messages=8,  # Shorter for optimization
                min_messages=1,
                model=self.model,
            )
            
            result = await sim.run()
            self.run_count += 1
            
            if not result:
                logger.debug(f"Run {self.run_count}: No result")
                return 0.0
            
            # Calculate utility
            utility = self._calculate_utility(result, sim)
            logger.debug(f"Run {self.run_count}: Utility = {utility:.4f}")
            
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
        outputs = {var["name"]: var["value"] for var in result["output_variables"]}
        
        # Find target agent
        target_agent = None
        for agent in sim.agents:
            if agent.name == self.target_agent:
                target_agent = agent
                break
        
        if not target_agent:
            return 0.0
        
        # Use agent's utility function if available
        if hasattr(target_agent, 'compute_utility'):
            try:
                return max(0.0, min(1.0, target_agent.compute_utility(outputs)))
            except:
                pass
        
        # Fallback utility calculation
        utility = 0.5  # Base utility
        
        # Deal success bonus
        if outputs.get("deal_reached", False):
            utility += 0.3
            
            # Agent-specific bonuses
            if hasattr(target_agent, 'strategy') and isinstance(target_agent.strategy, dict):
                final_price = outputs.get("final_price", 0)
                
                if "max_price" in target_agent.strategy:  # Buyer
                    max_price = target_agent.strategy["max_price"]
                    if final_price <= max_price:
                        utility += 0.2 * (1 - final_price / max_price)
                        
                elif "target_price" in target_agent.strategy:  # Seller
                    target_price = target_agent.strategy["target_price"]
                    if final_price >= target_price:
                        utility += 0.2 * min(1.0, final_price / target_price - 1)
        
        return max(0.0, min(1.0, utility))


async def main(config_path: Path, skip_optimization: bool, skip_logging: bool):
    """Enhanced simulation runner with optimization and logging."""
    
    print("🚀 Enhanced Multi-Agent Simulation Framework")
    print("=" * 60)
    
    runner = EnhancedSimulationRunner(config_path)
    
    # Determine execution mode
    if runner.optimization_enabled and not skip_optimization:
        print("🧠 Mode: Enhanced Optimization")
        results = await runner.run_enhanced_optimization()
    else:
        print("📊 Mode: Basic Simulation")
        results = await runner.run_basic_simulation()
    
    # Generate logging reports if enabled
    report_path = ""
    if runner.logging_enabled and not skip_logging:
        print("\n📊 Generating rich logging reports...")
        report_path = await runner.run_with_logging(results)
    
    # Save updated configuration
    output_config_path = config_path.with_name(f"{config_path.stem}_optimized.json")
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(runner.config, f, indent=2)
    
    # Print comprehensive summary
    runner.print_results_summary(results)
    
    if report_path:
        print(f"   • Rich logging report: {report_path}")
    
    print(f"\n🎉 Simulation completed successfully!")


@click.command()
@click.option(
    "--config",
    "config_path", 
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="src/configs/bike_negotiation_config.json",
    help="Path to simulation configuration JSON",
)
@click.option("--skip-optimization", is_flag=True, help="Skip optimization even if enabled in config")
@click.option("--skip-logging", is_flag=True, help="Skip rich logging")
def sync_main(config_path: Path, skip_optimization: bool, skip_logging: bool):
    """Synchronous wrapper for async main."""
    return asyncio.run(main(config_path, skip_optimization, skip_logging))


if __name__ == "__main__":
    sync_main()
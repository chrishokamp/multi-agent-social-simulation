"""
Enhanced optimization engine that integrates all optimization components.

Combines bandit algorithms, prompt templates, meta-learning, and tracking
into a comprehensive self-optimization system.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from datetime import datetime

from .bandit_optimizer import UCBOptimizer, ThompsonSamplingOptimizer
from .prompt_templates import PromptTemplate, PromptComponent, ComponentType, PromptMutator
from .meta_learner import MetaLearner, SimulationContext, PromptPattern
from .optimization_tracker import OptimizationTracker, ConvergenceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for the enhanced optimizer."""
    algorithm: str = "ucb"  # "ucb" or "thompson_sampling"
    exploration_factor: float = 1.0
    mutation_rate: float = 0.2
    crossover_rate: float = 0.3
    max_iterations: int = 50
    convergence_patience: int = 5
    min_iterations: int = 10
    utility_threshold: float = 0.95
    enable_meta_learning: bool = True
    enable_prompt_templates: bool = True
    save_results: bool = True
    results_dir: str = "optimization_results"
    

@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_prompt: str
    best_utility: float
    total_iterations: int
    converged: bool
    convergence_reason: str
    optimization_history: List[Dict[str, Any]]
    final_statistics: Dict[str, Any]
    learned_patterns: List[PromptPattern] = field(default_factory=list)


class EnhancedOptimizer:
    """Enhanced optimization engine with integrated components."""
    
    def __init__(self, config: OptimizationConfig, experiment_name: str = None):
        self.config = config
        self.experiment_name = experiment_name or f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.bandit_optimizer = self._create_bandit_optimizer()
        self.meta_learner = MetaLearner() if config.enable_meta_learning else None
        self.tracker = OptimizationTracker(self.experiment_name, config.results_dir)
        
        # State
        self.current_context: Optional[SimulationContext] = None
        self.template_population: List[PromptTemplate] = []
        self.iteration_count = 0
        
        logger.info(f"Initialized enhanced optimizer: {self.experiment_name}")
    
    def _create_bandit_optimizer(self):
        """Create the appropriate bandit optimizer."""
        if self.config.algorithm.lower() == "ucb":
            return UCBOptimizer(exploration_factor=self.config.exploration_factor)
        elif self.config.algorithm.lower() == "thompson_sampling":
            return ThompsonSamplingOptimizer(exploration_factor=self.config.exploration_factor)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    async def optimize(self, 
                      initial_prompt: str,
                      utility_function: callable,
                      context: Optional[SimulationContext] = None) -> OptimizationResult:
        """
        Run the optimization process.
        
        Args:
            initial_prompt: Starting prompt to optimize
            utility_function: Function that takes a prompt and returns utility (0-1)
            context: Simulation context for meta-learning
            
        Returns:
            OptimizationResult with the best prompt and statistics
        """
        logger.info(f"Starting optimization for {self.experiment_name}")
        
        self.current_context = context
        self.iteration_count = 0
        
        # Initialize with prompt templates if enabled
        if self.config.enable_prompt_templates:
            await self._initialize_prompt_templates(initial_prompt)
        else:
            # Use simple prompt variants
            await self._initialize_simple_prompts(initial_prompt)
        
        # Get meta-learning suggestions if available
        if self.meta_learner and context:
            await self._incorporate_meta_suggestions(context)
        
        # Main optimization loop
        converged = False
        convergence_reason = "Max iterations reached"
        
        while (self.iteration_count < self.config.max_iterations and not converged):
            # Select next prompt to evaluate
            arm_id, arm_stats = self.bandit_optimizer.select_arm()
            current_prompt = arm_stats.prompt
            
            logger.debug(f"Iteration {self.iteration_count}: Testing arm {arm_id}")
            
            # Evaluate utility
            try:
                utility = await self._evaluate_utility(utility_function, current_prompt)
                utility = max(0.0, min(1.0, utility))  # Clamp to [0, 1]
            except Exception as e:
                logger.error(f"Error evaluating utility: {e}")
                utility = 0.0
            
            # Update bandit optimizer
            self.bandit_optimizer.update(arm_id, utility, {
                'iteration': self.iteration_count,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update tracker
            self.tracker.add_step(current_prompt, utility, {
                'arm_id': arm_id,
                'algorithm': self.config.algorithm
            })
            
            # Update meta-learner
            if self.meta_learner and context:
                self.meta_learner.learn_from_simulation(context, current_prompt, utility)
            
            # Check for convergence
            if self.iteration_count >= self.config.min_iterations:
                converged, convergence_reason = self.tracker.check_convergence()
                
                # Early stopping if we hit utility threshold
                if utility >= self.config.utility_threshold:
                    converged = True
                    convergence_reason = "Utility threshold reached"
            
            # Evolutionary operations periodically
            if (self.config.enable_prompt_templates and 
                self.iteration_count > 0 and 
                self.iteration_count % 5 == 0):
                await self._perform_evolutionary_operations()
            
            # Adaptive parameter adjustment
            if self.iteration_count > 0 and self.iteration_count % 10 == 0:
                await self._adjust_parameters()
            
            self.iteration_count += 1
            
            logger.debug(f"Iteration {self.iteration_count-1} completed: utility={utility:.4f}")
        
        # Get final results
        best_arm_id, best_arm = self.bandit_optimizer.get_best_arm()
        
        result = OptimizationResult(
            best_prompt=best_arm.prompt,
            best_utility=best_arm.mean_reward,
            total_iterations=self.iteration_count,
            converged=converged,
            convergence_reason=convergence_reason,
            optimization_history=[
                {
                    'step': i,
                    'prompt': step.prompt,
                    'utility': step.utility,
                    'metadata': step.metadata
                }
                for i, step in enumerate(self.tracker.steps)
            ],
            final_statistics=self.tracker.get_statistics()
        )
        
        # Add learned patterns from meta-learner
        if self.meta_learner:
            patterns = list(self.meta_learner.prompt_patterns.values())
            result.learned_patterns = sorted(patterns, key=lambda p: p.success_rate, reverse=True)[:10]
        
        # Save results if requested
        if self.config.save_results:
            await self._save_results(result)
        
        logger.info(f"Optimization completed: {result.total_iterations} iterations, "
                   f"best utility: {result.best_utility:.4f}")
        
        return result
    
    async def _initialize_prompt_templates(self, initial_prompt: str):
        """Initialize prompt templates from the initial prompt."""
        # Extract components from initial prompt
        base_template = self._extract_template_from_prompt(initial_prompt)
        
        # Create variations through mutation
        self.template_population = [base_template]
        
        for i in range(4):  # Create 4 variants
            variant = base_template.mutate(self.config.mutation_rate)
            self.template_population.append(variant)
        
        # Add templates to bandit optimizer
        for i, template in enumerate(self.template_population):
            arm_id = f"template_{i}"
            prompt = template.generate_prompt()
            self.bandit_optimizer.add_arm(arm_id, prompt, {'template_id': i})
    
    async def _initialize_simple_prompts(self, initial_prompt: str):
        """Initialize with simple prompt variations."""
        # Create basic variations of the initial prompt
        variations = [
            initial_prompt,
            f"Enhanced version: {initial_prompt}",
            f"Optimized approach: {initial_prompt}",
            f"Strategic focus: {initial_prompt}",
            f"Goal-oriented: {initial_prompt}"
        ]
        
        for i, prompt in enumerate(variations):
            arm_id = f"variant_{i}"
            self.bandit_optimizer.add_arm(arm_id, prompt, {'variant_id': i})
    
    async def _incorporate_meta_suggestions(self, context: SimulationContext):
        """Incorporate suggestions from meta-learner."""
        suggestions = self.meta_learner.suggest_prompt_patterns(context, top_k=3)
        
        for i, pattern in enumerate(suggestions):
            arm_id = f"meta_suggestion_{i}"
            self.bandit_optimizer.add_arm(arm_id, pattern.content, {
                'meta_pattern': True,
                'pattern_id': pattern.pattern_id,
                'success_rate': pattern.success_rate
            })
    
    async def _evaluate_utility(self, utility_function: callable, prompt: str) -> float:
        """Evaluate utility of a prompt."""
        if asyncio.iscoroutinefunction(utility_function):
            return await utility_function(prompt)
        else:
            return utility_function(prompt)
    
    async def _perform_evolutionary_operations(self):
        """Perform evolutionary operations on prompt templates."""
        if not self.config.enable_prompt_templates or len(self.template_population) < 2:
            return
        
        # Get performance-based weights for selection
        arm_stats = self.bandit_optimizer.get_statistics()
        template_performances = {}
        
        for arm_id, stats in arm_stats.items():
            metadata = self.bandit_optimizer.arms[arm_id].metadata
            if 'template_id' in metadata:
                template_id = metadata['template_id']
                template_performances[template_id] = stats['mean_reward']
        
        # Select best templates for breeding
        if len(template_performances) >= 2:
            sorted_templates = sorted(
                template_performances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Crossover between top 2 templates
            parent1_id, parent2_id = sorted_templates[0][0], sorted_templates[1][0]
            parent1 = self.template_population[parent1_id]
            parent2 = self.template_population[parent2_id]
            
            if np.random.random() < self.config.crossover_rate:
                child1, child2 = PromptMutator.crossover(parent1, parent2)
                
                # Add children to population and bandit
                child_id1 = len(self.template_population)
                child_id2 = child_id1 + 1
                
                self.template_population.extend([child1, child2])
                
                # Add to bandit optimizer
                self.bandit_optimizer.add_arm(
                    f"child_{child_id1}", 
                    child1.generate_prompt(),
                    {'template_id': child_id1, 'generation': 'child'}
                )
                self.bandit_optimizer.add_arm(
                    f"child_{child_id2}",
                    child2.generate_prompt(), 
                    {'template_id': child_id2, 'generation': 'child'}
                )
    
    async def _adjust_parameters(self):
        """Adjust optimization parameters based on progress."""
        suggestions = ConvergenceAnalyzer.suggest_parameter_adjustment(self.tracker)
        
        if 'mutation_rate' in suggestions:
            adjustment = suggestions['mutation_rate']
            self.config.mutation_rate *= adjustment
            self.config.mutation_rate = max(0.05, min(0.5, self.config.mutation_rate))
            logger.info(f"Adjusted mutation rate to {self.config.mutation_rate:.3f}")
        
        if 'exploration' in suggestions:
            if 'increase' in suggestions['exploration'].lower():
                self.bandit_optimizer.exploration_factor *= 1.2
            elif 'decrease' in suggestions['exploration'].lower():
                self.bandit_optimizer.exploration_factor *= 0.8
            
            logger.info(f"Adjusted exploration factor to {self.bandit_optimizer.exploration_factor:.3f}")
    
    def _extract_template_from_prompt(self, prompt: str) -> PromptTemplate:
        """Extract a template structure from a prompt."""
        template = PromptTemplate(metadata={'source': 'initial_prompt'})
        
        # Simple heuristic-based extraction
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        
        for sentence in sentences:
            # Classify sentence type based on keywords
            lower_sentence = sentence.lower()
            
            if any(word in lower_sentence for word in ['goal', 'objective', 'aim', 'achieve', 'maximize', 'minimize']):
                component_type = ComponentType.OBJECTIVE
            elif any(word in lower_sentence for word in ['strategy', 'approach', 'method', 'focus', 'prioritize']):
                component_type = ComponentType.STRATEGY
            elif any(word in lower_sentence for word in ['must', 'should', 'cannot', 'never', 'always', 'ensure']):
                component_type = ComponentType.CONSTRAINT
            elif any(word in lower_sentence for word in ['you are', 'your role', 'persona', 'character']):
                component_type = ComponentType.PERSONALITY
            elif any(word in lower_sentence for word in ['context', 'situation', 'scenario', 'environment']):
                component_type = ComponentType.CONTEXT
            else:
                component_type = ComponentType.STRATEGY  # Default
            
            template.add_component(PromptComponent(
                type=component_type,
                content=sentence.strip() + '.'
            ))
        
        # Ensure we have at least one component
        if not template.components:
            template.add_component(PromptComponent(
                type=ComponentType.OBJECTIVE,
                content=prompt
            ))
        
        return template
    
    async def _save_results(self, result: OptimizationResult):
        """Save optimization results to disk."""
        # Save via tracker
        self.tracker.save_results()
        self.tracker.plot_optimization_progress()
        
        # Save meta-learner knowledge base
        if self.meta_learner:
            self.meta_learner.save_knowledge_base()
        
        # Save bandit optimizer state
        bandit_path = Path(self.config.results_dir) / f"{self.experiment_name}_bandit_state.json"
        self.bandit_optimizer.save_state(str(bandit_path))
        
        # Save comprehensive results
        results_path = Path(self.config.results_dir) / f"{self.experiment_name}_comprehensive_results.json"
        
        comprehensive_results = {
            'experiment_name': self.experiment_name,
            'config': {
                'algorithm': self.config.algorithm,
                'exploration_factor': self.config.exploration_factor,
                'mutation_rate': self.config.mutation_rate,
                'max_iterations': self.config.max_iterations,
                'enable_meta_learning': self.config.enable_meta_learning,
                'enable_prompt_templates': self.config.enable_prompt_templates
            },
            'results': {
                'best_prompt': result.best_prompt,
                'best_utility': result.best_utility,
                'total_iterations': result.total_iterations,
                'converged': result.converged,
                'convergence_reason': result.convergence_reason,
                'final_statistics': result.final_statistics
            },
            'bandit_statistics': self.bandit_optimizer.get_statistics(),
            'meta_learning_stats': (
                self.meta_learner.get_pattern_statistics() 
                if self.meta_learner else None
            ),
            'learned_patterns': [
                {
                    'content': p.content,
                    'success_rate': p.success_rate,
                    'usage_count': p.usage_count,
                    'domains': list(p.domain_tags)
                }
                for p in result.learned_patterns
            ]
        }
        
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"Saved comprehensive results to {results_path}")


# Factory function for easy creation
def create_enhanced_optimizer(
    algorithm: str = "ucb",
    exploration_factor: float = 1.0,
    max_iterations: int = 50,
    enable_meta_learning: bool = True,
    experiment_name: str = None
) -> EnhancedOptimizer:
    """Create an enhanced optimizer with sensible defaults."""
    
    config = OptimizationConfig(
        algorithm=algorithm,
        exploration_factor=exploration_factor,
        max_iterations=max_iterations,
        enable_meta_learning=enable_meta_learning
    )
    
    return EnhancedOptimizer(config, experiment_name)
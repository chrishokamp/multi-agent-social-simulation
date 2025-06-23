# Enhanced Self-Optimization Framework

This directory contains the enhanced self-optimization framework for multi-agent simulations. The framework uses advanced techniques including multi-armed bandit algorithms, structured prompt templates, meta-learning, and comprehensive tracking to optimize agent prompts.

## Quick Start

```bash
# Run enhanced optimization
make run-enhanced-optimization

# Run with Thompson Sampling
make run-enhanced-optimization-thompson

# Run tests
make test-optimization
```

## Components

### 1. Bandit Optimizers (`bandit_optimizer.py`)

Multi-armed bandit algorithms for intelligent exploration and exploitation:

- **UCBOptimizer**: Upper Confidence Bound algorithm
- **ThompsonSamplingOptimizer**: Bayesian approach with Beta distributions
- **ArmStatistics**: Performance tracking for each prompt variant

```python
from optimization.bandit_optimizer import UCBOptimizer

optimizer = UCBOptimizer(exploration_factor=1.0)
optimizer.add_arm("prompt1", "Your first prompt")
optimizer.add_arm("prompt2", "Your second prompt")

# Select next prompt to test
arm_id, arm = optimizer.select_arm()

# Update with reward
optimizer.update(arm_id, utility_score)
```

### 2. Prompt Templates (`prompt_templates.py`)

Structured, composable prompt components with systematic mutations:

- **PromptTemplate**: Container for multiple components
- **PromptComponent**: Individual pieces (objectives, strategies, constraints)
- **PromptMutator**: Mutation and crossover operations

```python
from optimization.prompt_templates import PromptTemplate, PromptComponent, ComponentType

template = PromptTemplate()
template.add_component(PromptComponent(
    ComponentType.OBJECTIVE,
    "Maximize your utility in this negotiation."
))

# Generate prompt
prompt = template.generate_prompt()

# Create variations
mutated = template.mutate(mutation_rate=0.2)
```

### 3. Meta-Learning (`meta_learner.py`)

Cross-simulation knowledge transfer and pattern recognition:

- **MetaLearner**: Main learning engine
- **PromptPattern**: Reusable prompt patterns
- **SimulationContext**: Context for similarity matching
- **TransferLearning**: Domain adaptation utilities

```python
from optimization.meta_learner import MetaLearner, SimulationContext

learner = MetaLearner()
context = SimulationContext(
    simulation_type="negotiation",
    domain="business",
    objectives=["profit"],
    constraints=["budget"],
    agent_types=["buyer", "seller"],
    output_variables=["price", "deal"]
)

# Learn from simulation
learner.learn_from_simulation(context, prompt, performance)

# Get suggestions for similar context
suggestions = learner.suggest_prompt_patterns(context)
```

### 4. Optimization Tracking (`optimization_tracker.py`)

Comprehensive tracking and convergence analysis:

- **OptimizationTracker**: Progress monitoring
- **ConvergenceAnalyzer**: Stopping criteria and analysis
- **OptimizationStep**: Individual optimization records

```python
from optimization.optimization_tracker import OptimizationTracker

tracker = OptimizationTracker("experiment_name", "results_dir")

# Track progress
tracker.add_step(prompt, utility, metrics)

# Check convergence
converged, reason = tracker.check_convergence()

# Generate visualizations
tracker.plot_optimization_progress()
tracker.save_results()
```

### 5. Enhanced Optimizer (`enhanced_optimizer.py`)

Main optimization engine that integrates all components:

```python
from optimization.enhanced_optimizer import create_enhanced_optimizer

optimizer = create_enhanced_optimizer(
    algorithm="ucb",
    max_iterations=30,
    enable_meta_learning=True
)

async def utility_function(prompt):
    # Your evaluation logic
    return utility_score

result = await optimizer.optimize(
    initial_prompt="Starting prompt",
    utility_function=utility_function,
    context=simulation_context
)
```

## Key Features

### Multi-Armed Bandit Algorithms

- **UCB**: Confidence-based exploration with theoretical guarantees
- **Thompson Sampling**: Bayesian approach for uncertain environments
- **Adaptive Parameters**: Automatic adjustment based on performance

### Structured Prompt Templates

- **Component Types**: Objectives, strategies, constraints, personality, context
- **Mutations**: Synonym replacement, emphasis modulation, reordering
- **Crossover**: Combine successful templates

### Meta-Learning

- **Pattern Extraction**: Automatic identification of successful patterns
- **Context Similarity**: Find relevant patterns based on simulation context
- **Knowledge Transfer**: Apply lessons across different domains
- **Persistent Learning**: Build knowledge base over time

### Advanced Analytics

- **Convergence Detection**: Multiple criteria for stopping
- **Progress Visualization**: Comprehensive plots and statistics
- **Performance Analysis**: Trend detection and plateau identification
- **Parameter Tuning**: Automatic adjustment suggestions

## Configuration

### Optimization Config

```python
from optimization.enhanced_optimizer import OptimizationConfig

config = OptimizationConfig(
    algorithm="ucb",              # "ucb" or "thompson_sampling"
    exploration_factor=1.0,       # Exploration vs exploitation balance
    mutation_rate=0.2,           # Template mutation probability
    crossover_rate=0.3,          # Template crossover probability
    max_iterations=50,           # Maximum optimization steps
    convergence_patience=5,       # Steps without improvement before stopping
    utility_threshold=0.95,      # Stop when this utility is reached
    enable_meta_learning=True,   # Use cross-simulation learning
    enable_prompt_templates=True, # Use structured templates
    save_results=True,           # Save comprehensive results
    results_dir="optimization_results"
)
```

### Convergence Criteria

```python
from optimization.optimization_tracker import ConvergenceCriteria

criteria = ConvergenceCriteria(
    min_steps=10,                # Minimum steps before checking convergence
    window_size=5,               # Window for improvement analysis
    utility_threshold=0.95,      # Utility-based stopping
    improvement_threshold=0.01,   # Minimum improvement to continue
    stability_threshold=0.02,     # Maximum variance for stability
    patience=5                   # Steps without improvement
)
```

## Usage Examples

### Basic Optimization

```python
import asyncio
from optimization.enhanced_optimizer import create_enhanced_optimizer

async def simple_optimization():
    optimizer = create_enhanced_optimizer(
        algorithm="ucb",
        max_iterations=20
    )
    
    async def evaluate_prompt(prompt):
        # Your simulation logic here
        return 0.75  # Return utility score
    
    result = await optimizer.optimize(
        initial_prompt="You are a helpful assistant.",
        utility_function=evaluate_prompt
    )
    
    print(f"Best prompt: {result.best_prompt}")
    print(f"Best utility: {result.best_utility}")

asyncio.run(simple_optimization())
```

### With Meta-Learning

```python
from optimization.meta_learner import SimulationContext

context = SimulationContext(
    simulation_type="negotiation",
    domain="retail",
    objectives=["customer_satisfaction", "profit"],
    constraints=["budget", "time"],
    agent_types=["customer", "salesperson"],
    output_variables=["purchase_amount", "satisfaction_score"]
)

result = await optimizer.optimize(
    initial_prompt=initial_prompt,
    utility_function=evaluate_prompt,
    context=context  # Enables meta-learning
)
```

### Custom Utility Function

```python
async def complex_utility_function(prompt):
    # Run simulation with the prompt
    simulation_result = await run_simulation(prompt)
    
    # Calculate multiple factors
    deal_success = simulation_result.get('deal_reached', False)
    price_efficiency = simulation_result.get('price_efficiency', 0.5)
    conversation_quality = simulation_result.get('quality_score', 0.5)
    
    # Combine into single utility
    utility = (
        0.4 * (1.0 if deal_success else 0.0) +
        0.3 * price_efficiency +
        0.3 * conversation_quality
    )
    
    return utility
```

## Testing

Run the comprehensive test suite:

```bash
# All optimization tests
python -m pytest tests/test_optimization.py -v

# Specific component tests
python -m pytest tests/test_optimization.py::TestBanditOptimizer -v
python -m pytest tests/test_optimization.py::TestPromptTemplates -v
python -m pytest tests/test_optimization.py::TestMetaLearner -v

# Integration tests
python -m pytest tests/test_optimization.py::TestOptimizationIntegration -v
```

## Performance Tips

1. **Start Small**: Begin with fewer iterations to test the setup
2. **Monitor Progress**: Use the tracking visualizations to identify issues
3. **Tune Parameters**: Adjust exploration factor based on your domain
4. **Use Meta-Learning**: Enable it for better cross-simulation transfer
5. **Validate Results**: Always test optimized prompts in production scenarios

## Troubleshooting

### Common Issues

**Slow Convergence:**
- Increase exploration factor
- Check if utility function is too noisy
- Verify prompt templates are meaningful

**Memory Issues:**
- Reduce meta-learning history size
- Limit template complexity
- Use smaller batch sizes

**Poor Performance:**
- Ensure utility function reflects true objectives
- Check for bugs in simulation logic
- Verify initial prompts are reasonable

### Debug Information

```python
# Check bandit statistics
stats = optimizer.bandit_optimizer.get_statistics()
print(f"Arms: {len(stats)}")
for arm_id, arm_stats in stats.items():
    print(f"{arm_id}: {arm_stats['mean_reward']:.3f} ± {arm_stats['std_dev']:.3f}")

# Analyze convergence
from optimization.optimization_tracker import ConvergenceAnalyzer
suggestions = ConvergenceAnalyzer.suggest_parameter_adjustment(optimizer.tracker)
print(f"Suggestions: {suggestions}")

# Meta-learning patterns
if optimizer.meta_learner:
    patterns = optimizer.meta_learner.get_pattern_statistics()
    print(f"Learned patterns: {patterns['total_patterns']}")
```

## Contributing

When adding new features:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Consider performance implications
5. Maintain backward compatibility

## Future Enhancements

Planned improvements include:

- Multi-objective optimization
- Neural bandit algorithms
- Distributed learning
- Real-time adaptation
- Advanced transfer learning
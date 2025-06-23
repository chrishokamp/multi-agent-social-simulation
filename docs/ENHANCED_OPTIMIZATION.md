# Enhanced Self-Optimization Framework

This document describes the advanced self-optimization framework for multi-agent simulations, featuring bandit algorithms, structured prompt templates, meta-learning, and comprehensive tracking.

## Overview

The enhanced optimization framework significantly improves upon the basic self-optimization by incorporating:

- **Multi-Armed Bandit Algorithms**: UCB and Thompson Sampling for intelligent exploration/exploitation
- **Structured Prompt Templates**: Composable prompt components with systematic mutations
- **Meta-Learning**: Cross-simulation knowledge transfer and pattern recognition
- **Advanced Analytics**: Convergence detection, optimization tracking, and visualization
- **Automated Parameter Adjustment**: Dynamic optimization based on performance trends

## Architecture

### Core Components

```
EnhancedOptimizer
├── BanditOptimizer (UCB/Thompson Sampling)
├── PromptTemplate System
├── MetaLearner
├── OptimizationTracker
└── ConvergenceAnalyzer
```

### Multi-Armed Bandit Algorithms

#### UCB (Upper Confidence Bound)
- Balances exploration and exploitation using confidence intervals
- Selects arms based on: `mean_reward + exploration_factor * sqrt(2 * ln(total_pulls) / arm_pulls)`
- Best for deterministic environments with clear winners

#### Thompson Sampling
- Uses Bayesian approach with Beta distributions
- Samples from posterior distributions to make decisions
- Better for environments with high uncertainty

### Prompt Template System

#### Component Types
- **Objective**: Goals and targets ("Maximize your utility")
- **Strategy**: Approaches and methods ("Use collaborative negotiation")
- **Constraint**: Rules and limitations ("Stay within budget")
- **Personality**: Agent characteristics ("Be assertive but fair")
- **Context**: Situational information ("You are in a business meeting")
- **Output Format**: Structure requirements ("Provide final price")

#### Mutation Operations
- **Synonym Replacement**: Replace words with alternatives
- **Emphasis Modulation**: Add/remove intensifiers
- **Sentence Reordering**: Change component order
- **Component Variation**: Use predefined alternatives

#### Crossover Operations
- **Uniform Crossover**: Random selection from both parents
- **Weighted Crossover**: Probability-based parent selection
- **Component-wise**: Exchange entire components by type

### Meta-Learning System

#### Pattern Extraction
Automatically identifies reusable patterns from successful prompts:
- Objective patterns (goal-oriented language)
- Strategy patterns (approach keywords)
- Constraint patterns (limitation phrases)
- Structural patterns (if-then logic)

#### Similarity Matching
Uses TF-IDF vectorization and cosine similarity to:
- Find similar simulation contexts
- Suggest relevant prompt patterns
- Transfer knowledge across domains

#### Knowledge Base
Maintains persistent storage of:
- Prompt patterns with performance metrics
- Simulation contexts and outcomes
- Pattern clusters for organization
- Domain mappings for transfer learning

## Usage

### Basic Usage

```python
from optimization.enhanced_optimizer import create_enhanced_optimizer

# Create optimizer
optimizer = create_enhanced_optimizer(
    algorithm="ucb",
    exploration_factor=1.0,
    max_iterations=30,
    enable_meta_learning=True
)

# Define utility function
async def evaluate_prompt(prompt):
    # Run simulation with prompt
    # Return utility score (0-1)
    return utility_score

# Run optimization
result = await optimizer.optimize(
    initial_prompt="Your initial agent prompt",
    utility_function=evaluate_prompt,
    context=simulation_context
)

print(f"Best prompt: {result.best_prompt}")
print(f"Best utility: {result.best_utility}")
```

### Command Line Usage

```bash
# Run enhanced optimization with UCB
make run-enhanced-optimization

# Run with Thompson Sampling
make run-enhanced-optimization-thompson

# Custom parameters
python scripts/enhanced_optimization_simulation.py \
    --config src/configs/enhanced_optimization_example.json \
    --target-agent Buyer \
    --algorithm ucb \
    --max-iterations 25 \
    --exploration-factor 1.2
```

### Configuration

Example configuration with optimization settings:

```json
{
  "model": "gpt-4o",
  "config": {
    "agents": [
      {
        "name": "Buyer",
        "utility_class": "BuyerAgent", 
        "strategy": {"max_price": 800},
        "prompt": "Initial prompt...",
        "self_improve": true
      }
    ],
    "output_variables": [...]
  },
  "optimization_config": {
    "algorithm": "ucb",
    "exploration_factor": 1.2,
    "mutation_rate": 0.15,
    "max_iterations": 30,
    "enable_meta_learning": true,
    "enable_prompt_templates": true
  }
}
```

## Advanced Features

### Convergence Detection

The system automatically detects convergence using multiple criteria:

- **Utility Threshold**: Stop when target utility is reached
- **Improvement Stagnation**: Stop when improvements become minimal
- **Stability**: Stop when performance variance is low
- **Patience**: Stop after N iterations without improvement

### Adaptive Parameters

The optimizer automatically adjusts parameters based on performance:

- **Exploration Factor**: Increased when stuck, decreased when progressing
- **Mutation Rate**: Adjusted based on improvement trends
- **Learning Rate**: Modified based on convergence rate

### Visualization and Analytics

Comprehensive tracking includes:

- **Progress Plots**: Utility over time, moving averages, convergence indicators
- **Performance Statistics**: Mean, variance, improvement trends
- **Pattern Analysis**: Success rates, usage patterns, domain clustering
- **Convergence Analysis**: Rate estimation, plateau detection

### Meta-Learning Capabilities

#### Cross-Simulation Learning
- Learn patterns from multiple simulation types
- Transfer knowledge between domains
- Build reusable prompt libraries

#### Pattern Clustering
- Automatically group similar patterns
- Identify high-performing clusters
- Suggest patterns based on context similarity

#### Domain Adaptation
- Map terminology between domains
- Adapt objectives and constraints
- Transfer strategies across scenarios

## Best Practices

### Algorithm Selection

**Use UCB when:**
- You have deterministic outcomes
- Clear winners exist
- Want interpretable confidence bounds

**Use Thompson Sampling when:**
- High uncertainty in outcomes
- Need better exploration
- Bayesian priors are available

### Parameter Tuning

**Exploration Factor:**
- Start with 1.0
- Increase (1.5-2.0) for high exploration
- Decrease (0.5-0.8) when exploitation is preferred

**Mutation Rate:**
- Start with 0.1-0.2
- Higher rates for diverse search
- Lower rates for fine-tuning

**Iterations:**
- Minimum 10 for meaningful results
- 20-50 typical for most problems
- Monitor convergence for early stopping

### Template Design

**Component Balance:**
- Include all relevant component types
- Avoid overly complex templates
- Provide meaningful variations

**Content Quality:**
- Use clear, specific language
- Include domain expertise
- Test component combinations

## Performance Considerations

### Computational Complexity

- **Bandit Operations**: O(k) where k = number of arms
- **Template Generation**: O(c) where c = number of components  
- **Meta-Learning**: O(n²) for similarity computation
- **Overall**: Linear in iterations, quadratic in pattern history

### Memory Usage

- **Pattern Storage**: Grows with successful patterns
- **History Tracking**: Configurable retention policies
- **Vectorization**: Cached for efficiency

### Optimization Tips

1. **Start Simple**: Begin with fewer components and iterations
2. **Monitor Progress**: Use tracking to identify issues early
3. **Tune Parameters**: Adjust based on domain characteristics
4. **Leverage Meta-Learning**: Reuse patterns across similar problems
5. **Validate Results**: Test optimized prompts thoroughly

## Troubleshooting

### Common Issues

**Low Convergence:**
- Increase exploration factor
- Check utility function stability
- Verify prompt template quality

**Slow Progress:**
- Reduce mutation rate
- Increase exploitation
- Check for local optima

**Memory Issues:**
- Limit pattern history
- Reduce template complexity
- Use smaller vectorization

### Debugging Tools

```python
# Check bandit statistics
stats = optimizer.bandit_optimizer.get_statistics()

# Analyze convergence
suggestions = ConvergenceAnalyzer.suggest_parameter_adjustment(tracker)

# Inspect meta-learning
patterns = meta_learner.get_pattern_statistics()
```

## Testing

Run comprehensive tests:

```bash
# Unit tests
make test-optimization

# Integration tests
python -m pytest tests/test_optimization.py::TestOptimizationIntegration -v

# Performance tests
python scripts/benchmark_optimization.py
```

## Future Enhancements

### Planned Features

1. **Multi-Objective Optimization**: Pareto-optimal solutions
2. **Hierarchical Bandits**: Nested optimization structures
3. **Neural Bandits**: Deep learning for complex patterns
4. **Distributed Learning**: Cross-instance knowledge sharing
5. **Real-time Adaptation**: Online learning during production

### Research Directions

- **Prompt Evolution**: Genetic algorithms for template breeding
- **Causal Discovery**: Understanding optimization mechanisms
- **Transfer Learning**: Improved cross-domain adaptation
- **Robust Optimization**: Handling distribution shifts

## References

- [Multi-Armed Bandit Algorithms](https://tor-lattimore.com/downloads/book/book.pdf)
- [Thompson Sampling](https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf)
- [UCB Algorithm](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
- [Meta-Learning Survey](https://arxiv.org/abs/2004.05439)
- [Prompt Engineering](https://arxiv.org/abs/2107.13586)
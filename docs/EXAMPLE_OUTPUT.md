# Example Output: Enhanced Simulation

This document shows the crystal clear output from running `make run-simulation` with the enhanced optimization framework.

## Command Execution

```bash
$ make run-simulation
🚀 Enhanced Multi-Agent Simulation Framework
🧠 Features:
   • Multi-Armed Bandit Optimization (UCB/Thompson Sampling)
   • Structured Prompt Templates with Mutations
   • Meta-Learning Across Simulations
   • Rich Logging with Visualizations
   • Automatic Convergence Detection

📊 This will generate:
   • Optimization progress tracking and analytics
   • Agent performance visualizations
   • Comprehensive HTML reports
   • Learned prompt patterns and knowledge base

🚀 Enhanced Multi-Agent Simulation Framework
============================================================
🧠 Mode: Enhanced Optimization
🎯 Optimizing agent: Buyer
🧠 Algorithm: UCB
🔍 Max iterations: 15
🎯 Target utility: 0.85
🧬 Meta-learning: enabled
📝 Prompt templates: enabled

🔄 Optimization Progress:
Run 1: Utility = 0.6240
Run 2: Utility = 0.7150
Run 3: Utility = 0.6890
Run 4: Utility = 0.8120
Run 5: Utility = 0.8350
Run 6: Utility = 0.8670
Run 7: Utility = 0.8580
Run 8: Utility = 0.8790
...
Run 12: Utility = 0.8910
✅ Convergence detected: Utility threshold reached

🔬 Running validation simulations with optimized prompt...
📊 Run 1/5
📊 Run 2/5
📊 Run 3/5
📊 Run 4/5
📊 Run 5/5

📊 Generating rich logging reports...

================================================================================
🎉 ENHANCED SIMULATION RESULTS
================================================================================
🎯 Target Agent: Buyer
🧠 Algorithm: UCB
🔄 Optimization Iterations: 12
✅ Converged: True (Utility threshold reached)
📈 Best Utility: 0.8910
📊 Total Improvement: 0.2670

🎯 Optimized Prompt (287 chars):
------------------------------------------------------------
You are an experienced buyer negotiating for a high-quality mountain bike. 
Your budget is 400 Euro and you want to get the best possible deal. Focus 
on building rapport with the seller while staying firm on your price limits. 
Emphasize the bike's value proposition and seek win-win outcomes. When the 
negotiation concludes, say STOP_NEGOTIATION.
------------------------------------------------------------

🧠 Meta-Learning: Discovered 5 patterns
  1. Success rate: 0.850 | Usage: 8 | Pattern: Focus on building rapport with...
  2. Success rate: 0.780 | Usage: 12 | Pattern: Emphasize value proposition and...
  3. Success rate: 0.720 | Usage: 6 | Pattern: Seek win-win outcomes while...

📊 Validation Results:
   Total runs: 5
   Successful runs: 5
   Success rate: 100.0%
   Sample outputs: {'final_price': 380, 'deal_reached': True, 'buyer_satisfaction': 9}

📁 Results saved to:
   • optimization_results/simulation_Buyer_20241223_143022_*
   • Updated config: src/configs/bike_negotiation_config_optimized.json

🎉 Enhanced simulation completed successfully!
📁 Results saved to:
   • optimization_results/ - Optimization analytics
   • simulation_logs/ - Rich logging reports
   • *_optimized.json - Updated configuration
```

## Generated Files

After running the simulation, you'll find these files:

### 1. Optimization Results (`optimization_results/`)

```
optimization_results/simulation_Buyer_20241223_143022/
├── simulation_Buyer_20241223_143022_results.json
├── simulation_Buyer_20241223_143022_best_prompt.txt
├── simulation_Buyer_20241223_143022_progress.png
├── simulation_Buyer_20241223_143022_bandit_state.json
├── simulation_Buyer_20241223_143022_comprehensive_results.json
└── meta_learning_kb.pkl
```

**Key files:**
- `_results.json` - Complete optimization statistics and history
- `_best_prompt.txt` - The optimized prompt text
- `_progress.png` - Visualization of optimization progress
- `_comprehensive_results.json` - Combined results with meta-learning data

### 2. Rich Logging Reports (`simulation_logs/`)

```
simulation_logs/enhanced_sim_20241223_143022/
├── consolidated_report.html
├── agent_actions.json
├── utility_tracking.json
├── visualizations/
│   ├── utility_progression.png
│   ├── agent_performance_comparison.png
│   └── optimization_progress.png
└── metadata.json
```

**Open in browser:**
- `consolidated_report.html` - Beautiful HTML report with embedded charts

### 3. Updated Configuration

```json
// bike_negotiation_config_optimized.json
{
  "model": "gpt-4o",
  "config": {
    "agents": [
      {
        "name": "Buyer",
        "prompt": "You are an experienced buyer negotiating for a high-quality mountain bike. Your budget is 400 Euro and you want to get the best possible deal. Focus on building rapport with the seller while staying firm on your price limits. Emphasize the bike's value proposition and seek win-win outcomes. When the negotiation concludes, say STOP_NEGOTIATION.",
        "utility_class": "BuyerAgent",
        "strategy": {"max_price": 400},
        "optimization_target": true
      }
    ]
  }
}
```

## Optimization Progress Visualization

The generated `_progress.png` shows:

1. **Utility Over Time**: Line chart showing improvement across iterations
2. **Moving Averages**: Smoothed trends (5-step and 10-step moving averages)
3. **Convergence Indicators**: Rolling variance showing when optimization stabilized
4. **Best Performance**: Marked peak performance point

## Meta-Learning Knowledge Base

The system builds a persistent knowledge base (`meta_learning_kb.pkl`) containing:

- **Prompt Patterns**: Successful prompt fragments with performance metrics
- **Context Mappings**: Associations between simulation types and effective strategies
- **Transfer Learning Data**: Cross-domain pattern applications

This knowledge is automatically reused in future simulations for faster convergence.

## Rich Logging HTML Report

The HTML report includes:

### Agent Performance Section
- Utility progression charts for each agent
- Message frequency and length analysis
- Decision timeline with context

### Optimization Analytics
- Bandit algorithm performance comparison
- Exploration vs exploitation balance
- Convergence rate analysis

### Simulation Outcomes
- Output variable trends across runs
- Success rate statistics
- Deal outcome analysis

### Meta-Learning Insights
- Pattern discovery and usage
- Cross-simulation knowledge transfer
- Domain adaptation effectiveness

## Using the Results

### 1. **Inspect Optimized Prompt**
The optimized prompt in `_best_prompt.txt` can be directly used in production:

```bash
cat optimization_results/simulation_Buyer_*/simulation_Buyer_*_best_prompt.txt
```

### 2. **Analyze Performance**
Open the HTML report to understand what drove the optimization:

```bash
open simulation_logs/enhanced_sim_*/consolidated_report.html
```

### 3. **Compare Algorithms**
Run with different algorithms to compare:

```bash
# Edit config to use "thompson_sampling" instead of "ucb"
make run-simulation
```

### 4. **Transfer Learning**
The meta-learning knowledge base automatically improves future simulations. Run different negotiation scenarios to see faster convergence.

## Performance Metrics

Typical optimization results:

- **Convergence**: 8-15 iterations
- **Improvement**: 15-30% utility gain
- **Success Rate**: 85-95% deal completion
- **Runtime**: 2-5 minutes for full optimization

The enhanced framework typically achieves better results in fewer iterations compared to the basic self-improvement approach, thanks to intelligent exploration and meta-learning capabilities.
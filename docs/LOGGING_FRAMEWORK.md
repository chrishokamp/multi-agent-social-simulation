# Rich Logging Framework for Multi-Agent Simulations

This document describes the comprehensive logging framework that provides detailed tracking of agent actions, utility function evolution, and beautiful visualizations for simulation experiments.

## Overview

The logging framework captures:
- **Agent Actions**: Every decision, message, and state change
- **Utility Tracking**: Utility function values over time for each agent
- **Conversation Flow**: Complete dialogue history with metadata
- **Performance Metrics**: Aggregated statistics and KPIs
- **Visualizations**: Beautiful charts and reports

## Quick Start

### Basic Usage

```python
# Use the enhanced simulation with logging
from engine.logged_simulation import LoggedSelectorGCSimulation

# Create simulation with logging
sim = LoggedSelectorGCSimulation(
    config=your_config,
    environment=environment,
    log_dir=Path("logs/my_experiment")
)

# Run simulation - logs are automatically generated
result = await sim.run()
```

### Run via Make Command

```bash
# Run simulation with automatic logging and visualization
make run-simulation

# Run basic simulation without logging
make run-simulation-simple
```

The enhanced `make run-simulation` command will:
1. Run the simulation with comprehensive logging
2. Generate beautiful visualizations
3. Create HTML and PDF reports
4. Save all logs in `simulation_logs/` directory

## Architecture

### Core Components

#### SimulationLogger
- Main coordinator for all logging activities
- Manages agent loggers and metrics collection
- Handles file I/O and data persistence

#### AgentLogger
- Tracks individual agent actions and decisions
- Records utility function evolution
- Captures strategy changes and learning

#### MetricsCollector
- Aggregates simulation-wide metrics
- Provides statistical summaries
- Tracks performance indicators

### Data Models

#### AgentAction
```python
@dataclass
class AgentAction:
    timestamp: datetime
    agent_name: str
    action_type: str  # 'message', 'decision', 'strategy_update'
    content: Any
    metadata: Dict[str, Any]
    utility_before: Optional[float]
    utility_after: Optional[float]
```

#### UtilitySnapshot
```python
@dataclass
class UtilitySnapshot:
    timestamp: datetime
    round_number: int
    agent_name: str
    utility_value: float
    environment_state: Dict[str, Any]
```

## Generated Artifacts

### Log Files Structure
```
logs/simulation_id/
├── simulation.log              # Main simulation log
├── messages.json              # All conversation messages
├── agent_AgentName.json       # Per-agent actions and utilities
├── metrics.json               # Aggregated metrics summary
├── simulation_info.json       # Configuration and metadata
└── visualizations/           # Generated charts
    ├── utility_trends.png
    ├── message_flow.png
    ├── action_distribution.png
    ├── utility_heatmap.png
    └── summary_dashboard.png
```

### Visualizations

#### Utility Trends
Line chart showing utility evolution for each agent over simulation rounds.

#### Message Flow Timeline
Visual representation of conversation flow between agents.

#### Action Distribution
Pie charts showing the distribution of action types per agent.

#### Utility Heatmap
Heatmap visualization of utility values across agents and rounds.

#### Summary Dashboard
Comprehensive overview with multiple metrics and visualizations.

### Reports

#### HTML Report
- Interactive report with embedded visualizations
- Agent performance summaries
- Conversation highlights
- Metrics tables

#### PDF Report
- Print-friendly version of HTML report
- Suitable for documentation and sharing

#### Consolidated Report
- Multi-run analysis and trends
- Utility evolution across self-optimization runs
- Performance comparisons

## Advanced Usage

### Custom Logging

```python
# Get agent logger
agent_logger = sim.logger.get_agent_logger("MyAgent")

# Log custom actions
agent_logger.log_action(
    action_type="strategy_update",
    content="Updated bidding strategy",
    metadata={"new_threshold": 0.8},
    utility_before=0.6,
    utility_after=0.7
)

# Log utility manually
agent_logger.log_utility(
    round_number=5,
    utility_value=0.75,
    environment_state={"final_price": 500}
)
```

### Custom Metrics

```python
# Record custom metrics
sim.logger.metrics.record("negotiation_rounds", 8)
sim.logger.metrics.record("agreement_reached", True)
sim.logger.metrics.record("satisfaction_score", 0.85)

# Get metrics summary
summary = sim.logger.metrics.get_summary()
```

### Visualization Customization

```python
from logging_framework.visualization import SimulationVisualizer

# Create custom visualizations
visualizer = SimulationVisualizer(log_dir)

# Generate specific plots
fig = visualizer.plot_utility_trends(save_path="custom_utility.png")
fig = visualizer.plot_message_flow(save_path="custom_flow.png")

# Create custom dashboard
fig = visualizer.create_summary_dashboard(save_path="dashboard.png")
```

### Report Generation

```python
from logging_framework.reporters import HTMLReporter, PDFReporter

# Generate HTML report
html_reporter = HTMLReporter(log_dir)
html_path = html_reporter.generate_report()

# Generate PDF report (requires wkhtmltopdf)
pdf_reporter = PDFReporter(log_dir)
pdf_path = pdf_reporter.generate_report()
```

## Configuration

### Environment Variables

```bash
# Optional: Specify default log directory
export SIMULATION_LOG_DIR="/path/to/logs"

# For PDF generation (macOS)
brew install wkhtmltopdf
```

### Dependencies

The framework requires additional visualization dependencies:

```bash
pip install matplotlib seaborn pandas jinja2 markdown pdfkit
```

## Integration with Existing Code

### Minimal Integration

Replace your existing simulation import:

```python
# Before
from engine.simulation import SelectorGCSimulation

# After  
from engine.logged_simulation import LoggedSelectorGCSimulation as SelectorGCSimulation
```

### Custom Agent Integration

For custom agents, ensure they implement utility tracking:

```python
class MyCustomAgent(UtilityAgent):
    def compute_utility(self, environment):
        # Your utility calculation
        utility = calculate_my_utility(environment)
        
        # Optionally log the calculation
        if hasattr(self, '_logger'):
            self._logger.log_action(
                'utility_calculation',
                f"Calculated utility: {utility}",
                metadata={'method': 'custom'}
            )
        
        return utility
```

## Best Practices

### Performance Considerations
- Logging adds ~10-15% overhead to simulation time
- Large simulations may generate substantial log files
- Use `no-visualizations` flag for batch experiments

### Analysis Workflow
1. Run simulations with logging enabled
2. Review individual run reports for detailed analysis
3. Use consolidated reports for trend analysis
4. Export key metrics for statistical analysis

### Debugging
- Check `simulation.log` for runtime errors
- Verify agent utility calculations in agent logs
- Use message flow visualization to understand conversation dynamics

## Examples

### Running a Logged Experiment

```bash
# Run with custom output directory
make run-simulation SIMULATION_CONFIG=my_config.json

# Run with PDF generation
python scripts/self_optimize_negotiation_with_logging.py \
    --config my_config.json \
    --generate-pdf \
    --output-dir experiment_results/
```

### Analyzing Results

```python
# Load and analyze results
from logging_framework.core import SimulationLogger

logger = SimulationLogger.load_from_directory("logs/simulation_id")
summary = logger.get_summary()

print(f"Total rounds: {summary['total_rounds']}")
print(f"Agents: {summary['agents']}")

# Get utility trends
for agent_name in summary['agents']:
    agent_logger = logger.get_agent_logger(agent_name)
    trend = agent_logger.get_utility_trend()
    print(f"{agent_name} utility progression: {trend}")
```

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest src/backend/tests/unit/test_logging_framework.py -v

# Integration tests
pytest src/backend/tests/integration/test_logged_simulation.py -v

# Full test suite
pytest src/backend/tests/ -v
```

## Troubleshooting

### Common Issues

**Missing visualizations**: Install required dependencies
```bash
pip install matplotlib seaborn pandas
```

**PDF generation fails**: Install wkhtmltopdf
```bash
# macOS
brew install wkhtmltopdf

# Ubuntu
sudo apt-get install wkhtmltopdf
```

**Large log files**: Use selective logging or implement log rotation

**Performance issues**: Disable visualizations for batch runs
```bash
python scripts/self_optimize_negotiation_with_logging.py --no-visualizations
```

## Future Enhancements

- Real-time dashboard for live simulations
- Log compression and archival
- Distributed logging for parallel simulations
- Machine learning-based pattern detection
- Custom visualization plugins
- Integration with experiment tracking platforms
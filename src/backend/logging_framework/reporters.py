"""
Report generation components for simulation results.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from jinja2 import Template
import pdfkit
import markdown
from .visualization import SimulationVisualizer


class SimulationReporter:
    """Base class for generating simulation reports."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.visualizer = SimulationVisualizer(log_dir)
        self._load_data()
    
    def _load_data(self):
        """Load all simulation data."""
        # Load messages
        with open(self.log_dir / "messages.json", 'r') as f:
            self.messages = json.load(f)
        
        # Load metrics
        with open(self.log_dir / "metrics.json", 'r') as f:
            self.metrics = json.load(f)
        
        # Load agent data
        self.agent_data = {}
        for agent_file in self.log_dir.glob("agent_*.json"):
            agent_name = agent_file.stem.replace("agent_", "")
            with open(agent_file, 'r') as f:
                self.agent_data[agent_name] = json.load(f)
        
        # Load simulation info if exists
        self.simulation_info = {}
        if (self.log_dir / "simulation_info.json").exists():
            with open(self.log_dir / "simulation_info.json", 'r') as f:
                self.simulation_info = json.load(f)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the simulation."""
        summary = {
            'simulation_id': self.simulation_info.get('simulation_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'total_messages': len(self.messages),
            'total_rounds': max([msg['round'] for msg in self.messages]) if self.messages else 0,
            'agents': list(self.agent_data.keys()),
            'metrics_summary': self.metrics,
            'agent_summaries': {}
        }
        
        # Add agent-specific summaries
        for agent_name, data in self.agent_data.items():
            agent_summary = {
                'total_actions': len(data['actions']),
                'action_types': {},
                'utility_progression': [],
                'final_utility': None
            }
            
            # Count action types
            for action in data['actions']:
                action_type = action['action_type']
                agent_summary['action_types'][action_type] = \
                    agent_summary['action_types'].get(action_type, 0) + 1
            
            # Utility progression
            if data['utility_history']:
                agent_summary['utility_progression'] = [
                    {'round': u['round_number'], 'utility': u['utility_value']}
                    for u in data['utility_history']
                ]
                agent_summary['final_utility'] = data['utility_history'][-1]['utility_value']
            
            summary['agent_summaries'][agent_name] = agent_summary
        
        return summary
    
    def generate_markdown_report(self) -> str:
        """Generate a markdown report."""
        summary = self.generate_summary()
        
        report = f"""# Simulation Report

## Overview
- **Simulation ID**: {summary['simulation_id']}
- **Generated**: {summary['timestamp']}
- **Total Messages**: {summary['total_messages']}
- **Total Rounds**: {summary['total_rounds']}
- **Agents**: {', '.join(summary['agents'])}

## Agent Performance

"""
        
        # Add agent details
        for agent_name, agent_summary in summary['agent_summaries'].items():
            final_utility = f"{agent_summary['final_utility']:.4f}" if agent_summary['final_utility'] is not None else 'N/A'
            report += f"""### {agent_name}
- **Total Actions**: {agent_summary['total_actions']}
- **Final Utility**: {final_utility}
- **Action Distribution**: {json.dumps(agent_summary['action_types'], indent=2)}

"""
        
        # Add metrics summary
        report += """## Metrics Summary

"""
        for metric_name, stats in summary['metrics_summary'].items():
            if isinstance(stats, dict) and 'mean' in stats:
                report += f"""### {metric_name}
- Mean: {stats['mean']:.4f}
- Min: {stats['min']:.4f}
- Max: {stats['max']:.4f}
- Count: {stats['count']}

"""
        
        # Add conversation highlights
        report += """## Conversation Highlights

"""
        # Show first and last few messages
        if self.messages:
            report += "### Opening Messages\n"
            for msg in self.messages[:5]:
                report += f"- **{msg['agent']}** (Round {msg['round']}): {msg['message'][:100]}...\n"
            
            if len(self.messages) > 10:
                report += "\n### Closing Messages\n"
                for msg in self.messages[-5:]:
                    report += f"- **{msg['agent']}** (Round {msg['round']}): {msg['message'][:100]}...\n"
        
        return report


class HTMLReporter(SimulationReporter):
    """Generate HTML reports with embedded visualizations."""
    
    def generate_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate a complete HTML report."""
        output_path = output_path or self.log_dir / "report.html"
        
        # Generate visualizations
        viz_dir = self.log_dir / "visualizations"
        self.visualizer.save_all_visualizations(viz_dir)
        
        # Generate summary
        summary = self.generate_summary()
        
        # HTML template
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Simulation Report - {{ simulation_id }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #333;
        }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        .visualization img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .agent-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .message {
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Simulation Report</h1>
    
    <div class="metric-card">
        <h2>Overview</h2>
        <div class="summary-grid">
            <div>
                <strong>Simulation ID:</strong> {{ simulation_id }}
            </div>
            <div>
                <strong>Generated:</strong> {{ timestamp }}
            </div>
            <div>
                <strong>Total Messages:</strong> {{ total_messages }}
            </div>
            <div>
                <strong>Total Rounds:</strong> {{ total_rounds }}
            </div>
        </div>
    </div>
    
    <div class="visualization">
        <h2>Summary Dashboard</h2>
        <img src="visualizations/summary_dashboard.png" alt="Summary Dashboard">
    </div>
    
    <h2>Agent Performance</h2>
    {% for agent_name, agent_data in agent_summaries.items() %}
    <div class="agent-section">
        <h3>{{ agent_name }}</h3>
        <div class="summary-grid">
            <div>
                <strong>Total Actions:</strong> {{ agent_data.total_actions }}
            </div>
            <div>
                <strong>Final Utility:</strong> 
                {% if agent_data.final_utility %}
                    {{ "%.4f"|format(agent_data.final_utility) }}
                {% else %}
                    N/A
                {% endif %}
            </div>
        </div>
        
        <h4>Action Distribution</h4>
        <table>
            <tr>
                <th>Action Type</th>
                <th>Count</th>
            </tr>
            {% for action_type, count in agent_data.action_types.items() %}
            <tr>
                <td>{{ action_type }}</td>
                <td>{{ count }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endfor %}
    
    <div class="visualization">
        <h2>Utility Trends</h2>
        <img src="visualizations/utility_trends.png" alt="Utility Trends">
    </div>
    
    <div class="visualization">
        <h2>Utility Heatmap</h2>
        <img src="visualizations/utility_heatmap.png" alt="Utility Heatmap">
    </div>
    
    <div class="visualization">
        <h2>Message Flow</h2>
        <img src="visualizations/message_flow.png" alt="Message Flow">
    </div>
    
    <div class="visualization">
        <h2>Action Distribution</h2>
        <img src="visualizations/action_distribution.png" alt="Action Distribution">
    </div>
    
    <h2>Metrics Summary</h2>
    <div class="metric-card">
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Min</th>
                <th>Max</th>
                <th>Count</th>
            </tr>
            {% for metric_name, stats in metrics_summary.items() %}
            {% if stats.mean is defined %}
            <tr>
                <td>{{ metric_name }}</td>
                <td>{{ "%.4f"|format(stats.mean) }}</td>
                <td>{{ "%.4f"|format(stats.min) }}</td>
                <td>{{ "%.4f"|format(stats.max) }}</td>
                <td>{{ stats.count }}</td>
            </tr>
            {% endif %}
            {% endfor %}
        </table>
    </div>
    
    <h2>Conversation Sample</h2>
    <div class="metric-card">
        {% if messages %}
            <h3>Opening Messages</h3>
            {% for msg in messages[:5] %}
            <div class="message">
                <strong>{{ msg.agent }}</strong> (Round {{ msg.round }}): 
                {{ msg.message[:200] }}{% if msg.message|length > 200 %}...{% endif %}
            </div>
            {% endfor %}
            
            {% if messages|length > 10 %}
            <h3>Closing Messages</h3>
            {% for msg in messages[-5:] %}
            <div class="message">
                <strong>{{ msg.agent }}</strong> (Round {{ msg.round }}): 
                {{ msg.message[:200] }}{% if msg.message|length > 200 %}...{% endif %}
            </div>
            {% endfor %}
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
        """)
        
        # Prepare data for template
        template_data = {
            'simulation_id': summary['simulation_id'],
            'timestamp': summary['timestamp'],
            'total_messages': summary['total_messages'],
            'total_rounds': summary['total_rounds'],
            'agent_summaries': summary['agent_summaries'],
            'metrics_summary': summary['metrics_summary'],
            'messages': self.messages
        }
        
        # Render and save HTML
        html_content = template.render(**template_data)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path


class PDFReporter(HTMLReporter):
    """Generate PDF reports from HTML reports."""
    
    def generate_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate a PDF report."""
        output_path = output_path or self.log_dir / "report.pdf"
        
        # First generate HTML report
        html_path = self.log_dir / "temp_report.html"
        self.generate_report(html_path)
        
        # Convert HTML to PDF
        try:
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None,
                'enable-local-file-access': None
            }
            
            pdfkit.from_file(str(html_path), str(output_path), options=options)
            
            # Clean up temporary HTML
            html_path.unlink()
            
        except Exception as e:
            print(f"Warning: Could not generate PDF report: {e}")
            print("Make sure wkhtmltopdf is installed (brew install wkhtmltopdf on macOS)")
            
            # Fallback: keep HTML report
            html_path.rename(output_path.with_suffix('.html'))
            return output_path.with_suffix('.html')
        
        return output_path
#!/usr/bin/env python
"""
Enhanced self-optimizing negotiation simulation with rich logging and visualization.
"""
import asyncio
import json
from pathlib import Path
from pprint import pprint
from datetime import datetime

import copy
import click
import dotenv
import random
import math
import json
import os


import sys

# Allow running without installation by adjusting PYTHONPATH
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src" / "backend"))

# Preserve backward compatibility - disable streaming by default for CLI
if 'ENABLE_REALTIME_STREAMING' not in os.environ:
    os.environ['ENABLE_REALTIME_STREAMING'] = 'false'

from engine.simulation import SelectorGCSimulation
from logging_framework.reporters import HTMLReporter, PDFReporter
from logging_framework.visualization import SimulationVisualizer
from logging_framework.enhanced_visualization import EnhancedSimulationVisualizer

dotenv.load_dotenv()


def materialise_config(cfg:dict)->dict:
    """
    Return a *static* copy of `cfg` by
      1) sampling all entries under cfg['variables'] (if present);
      2) filling `{var}` placeholders in agent prompts / strategy dicts;
      3) removing the `variables` section.
    A config lacking the 'variables' key is considered already static.
    """
    static = copy.deepcopy(cfg)

    var_rules = static.pop("variables", None)
    if not var_rules:            # already static
        return static['config']
    if "config" not in static:
        return static

    SAFE = {"randint": random.randint, "choice": random.choice,
            "min": min, "max": max, "abs": abs, "round": round, "math": math}

    values = {}
    for name, rule in var_rules.items():
        if "range" in rule:
            r = rule["range"]
            step = r.get("step",1)
            values[name] = random.randrange(r["min"], r["max"]+1, step)
        elif "choice" in rule:
            values[name] = random.choice(rule["choice"])
        elif "expr" in rule:
            values[name] = eval(rule["expr"], SAFE, values)
        else:
            raise ValueError(f"Unknown rule for variable '{name}'")

    for ag in static["config"]["agents"]:
        # prompt
        ag["prompt"] = ag["prompt"].format(**values)
        # strategy dict
        ag["strategy"] = {k: values.get(v, v) for k,v in ag["strategy"].items()}

    return static["config"]


async def run_once(config: dict, environment: dict, model: str | None = None, log_dir: Path = None):
    """Run a single simulation with logging."""
    sim = SelectorGCSimulation(
        config,
        environment=environment,
        model=model,
        log_dir=log_dir
    )
    result = await sim.run()
    return result, sim


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to simulation configuration JSON",
)
@click.option("--max-messages", default=10, show_default=True, help="Maximum conversation length")
@click.option("--min-messages", default=1, show_default=True, help="Minimum messages for valid result")
@click.option("--output-dir", type=click.Path(path_type=Path), help="Output directory for logs and reports")
@click.option("--generate-pdf", is_flag=True, help="Generate PDF report (requires wkhtmltopdf)")
@click.option("--no-visualizations", is_flag=True, help="Skip generating visualizations")
def main(config_path: Path, max_messages: int, min_messages: int, 
         output_dir: Path | None, generate_pdf: bool, no_visualizations: bool):
    """Run a self-optimising negotiation simulation with rich logging."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    config = raw.get("config", raw)
    num_runs = raw.get("num_runs", 5)
    model = raw.get("model") or config.get("model")

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"simulation_logs/{config_path.stem}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Logging to: {output_dir}")
    print(f"📝 Configuration: {config_path.name}")
    print(f"🔄 Planned runs: {num_runs}")
    print(f"🤖 Model: {model or 'default'}")
    print()

    environment = {"runs": [], "outputs": {}}
    history = []
    all_sim_dirs = []

    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"Run {run_idx}/{num_runs}")
        print(f"{'='*60}")
        config = materialise_config(raw)
        
        # Create run-specific log directory
        run_log_dir = output_dir / f"run_{run_idx:03d}"
        
        result, sim = asyncio.run(
            run_once(config, environment, model, log_dir=run_log_dir)
        )
        
        if not result:
            print("No result returned - creating default outputs\n")
            # Create default outputs for failed negotiations
            outputs = {
                "final_price": 0,
                "deal_reached": False,
                "negotiation_rounds": len(sim.group_chat.messages) if hasattr(sim, 'group_chat') else 0,
                "buyer_satisfaction": 1,
                "seller_satisfaction": 1,
                "last_offer_made": 0,
                "last_offer_received": 0
            }
            result = {
                "messages": sim.group_chat.messages if hasattr(sim, 'group_chat') else [],
                "output_variables": [{"name": k, "value": v} for k, v in outputs.items()]
            }
        else:
            pprint(result)
            outputs = result["output_variables"]
        
        # Calculate utilities for each agent
        agent_utilities = {}
        for agent in sim.agents:
            if hasattr(agent, 'compute_utility'):
                utility = agent.compute_utility({"outputs": outputs})
                agent_utilities[agent.name] = utility
                print(f"{agent.name} utility: {utility}")
                
                # Log the final utility to the simulation logger
                if hasattr(sim, 'logger'):
                    sim.logger.log_utility_update(agent.name, utility, {"outputs": outputs})
        
        # Save logs again after updating utilities
        if hasattr(sim, 'logger'):
            sim.logger.save_logs()
            
        history.append({
            "run_id": run_idx, 
            "outputs": outputs,
            "utilities": agent_utilities,
            "log_dir": str(run_log_dir)
        })

        # persist improved prompts for next run
        for agent in sim.agents:
            for cfg in config["agents"]:
                if cfg["name"] == agent.name:
                    cfg["prompt"] = getattr(agent, "system_prompt", cfg.get("prompt"))
        
        all_sim_dirs.append(run_log_dir)
        
        # Generate visualizations for this run if not disabled
        if not no_visualizations:
            print(f"\n📊 Generating visualizations for run {run_idx}...")
            try:
                # Generate standard visualizations
                visualizer = SimulationVisualizer(run_log_dir)
                viz_dir = visualizer.save_all_visualizations()
                print(f"   ✅ Standard visualizations saved to: {viz_dir}")
                
                # Generate enhanced visualizations
                print(f"   🎨 Generating enhanced visualizations...")
                enhanced_visualizer = EnhancedSimulationVisualizer(run_log_dir)
                enhanced_viz_dir = enhanced_visualizer.save_all_enhanced_visualizations()
                print(f"   ✅ Enhanced visualizations saved to: {enhanced_viz_dir}")
                
            except ImportError as e:
                print(f"   ⚠️  Skipping visualizations - missing dependencies: {e}")
                print(f"   💡 Install with: pip install matplotlib seaborn pandas scipy")
            except Exception as e:
                print(f"   ⚠️  Visualization error: {e}")

    # Save consolidated history
    out_path = output_dir / "consolidated_history.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nConsolidated history written to {out_path}")

    # Generate reports for each run
    print(f"\n📄 Generating detailed reports...")
    for idx, sim_dir in enumerate(all_sim_dirs, 1):
        print(f"   📋 Creating report for run {idx}...")
        
        try:
            html_reporter = HTMLReporter(sim_dir)
            html_path = html_reporter.generate_report()
            print(f"      ✅ HTML report: {html_path.name}")
            
            if generate_pdf:
                try:
                    pdf_reporter = PDFReporter(sim_dir)
                    pdf_path = pdf_reporter.generate_report()
                    print(f"      ✅ PDF report: {pdf_path.name}")
                except Exception as e:
                    print(f"      ⚠️  PDF generation failed: {e}")
                    print(f"      💡 Install wkhtmltopdf: brew install wkhtmltopdf (macOS)")
        except Exception as e:
            print(f"      ❌ Report generation failed: {e}")

    # Generate consolidated report
    print(f"\n📊 Generating consolidated multi-run analysis...")
    try:
        _generate_consolidated_report(output_dir, history, config_path)
        print(f"   ✅ Consolidated report created")
    except Exception as e:
        print(f"   ⚠️  Consolidated report error: {e}")

    print(f"\n{'='*80}")
    print(f"🎉 SIMULATION COMPLETE - Rich Logging Framework Results")
    print(f"{'='*80}")
    print(f"📁 All artifacts saved to: {output_dir}")
    print(f"")
    print(f"🔍 Key files to explore:")
    print(f"   📊 Consolidated Report: {output_dir}/consolidated_report.html")
    print(f"   📈 Cross-run Charts:    {output_dir}/consolidated_visualizations/")
    print(f"   📋 Individual Reports:  {output_dir}/run_*/report.html")
    print(f"   📝 Raw Agent Logs:     {output_dir}/run_*/agent_*.json")
    print(f"   💬 Conversation Data:   {output_dir}/run_*/messages.json")
    print(f"")
    print(f"💡 Quick start:")
    print(f"   • Open {output_dir.name}/consolidated_report.html in your browser")
    print(f"   • View utility evolution and cross-run analysis")
    print(f"   • Explore individual run reports for detailed insights")
    print(f"{'='*80}")


def _generate_consolidated_report(output_dir: Path, history: list, config_path: Path):
    """Generate a consolidated report across all runs."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Create consolidated visualizations directory
    viz_dir = output_dir / "consolidated_visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Prepare data
    runs_data = []
    for run in history:
        run_data = {"run_id": run["run_id"]}
        run_data.update(run["outputs"])
        # Only include numeric utility values
        for agent, util in run["utilities"].items():
            if isinstance(util, (int, float)) and not isinstance(util, bool):
                run_data[f"{agent}_utility"] = util
            else:
                run_data[f"{agent}_utility"] = None  # Use None for non-numeric values
        runs_data.append(run_data)
    
    df = pd.DataFrame(runs_data)
    
    # Plot 1: Utility trends across runs for each agent
    fig, ax = plt.subplots(figsize=(12, 6))
    
    utility_columns = [col for col in df.columns if col.endswith('_utility')]
    plotted_any = False
    for col in utility_columns:
        # Filter out None values for plotting
        valid_data = df[['run_id', col]].dropna()
        if not valid_data.empty:
            agent_name = col.replace('_utility', '')
            ax.plot(valid_data['run_id'], valid_data[col], marker='o', label=agent_name, linewidth=2, markersize=8)
            plotted_any = True
    
    if plotted_any:
        ax.set_xlabel('Run Number', fontsize=12)
        ax.set_ylabel('Utility Value', fontsize=12)
        ax.set_title('Agent Utility Evolution Across Self-Optimization Runs', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No numeric utility data available', 
               transform=ax.transAxes, ha='center', va='center', fontsize=16)
        ax.set_title('Agent Utility Evolution Across Self-Optimization Runs', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(viz_dir / "utility_evolution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 2: Output variables evolution (if numerical)
    output_vars = [col for col in df.columns if not col.endswith('_utility') and col != 'run_id']
    numeric_vars = []
    
    for var in output_vars:
        try:
            pd.to_numeric(df[var])
            numeric_vars.append(var)
        except:
            pass
    
    if numeric_vars:
        fig, axes = plt.subplots(len(numeric_vars), 1, figsize=(10, 4*len(numeric_vars)))
        if len(numeric_vars) == 1:
            axes = [axes]
        
        for idx, var in enumerate(numeric_vars):
            ax = axes[idx]
            ax.plot(df['run_id'], pd.to_numeric(df[var]), marker='s', color='green', linewidth=2, markersize=8)
            ax.set_xlabel('Run Number', fontsize=12)
            ax.set_ylabel(var, fontsize=12)
            ax.set_title(f'{var} Evolution', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Output Variables Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(viz_dir / "output_evolution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Generate HTML summary
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Consolidated Simulation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1, h2 {{
                color: #333;
            }}
            .summary-card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            img {{
                max-width: 100%;
                margin: 20px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            .run-link {{
                color: #007bff;
                text-decoration: none;
            }}
            .run-link:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <h1>Consolidated Simulation Report</h1>
        
        <div class="summary-card">
            <h2>Overview</h2>
            <p><strong>Configuration:</strong> {config_path.name}</p>
            <p><strong>Total Runs:</strong> {len(history)}</p>
            <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-card">
            <h2>Utility Evolution</h2>
            <img src="consolidated_visualizations/utility_evolution.png" alt="Utility Evolution">
        </div>
    """
    
    if numeric_vars:
        html_content += """
        <div class="summary-card">
            <h2>Output Variables Evolution</h2>
            <img src="consolidated_visualizations/output_evolution.png" alt="Output Evolution">
        </div>
        """
    
    html_content += """
        <div class="summary-card">
            <h2>Run Details</h2>
            <table>
                <tr>
                    <th>Run</th>
    """
    
    # Add column headers for outputs and utilities
    if history:
        for var in history[0]['outputs'].keys():
            html_content += f"<th>{var}</th>"
        for agent in history[0]['utilities'].keys():
            html_content += f"<th>{agent} Utility</th>"
    
    html_content += "<th>Report</th></tr>"
    
    # Add rows for each run
    for run in history:
        html_content += f"<tr><td>{run['run_id']}</td>"
        
        for var_value in run['outputs'].values():
            html_content += f"<td>{var_value}</td>"
        
        for util_value in run['utilities'].values():
            if isinstance(util_value, (int, float)) and not isinstance(util_value, bool):
                html_content += f"<td>{util_value:.4f}</td>"
            else:
                html_content += "<td>N/A</td>"
        
        html_content += f'<td><a class="run-link" href="run_{run["run_id"]:03d}/report.html">View Report</a></td>'
        html_content += "</tr>"
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_dir / "consolidated_report.html", 'w') as f:
        f.write(html_content)
    
    print(f"Consolidated report saved to: {output_dir / 'consolidated_report.html'}")


if __name__ == "__main__":
    random.seed(7236)
    main()
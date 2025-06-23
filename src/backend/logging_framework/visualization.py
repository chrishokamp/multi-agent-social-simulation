"""
Visualization components for simulation logs.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np


class SimulationVisualizer:
    """Creates visualizations from simulation logs."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.figures: List[Figure] = []
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_agent_data(self, agent_name: str) -> Dict[str, Any]:
        """Load agent log data."""
        with open(self.log_dir / f"agent_{agent_name}.json", 'r') as f:
            return json.load(f)
    
    def load_messages(self) -> List[Dict[str, Any]]:
        """Load conversation messages."""
        with open(self.log_dir / "messages.json", 'r') as f:
            return json.load(f)
    
    def load_metrics(self) -> Dict[str, Any]:
        """Load metrics summary."""
        with open(self.log_dir / "metrics.json", 'r') as f:
            return json.load(f)
    
    def plot_utility_trends(self, save_path: Optional[Path] = None) -> Figure:
        """Plot utility trends for all agents."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get all agent files
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        for agent_file in agent_files:
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            
            if data['utility_history']:
                rounds = [u['round_number'] for u in data['utility_history']]
                utilities = [u['utility_value'] for u in data['utility_history']]
                ax.plot(rounds, utilities, marker='o', label=agent_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('Round Number', fontsize=12)
        ax.set_ylabel('Utility Value', fontsize=12)
        ax.set_title('Agent Utility Values Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add value annotations on the last point for each line
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if len(x_data) > 0:
                ax.annotate(f'{y_data[-1]:.3f}', 
                           xy=(x_data[-1], y_data[-1]),
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_message_flow(self, save_path: Optional[Path] = None) -> Figure:
        """Create a timeline visualization of message flow."""
        messages = self.load_messages()
        
        if not messages:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create agent name to y-position mapping
        agents = sorted(list(set(msg['agent'] for msg in messages)))
        agent_positions = {agent: i for i, agent in enumerate(agents)}
        
        # Plot messages
        for i, msg in enumerate(messages):
            y_pos = agent_positions[msg['agent']]
            
            # Plot message as a horizontal bar
            ax.barh(y_pos, 0.8, left=i, height=0.8, 
                   label=msg['agent'] if i == 0 else "",
                   alpha=0.7)
            
            # Add message preview
            msg_preview = msg['message'][:50] + "..." if len(msg['message']) > 50 else msg['message']
            ax.text(i + 0.4, y_pos, msg_preview, 
                   ha='center', va='center', fontsize=8, wrap=True)
        
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents)
        ax.set_xlabel('Message Sequence', fontsize=12)
        ax.set_title('Conversation Flow Timeline', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, len(messages) - 0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_action_distribution(self, save_path: Optional[Path] = None) -> Figure:
        """Plot distribution of action types per agent."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        for idx, agent_file in enumerate(agent_files[:4]):  # Max 4 agents
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            
            if data['actions']:
                # Count action types
                action_types = {}
                for action in data['actions']:
                    action_type = action['action_type']
                    action_types[action_type] = action_types.get(action_type, 0) + 1
                
                # Create pie chart
                ax = axes[idx]
                ax.pie(action_types.values(), labels=action_types.keys(), 
                       autopct='%1.1f%%', startangle=90)
                ax.set_title(f'{agent_name} - Action Distribution', fontsize=12)
        
        # Hide unused subplots
        for idx in range(len(agent_files), 4):
            axes[idx].axis('off')
        
        plt.suptitle('Agent Action Type Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_utility_heatmap(self, save_path: Optional[Path] = None) -> Figure:
        """Create a heatmap of utility values across agents and rounds."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        if not agent_files:
            return None
        
        # Collect utility data
        utility_data = {}
        max_rounds = 0
        
        for agent_file in agent_files:
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            
            if data['utility_history']:
                utilities = {u['round_number']: u['utility_value'] 
                           for u in data['utility_history']}
                utility_data[agent_name] = utilities
                max_rounds = max(max_rounds, max(utilities.keys()))
        
        # Create matrix
        agents = sorted(utility_data.keys())
        rounds = list(range(1, max_rounds + 1))
        
        matrix = np.zeros((len(agents), len(rounds)))
        for i, agent in enumerate(agents):
            for j, round_num in enumerate(rounds):
                matrix[i, j] = utility_data[agent].get(round_num, np.nan)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(matrix, 
                    xticklabels=rounds,
                    yticklabels=agents,
                    cmap='coolwarm',
                    center=0.5,
                    annot=True,
                    fmt='.3f',
                    cbar_kws={'label': 'Utility Value'},
                    ax=ax)
        
        ax.set_xlabel('Round Number', fontsize=12)
        ax.set_ylabel('Agent', fontsize=12)
        ax.set_title('Utility Values Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def create_summary_dashboard(self, save_path: Optional[Path] = None) -> Figure:
        """Create a comprehensive dashboard with multiple visualizations."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Utility trends (top, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_utility_trends_on_ax(ax1)
        
        # 2. Final utilities bar chart (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_final_utilities_on_ax(ax2)
        
        # 3. Message count per agent (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_message_counts_on_ax(ax3)
        
        # 4. Action type distribution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_total_actions_on_ax(ax4)
        
        # 5. Metrics summary (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_metrics_summary_on_ax(ax5)
        
        # 6. Conversation length over time (bottom, spanning all columns)
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_conversation_dynamics_on_ax(ax6)
        
        plt.suptitle('Simulation Summary Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def _plot_utility_trends_on_ax(self, ax):
        """Helper to plot utility trends on a given axis."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        for agent_file in agent_files:
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            
            if data['utility_history']:
                rounds = [u['round_number'] for u in data['utility_history']]
                utilities = [u['utility_value'] for u in data['utility_history']]
                ax.plot(rounds, utilities, marker='o', label=agent_name, linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Utility')
        ax.set_title('Utility Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_utilities_on_ax(self, ax):
        """Helper to plot final utilities as bar chart."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        agents = []
        final_utilities = []
        
        for agent_file in agent_files:
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            
            if data['utility_history']:
                agents.append(agent_name)
                final_utilities.append(data['utility_history'][-1]['utility_value'])
        
        ax.bar(agents, final_utilities)
        ax.set_ylabel('Final Utility')
        ax.set_title('Final Utility Values')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_message_counts_on_ax(self, ax):
        """Helper to plot message counts per agent."""
        messages = self.load_messages()
        
        if messages:
            agent_counts = {}
            for msg in messages:
                agent = msg['agent']
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            agents = list(agent_counts.keys())
            counts = list(agent_counts.values())
            
            ax.bar(agents, counts)
            ax.set_ylabel('Message Count')
            ax.set_title('Messages per Agent')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_total_actions_on_ax(self, ax):
        """Helper to plot total actions across all agents."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        action_types = {}
        
        for agent_file in agent_files:
            data = self.load_agent_data(agent_file.stem.replace("agent_", ""))
            
            for action in data['actions']:
                action_type = action['action_type']
                action_types[action_type] = action_types.get(action_type, 0) + 1
        
        if action_types:
            ax.pie(action_types.values(), labels=action_types.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Action Type Distribution')
    
    def _plot_metrics_summary_on_ax(self, ax):
        """Helper to plot metrics summary as text."""
        metrics = self.load_metrics()
        
        ax.axis('off')
        
        text = "Metrics Summary\n" + "="*20 + "\n"
        
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict) and 'mean' in stats:
                text += f"\n{metric_name}:\n"
                text += f"  Mean: {stats['mean']:.3f}\n"
                text += f"  Min: {stats['min']:.3f}\n"
                text += f"  Max: {stats['max']:.3f}\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace')
    
    def _plot_conversation_dynamics_on_ax(self, ax):
        """Helper to plot conversation dynamics."""
        messages = self.load_messages()
        
        if messages:
            rounds = [msg['round'] for msg in messages]
            
            # Create bins for rounds
            round_counts = {}
            for r in rounds:
                round_counts[r] = round_counts.get(r, 0) + 1
            
            sorted_rounds = sorted(round_counts.keys())
            counts = [round_counts[r] for r in sorted_rounds]
            
            ax.bar(sorted_rounds, counts)
            ax.set_xlabel('Round Number')
            ax.set_ylabel('Messages per Round')
            ax.set_title('Conversation Dynamics')
            ax.grid(True, alpha=0.3)
    
    def save_all_visualizations(self, output_dir: Optional[Path] = None):
        """Generate and save all visualizations."""
        output_dir = output_dir or self.log_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)
        
        # Generate all plots
        self.plot_utility_trends(output_dir / "utility_trends.png")
        self.plot_message_flow(output_dir / "message_flow.png")
        self.plot_action_distribution(output_dir / "action_distribution.png")
        self.plot_utility_heatmap(output_dir / "utility_heatmap.png")
        self.create_summary_dashboard(output_dir / "summary_dashboard.png")
        
        # Close all figures to free memory
        for fig in self.figures:
            plt.close(fig)
        
        return output_dir
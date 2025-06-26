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
        try:
            with open(self.log_dir / f"agent_{agent_name}.json", 'r') as f:
                data = json.load(f)
                # Ensure required fields exist with defaults
                if not isinstance(data, dict):
                    return {'utility_history': [], 'actions': []}
                data.setdefault('utility_history', [])
                data.setdefault('actions', [])
                return data
        except Exception as e:
            print(f"Error loading agent data for {agent_name}: {e}")
            return {'utility_history': [], 'actions': []}
    
    def load_messages(self) -> List[Dict[str, Any]]:
        """Load conversation messages."""
        try:
            with open(self.log_dir / "messages.json", 'r') as f:
                data = json.load(f)
                # Ensure we return a list
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # If it's a dict, it might be wrapped
                    if 'messages' in data:
                        messages_data = data['messages']
                        # Ensure messages_data is a list
                        if isinstance(messages_data, list):
                            return messages_data
                        elif isinstance(messages_data, dict):
                            return [messages_data]
                        else:
                            return []
                    else:
                        # Convert single message to list
                        return [data]
                else:
                    return []
        except Exception as e:
            print(f"Error loading messages: {e}")
            return []
    
    def load_metrics(self) -> Dict[str, Any]:
        """Load metrics summary."""
        try:
            with open(self.log_dir / "metrics.json", 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return {}
    
    def plot_utility_trends(self, save_path: Optional[Path] = None) -> Figure:
        """Plot utility trends for all agents."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get all agent files
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        for agent_file in agent_files:
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            
            utility_history = data.get('utility_history', [])
            if utility_history:
                # Extract rounds and utilities, filtering out non-numeric utility values
                valid_entries = []
                for u in utility_history:
                    if isinstance(u, dict):
                        round_num = u.get('round_number', 0)
                        util_val = u.get('utility_value', 0.0)
                        # Check if utility_value is numeric (not a dict or other non-numeric type)
                        if isinstance(util_val, (int, float)) and not isinstance(util_val, bool):
                            valid_entries.append((round_num, util_val))
                
                if valid_entries:  # Only plot if we have valid data
                    rounds, utilities = zip(*valid_entries)
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
        try:
            agents = []
            for msg in messages:
                if isinstance(msg, dict) and 'agent' in msg:
                    agent = msg.get('agent')
                    # Handle case where agent is a dict
                    if isinstance(agent, dict):
                        agent_str = agent.get('name', str(agent))
                    else:
                        agent_str = str(agent)
                    if agent_str not in agents:
                        agents.append(agent_str)
            agents = sorted(agents)
        except (TypeError, KeyError) as e:
            print(f"Error extracting agents from messages: {e}")
            return None
        
        if not agents:
            print("No valid agents found in messages")
            return None
            
        agent_positions = {agent: i for i, agent in enumerate(agents)}
        
        # Plot messages
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'agent' not in msg:
                continue
            agent = msg.get('agent', '')
            # Handle case where agent is a dict
            if isinstance(agent, dict):
                agent = agent.get('name', str(agent))
            else:
                agent = str(agent)
            if agent not in agent_positions:
                continue
            y_pos = agent_positions[agent]
            
            # Plot message as a horizontal bar
            ax.barh(y_pos, 0.8, left=i, height=0.8, 
                   label=agent if i == 0 else "",
                   alpha=0.7)
            
            # Add message preview
            message_text = msg.get('message', '')
            msg_preview = message_text[:50] + "..." if len(message_text) > 50 else message_text
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
            
            actions = data.get('actions', [])
            if actions:
                # Count action types
                action_types = {}
                for action in actions:
                    if isinstance(action, dict):
                        action_type = action.get('action_type', 'unknown')
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
            
            utility_history = data.get('utility_history', [])
            if utility_history:
                utilities = {}
                for u in utility_history:
                    if isinstance(u, dict):
                        round_num = u.get('round_number', 0)
                        util_val = u.get('utility_value', 0.0)
                        # Only include numeric utility values
                        if isinstance(util_val, (int, float)) and not isinstance(util_val, bool):
                            utilities[round_num] = util_val
                if utilities:
                    utility_data[agent_name] = utilities
                    max_rounds = max(max_rounds, max(utilities.keys()))
        
        # Handle case with no utility data
        if not utility_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No utility data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16, color='gray')
            ax.set_title('Utility Values Heatmap', fontsize=14, fontweight='bold')
            ax.axis('off')
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.figures.append(fig)
            return fig
            
        # Create matrix
        agents = sorted(utility_data.keys())
        if max_rounds == 0:
            # If all utilities are at round 0, include round 0
            rounds = [0]
        else:
            rounds = list(range(0, max_rounds + 1))
        
        matrix = np.zeros((len(agents), len(rounds)))
        for i, agent in enumerate(agents):
            for j, round_num in enumerate(rounds):
                matrix[i, j] = utility_data[agent].get(round_num, np.nan)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Determine if we have negative values
        if matrix.size == 0 or np.all(np.isnan(matrix)):
            # Handle empty or all-NaN matrix
            vmin, vmax = -1, 1
        else:
            vmin = np.nanmin(matrix)
            vmax = np.nanmax(matrix)
        
        # Use appropriate colormap and center
        if vmin < 0:
            cmap = 'RdBu_r'  # Red for negative, Blue for positive
            center = 0
        else:
            cmap = 'coolwarm'
            center = (vmin + vmax) / 2
        
        sns.heatmap(matrix, 
                    xticklabels=rounds,
                    yticklabels=agents,
                    cmap=cmap,
                    center=center,
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
            
            utility_history = data.get('utility_history', [])
            if utility_history:
                # Extract rounds and utilities, filtering out non-numeric utility values
                valid_entries = []
                for u in utility_history:
                    if isinstance(u, dict):
                        round_num = u.get('round_number', 0)
                        util_val = u.get('utility_value', 0.0)
                        # Check if utility_value is numeric (not a dict or other non-numeric type)
                        if isinstance(util_val, (int, float)) and not isinstance(util_val, bool):
                            valid_entries.append((round_num, util_val))
                
                if valid_entries:  # Only plot if we have valid data
                    rounds, utilities = zip(*valid_entries)
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
            
            utility_history = data.get('utility_history', [])
            if utility_history:
                # Find the last numeric utility value
                for u in reversed(utility_history):
                    if isinstance(u, dict):
                        util_val = u.get('utility_value', 0.0)
                        if isinstance(util_val, (int, float)) and not isinstance(util_val, bool):
                            agents.append(agent_name)
                            final_utilities.append(util_val)
                            break
        
        if agents and final_utilities:
            ax.bar(agents, final_utilities)
            ax.set_ylabel('Final Utility')
            ax.set_title('Final Utility Values')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No numeric utility data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Final Utility Values')
            ax.axis('off')
    
    def _plot_message_counts_on_ax(self, ax):
        """Helper to plot message counts per agent."""
        messages = self.load_messages()
        
        if messages:
            agent_counts = {}
            for msg in messages:
                if isinstance(msg, dict) and 'agent' in msg:
                    agent = msg['agent']
                    # Handle case where agent is a dict
                    if isinstance(agent, dict):
                        # Try to get a string representation
                        agent_str = agent.get('name', str(agent))
                    else:
                        agent_str = str(agent)
                    agent_counts[agent_str] = agent_counts.get(agent_str, 0) + 1
            
            if agent_counts:
                agents = list(agent_counts.keys())
                counts = list(agent_counts.values())
                
                ax.bar(agents, counts)
                ax.set_ylabel('Message Count')
                ax.set_title('Messages per Agent')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No valid messages found', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Messages per Agent')
    
    def _plot_total_actions_on_ax(self, ax):
        """Helper to plot total actions across all agents."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        action_types = {}
        
        for agent_file in agent_files:
            data = self.load_agent_data(agent_file.stem.replace("agent_", ""))
            
            actions = data.get('actions', [])
            for action in actions:
                if isinstance(action, dict):
                    action_type = action.get('action_type', 'unknown')
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
                text += f"  Mean: {stats.get('mean', 0.0):.3f}\n"
                if 'min' in stats:
                    text += f"  Min: {stats.get('min', 0.0):.3f}\n"
                if 'max' in stats:
                    text += f"  Max: {stats.get('max', 0.0):.3f}\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace')
    
    def _plot_conversation_dynamics_on_ax(self, ax):
        """Helper to plot conversation dynamics."""
        messages = self.load_messages()
        
        if messages:
            rounds = []
            for msg in messages:
                if isinstance(msg, dict) and 'round' in msg:
                    round_val = msg.get('round', 0)
                    # Only add valid numeric rounds
                    if round_val is not None and isinstance(round_val, (int, float)):
                        rounds.append(int(round_val))
            
            if not rounds:
                ax.text(0.5, 0.5, 'No round data available', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Conversation Dynamics')
                return
            
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
"""
Enhanced visualization components for simulation logs with rich, informative, and beautiful plots.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
import colorsys
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches


class EnhancedSimulationVisualizer:
    """Creates enhanced visualizations from simulation logs with beautiful styling."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.figures: List[Figure] = []
        
        # Set enhanced style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Custom color palette - vibrant and professional
        self.colors = [
            '#FF6B6B',  # Coral Red
            '#4ECDC4',  # Turquoise
            '#45B7D1',  # Sky Blue
            '#96CEB4',  # Sage Green
            '#FECA57',  # Bright Yellow
            '#DDA0DD',  # Plum
            '#98D8C8',  # Mint
            '#F7DC6F',  # Soft Yellow
        ]
        
        # Set default figure parameters
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = '#FAFAFA'
        plt.rcParams['axes.edgecolor'] = '#CCCCCC'
        plt.rcParams['grid.color'] = '#EEEEEE'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
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
    
    def plot_enhanced_utility_trends(self, save_path: Optional[Path] = None) -> Figure:
        """Plot enhanced utility trends with confidence bands and annotations."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get all agent files
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        for idx, agent_file in enumerate(agent_files):
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            color = self.colors[idx % len(self.colors)]
            
            if data['utility_history']:
                rounds = [u['round_number'] for u in data['utility_history']]
                utilities = [u['utility_value'] for u in data['utility_history']]
                
                # Main line plot
                line = ax.plot(rounds, utilities, 
                              color=color, 
                              linewidth=3, 
                              marker='o', 
                              markersize=10, 
                              label=agent_name,
                              markeredgecolor='white',
                              markeredgewidth=2,
                              alpha=0.9)[0]
                
                # Add trend line
                if len(rounds) > 1:
                    try:
                        z = np.polyfit(rounds, utilities, 1)
                        p = np.poly1d(z)
                        ax.plot(rounds, p(rounds), 
                               color=color, 
                               linestyle='--', 
                               alpha=0.5, 
                               linewidth=2)
                    except np.linalg.LinAlgError:
                        # Skip trend line if polyfit fails
                        pass
                
                # Add confidence band using rolling window
                if len(utilities) > 3:
                    window = min(3, len(utilities) // 2)
                    df = pd.DataFrame({'rounds': rounds, 'utilities': utilities})
                    rolling_mean = df['utilities'].rolling(window=window, center=True).mean()
                    rolling_std = df['utilities'].rolling(window=window, center=True).std()
                    
                    ax.fill_between(rounds, 
                                   rolling_mean - rolling_std, 
                                   rolling_mean + rolling_std,
                                   color=color, 
                                   alpha=0.1)
                
                # Add zero line for reference when there are negative values
                if any(u < 0 for u in utilities):
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
                
                # Annotate key points
                # Starting value
                ax.annotate(f'{utilities[0]:.3f}',
                           xy=(rounds[0], utilities[0]),
                           xytext=(-20, 10),
                           textcoords='offset points',
                           fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor=color, 
                                   alpha=0.7,
                                   edgecolor='white'),
                           color='white',
                           fontweight='bold')
                
                # Ending value
                ax.annotate(f'{utilities[-1]:.3f}',
                           xy=(rounds[-1], utilities[-1]),
                           xytext=(10, 10),
                           textcoords='offset points',
                           fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor=color, 
                                   alpha=0.7,
                                   edgecolor='white'),
                           color='white',
                           fontweight='bold')
                
                # Max value if different from start/end
                max_idx = np.argmax(utilities)
                if max_idx not in [0, len(utilities)-1]:
                    ax.annotate(f'Peak: {utilities[max_idx]:.3f}',
                               xy=(rounds[max_idx], utilities[max_idx]),
                               xytext=(0, 20),
                               textcoords='offset points',
                               fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='gold', 
                                       alpha=0.8),
                               ha='center',
                               arrowprops=dict(arrowstyle='->', color=color))
        
        # Enhanced styling
        ax.set_xlabel('Round Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Utility Value', fontsize=14, fontweight='bold')
        ax.set_title('Agent Utility Evolution with Trends', fontsize=18, fontweight='bold', pad=20)
        
        # Enhanced legend
        legend = ax.legend(loc='best', 
                          frameon=True, 
                          shadow=True, 
                          fancybox=True,
                          framealpha=0.9,
                          fontsize=12)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add subtle background gradient
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        self.figures.append(fig)
        return fig
    
    def plot_sankey_message_flow(self, save_path: Optional[Path] = None) -> Figure:
        """Create a Sankey-style diagram showing message flow between agents."""
        messages = self.load_messages()
        
        if not messages:
            return None
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create agent positions
        agents = sorted(list(set(msg['agent'] for msg in messages)))
        agent_y_pos = {agent: i for i, agent in enumerate(agents)}
        
        # Color map for agents
        agent_colors = {agent: self.colors[i % len(self.colors)] 
                       for i, agent in enumerate(agents)}
        
        # Plot timeline background
        for i in range(0, len(messages), 2):
            ax.axvspan(i-0.5, i+0.5, alpha=0.05, color='gray', zorder=0)
        
        # Plot messages as curved arrows
        for i, msg in enumerate(messages):
            y_pos = agent_y_pos[msg['agent']]
            
            # Create a fancy box for the message
            fancy_box = FancyBboxPatch((i-0.4, y_pos-0.35), 0.8, 0.7,
                                      boxstyle="round,pad=0.1",
                                      facecolor=agent_colors[msg['agent']],
                                      edgecolor='white',
                                      linewidth=2,
                                      alpha=0.8)
            ax.add_patch(fancy_box)
            
            # Add message preview with word wrap
            msg_preview = msg['message'][:40] + "..." if len(msg['message']) > 40 else msg['message']
            ax.text(i, y_pos, msg_preview, 
                   ha='center', va='center', 
                   fontsize=9, 
                   color='white',
                   fontweight='bold',
                   wrap=True)
            
            # Draw connections between messages
            if i > 0:
                prev_agent = messages[i-1]['agent']
                prev_y = agent_y_pos[prev_agent]
                
                # Draw curved connection
                if prev_agent != msg['agent']:
                    ax.annotate('', xy=(i-0.4, y_pos), 
                               xytext=(i-1+0.4, prev_y),
                               arrowprops=dict(arrowstyle='->', 
                                             connectionstyle="arc3,rad=0.3",
                                             color='gray',
                                             alpha=0.5,
                                             linewidth=2))
        
        # Styling
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents, fontsize=12, fontweight='bold')
        ax.set_xlabel('Message Sequence', fontsize=14, fontweight='bold')
        ax.set_title('Conversation Flow Diagram', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlim(-1, len(messages))
        ax.set_ylim(-0.5, len(agents)-0.5)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add agent color legend
        patches = [mpatches.Patch(color=color, label=agent) 
                  for agent, color in agent_colors.items()]
        ax.legend(handles=patches, loc='upper right', title='Agents')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        self.figures.append(fig)
        return fig
    
    def plot_utility_journey_map(self, save_path: Optional[Path] = None) -> Figure:
        """Create a journey map showing utility progression with annotations."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        agent_files = list(self.log_dir.glob("agent_*.json"))
        messages = self.load_messages()
        
        # Create time axis based on messages
        message_times = list(range(len(messages)))
        
        for idx, agent_file in enumerate(agent_files):
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            color = self.colors[idx % len(self.colors)]
            
            if data['utility_history'] and data['actions']:
                # Map utilities to message timeline
                utilities = []
                utility_dict = {u['round_number']: u['utility_value'] 
                               for u in data['utility_history']}
                
                current_utility = 0
                for i, msg in enumerate(messages):
                    round_num = msg.get('round', 1)
                    if round_num in utility_dict:
                        current_utility = utility_dict[round_num]
                    utilities.append(current_utility)
                
                # Plot utility line
                ax.plot(message_times, utilities, 
                       color=color, 
                       linewidth=3, 
                       label=agent_name,
                       alpha=0.8)
                
                # Add action markers
                for action in data['actions']:
                    round_num = action.get('round', 1)
                    # Find corresponding message index
                    msg_idx = next((i for i, msg in enumerate(messages) 
                                   if msg.get('round', 1) == round_num 
                                   and msg['agent'] == agent_name), None)
                    
                    if msg_idx is not None and msg_idx < len(utilities):
                        # Different markers for different action types
                        marker_style = {
                            'negotiate': 'o',
                            'offer': 's',
                            'accept': '^',
                            'reject': 'v',
                            'counter': 'D'
                        }.get(action['action_type'], 'o')
                        
                        ax.scatter(msg_idx, utilities[msg_idx], 
                                 color=color, 
                                 s=200, 
                                 marker=marker_style,
                                 edgecolor='white',
                                 linewidth=2,
                                 zorder=5)
                        
                        # Add action label
                        ax.annotate(action['action_type'],
                                   xy=(msg_idx, utilities[msg_idx]),
                                   xytext=(0, 15),
                                   textcoords='offset points',
                                   fontsize=9,
                                   ha='center',
                                   bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor=color,
                                           alpha=0.7,
                                           edgecolor='white'),
                                   color='white')
        
        # Add phase annotations
        if len(messages) > 0:
            phases = [
                (0, len(messages)//3, "Opening Phase", '#FFE5E5'),
                (len(messages)//3, 2*len(messages)//3, "Negotiation Phase", '#E5F5FF'),
                (2*len(messages)//3, len(messages), "Closing Phase", '#E5FFE5')
            ]
            
            for start, end, label, color in phases:
                if start < len(messages):
                    ax.axvspan(start, min(end, len(messages)-1), 
                             alpha=0.1, 
                             color=color, 
                             label=label)
                    ax.text((start + min(end, len(messages)-1))/2, 
                           ax.get_ylim()[1]*0.95,
                           label, 
                           ha='center', 
                           fontsize=12, 
                           fontweight='bold',
                           color='gray')
        
        # Styling
        ax.set_xlabel('Conversation Timeline', fontsize=14, fontweight='bold')
        ax.set_ylabel('Utility Value', fontsize=14, fontweight='bold')
        ax.set_title('Agent Utility Journey Map', fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.2, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        self.figures.append(fig)
        return fig
    
    def plot_negotiation_dynamics_heatmap(self, save_path: Optional[Path] = None) -> Figure:
        """Create an advanced heatmap showing negotiation dynamics."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        messages = self.load_messages()
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        if not messages or not agent_files:
            return None
        
        # Create interaction matrix
        agents = sorted(list(set(msg['agent'] for msg in messages)))
        n_agents = len(agents)
        n_rounds = max(msg.get('round', 1) for msg in messages)
        
        # Utility matrix
        utility_matrix = np.zeros((n_agents, n_rounds))
        
        for i, agent in enumerate(agents):
            try:
                data = self.load_agent_data(agent)
                for util in data['utility_history']:
                    round_idx = util['round_number'] - 1
                    if round_idx < n_rounds:
                        utility_matrix[i, round_idx] = util['utility_value']
            except:
                pass
        
        # Create custom colormap
        colors_r = np.linspace(0.8, 1, 128)
        colors_g = np.linspace(0.2, 1, 128)
        colors_b = np.linspace(0.2, 0.2, 128)
        colors_rgb = np.column_stack([colors_r, colors_g, colors_b])
        
        # Determine color scale based on utility values
        if utility_matrix.size == 0 or np.all(np.isnan(utility_matrix)):
            # Handle empty or all-NaN matrix
            vmin, vmax = -1, 1
        else:
            vmin = np.nanmin(utility_matrix)
            vmax = np.nanmax(utility_matrix)
        
        # Use diverging colormap centered at 0 if there are negative values
        if vmin < 0:
            cmap = 'RdBu_r'  # Red for negative, Blue for positive
            center = 0
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        else:
            cmap = 'RdYlGn'
            center = None
        
        # Main heatmap
        im = ax1.imshow(utility_matrix, 
                       aspect='auto', 
                       cmap=cmap,
                       interpolation='bilinear',
                       vmin=vmin,
                       vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Utility Value', fontsize=12, fontweight='bold')
        
        # Annotations
        for i in range(n_agents):
            for j in range(n_rounds):
                if utility_matrix[i, j] != 0:
                    text_color = 'white' if utility_matrix[i, j] < 0.5 else 'black'
                    ax1.text(j, i, f'{utility_matrix[i, j]:.2f}',
                            ha='center', va='center',
                            color=text_color,
                            fontsize=10,
                            fontweight='bold')
        
        ax1.set_xticks(range(n_rounds))
        ax1.set_xticklabels([f'R{i+1}' for i in range(n_rounds)])
        ax1.set_yticks(range(n_agents))
        ax1.set_yticklabels(agents, fontsize=12)
        ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Agent', fontsize=12, fontweight='bold')
        ax1.set_title('Utility Dynamics Heatmap', fontsize=16, fontweight='bold')
        
        # Message frequency plot
        round_messages = [0] * n_rounds
        for msg in messages:
            round_idx = msg.get('round', 1) - 1
            if round_idx < n_rounds:
                round_messages[round_idx] += 1
        
        ax2.bar(range(n_rounds), round_messages, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Messages', fontsize=12, fontweight='bold')
        ax2.set_title('Message Frequency per Round', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(n_rounds))
        ax2.set_xticklabels([f'R{i+1}' for i in range(n_rounds)])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        self.figures.append(fig)
        return fig
    
    def plot_agent_strategy_radar(self, save_path: Optional[Path] = None) -> Figure:
        """Create radar charts showing agent performance across multiple dimensions."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        if not agent_files:
            return None
        
        # Calculate metrics for each agent
        agent_metrics = {}
        
        for agent_file in agent_files:
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            
            metrics = {
                'Final Utility': 0,
                'Avg Utility': 0,
                'Utility Growth': 0,
                'Action Diversity': 0,
                'Message Count': 0,
                'Negotiation Efficiency': 0
            }
            
            if data['utility_history']:
                utilities = [u['utility_value'] for u in data['utility_history']]
                metrics['Final Utility'] = utilities[-1]
                metrics['Avg Utility'] = np.mean(utilities)
                metrics['Utility Growth'] = utilities[-1] - utilities[0] if len(utilities) > 1 else 0
            
            if data['actions']:
                action_types = set(a['action_type'] for a in data['actions'])
                metrics['Action Diversity'] = len(action_types) / max(len(data['actions']), 1)
            
            # Count messages
            messages = self.load_messages()
            metrics['Message Count'] = sum(1 for m in messages if m['agent'] == agent_name)
            
            # Negotiation efficiency (utility per message)
            if metrics['Message Count'] > 0:
                metrics['Negotiation Efficiency'] = metrics['Final Utility'] / metrics['Message Count']
            
            agent_metrics[agent_name] = metrics
        
        # Normalize metrics to 0-1 scale
        all_values = {}
        for metric in list(agent_metrics.values())[0].keys():
            values = [agent_metrics[agent][metric] for agent in agent_metrics]
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                for agent in agent_metrics:
                    agent_metrics[agent][metric] = (agent_metrics[agent][metric] - min_val) / (max_val - min_val)
            else:
                for agent in agent_metrics:
                    agent_metrics[agent][metric] = 0.5
        
        # Create radar chart
        n_agents = len(agent_metrics)
        n_cols = min(3, n_agents)
        n_rows = (n_agents + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 6*n_rows))
        
        for idx, (agent_name, metrics) in enumerate(agent_metrics.items()):
            ax = plt.subplot(n_rows, n_cols, idx + 1, projection='polar')
            
            # Prepare data
            categories = list(metrics.keys())
            values = list(metrics.values())
            values += values[:1]  # Complete the circle
            
            # Angles for each axis
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot
            color = self.colors[idx % len(self.colors)]
            ax.plot(angles, values, color=color, linewidth=2, label=agent_name)
            ax.fill(angles, values, color=color, alpha=0.25)
            
            # Fix axis to go in the right order
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw axis lines for each angle and label
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10)
            
            # Set y-axis limits and labels
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75])
            ax.set_yticklabels(['0.25', '0.5', '0.75'], size=8)
            ax.grid(True)
            
            # Add title
            ax.set_title(agent_name, size=14, fontweight='bold', pad=20)
        
        plt.suptitle('Agent Performance Radar Charts', size=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        self.figures.append(fig)
        return fig
    
    def create_enhanced_dashboard(self, save_path: Optional[Path] = None) -> Figure:
        """Create a comprehensive, beautiful dashboard with all visualizations."""
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # Title
        fig.suptitle('Enhanced Simulation Analysis Dashboard', 
                    fontsize=24, 
                    fontweight='bold',
                    y=0.98)
        
        # 1. Utility trends (top, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_enhanced_utility_trends_on_ax(ax1)
        
        # 2. Strategy radar (top right)
        ax2 = fig.add_subplot(gs[0, 2], projection='polar')
        self._plot_mini_radar_on_ax(ax2)
        
        # 3. Message flow (second row, spanning all columns)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_message_timeline_on_ax(ax3)
        
        # 4. Utility heatmap (third row, left 2 columns)
        ax4 = fig.add_subplot(gs[2, :2])
        self._plot_heatmap_on_ax(ax4)
        
        # 5. Action distribution (third row, right)
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_action_sunburst_on_ax(ax5)
        
        # 6. Metrics summary (fourth row, left)
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_metrics_cards_on_ax(ax6)
        
        # 7. Utility distribution (fourth row, center)
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_utility_distribution_on_ax(ax7)
        
        # 8. Performance indicators (fourth row, right)
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_performance_gauges_on_ax(ax8)
        
        # 9. Conversation statistics (bottom row, spanning all)
        ax9 = fig.add_subplot(gs[4, :])
        self._plot_conversation_stats_on_ax(ax9)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        self.figures.append(fig)
        return fig
    
    def _plot_enhanced_utility_trends_on_ax(self, ax):
        """Enhanced utility trends for dashboard."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        for idx, agent_file in enumerate(agent_files):
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            color = self.colors[idx % len(self.colors)]
            
            if data['utility_history']:
                rounds = [u['round_number'] for u in data['utility_history']]
                utilities = [u['utility_value'] for u in data['utility_history']]
                
                ax.plot(rounds, utilities, 
                       color=color,
                       linewidth=3,
                       marker='o',
                       markersize=8,
                       label=agent_name,
                       markeredgecolor='white',
                       markeredgewidth=2)
                
                # Add sparkline effect
                ax.fill_between(rounds, 0, utilities, 
                               color=color, 
                               alpha=0.1)
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Utility', fontsize=12, fontweight='bold')
        ax.set_title('Utility Evolution', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FAFAFA')
    
    def _plot_mini_radar_on_ax(self, ax):
        """Mini radar chart for dashboard."""
        # Aggregate metrics across all agents
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        categories = ['Efficiency', 'Growth', 'Activity', 'Success', 'Consistency']
        values = [0.7, 0.6, 0.8, 0.75, 0.65]  # Example values
        values += values[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors[0])
        ax.fill(angles, values, alpha=0.25, color=self.colors[0])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
    
    def _plot_message_timeline_on_ax(self, ax):
        """Enhanced message timeline for dashboard."""
        messages = self.load_messages()
        
        if messages:
            agents = sorted(list(set(msg['agent'] for msg in messages)))
            agent_colors = {agent: self.colors[i % len(self.colors)] 
                           for i, agent in enumerate(agents)}
            
            for i, msg in enumerate(messages):
                agent = msg['agent']
                y_pos = agents.index(agent)
                
                # Message bubble
                circle = plt.Circle((i, y_pos), 0.4, 
                                  color=agent_colors[agent],
                                  alpha=0.8,
                                  edgecolor='white',
                                  linewidth=2)
                ax.add_patch(circle)
                
                # Connect messages
                if i > 0:
                    ax.plot([i-1, i], [agents.index(messages[i-1]['agent']), y_pos],
                           'gray', alpha=0.3, linewidth=2)
            
            ax.set_ylim(-0.5, len(agents)-0.5)
            ax.set_xlim(-0.5, len(messages)-0.5)
            ax.set_yticks(range(len(agents)))
            ax.set_yticklabels(agents)
            ax.set_xlabel('Message Sequence', fontsize=12, fontweight='bold')
            ax.set_title('Conversation Flow', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.2, axis='x')
    
    def _plot_heatmap_on_ax(self, ax):
        """Simplified heatmap for dashboard."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        if agent_files:
            # Create utility matrix
            data_matrix = []
            agent_names = []
            
            for agent_file in agent_files:
                agent_name = agent_file.stem.replace("agent_", "")
                agent_names.append(agent_name)
                data = self.load_agent_data(agent_name)
                
                if data['utility_history']:
                    utilities = [u['utility_value'] for u in data['utility_history']]
                    data_matrix.append(utilities)
            
            if data_matrix:
                # Pad arrays to same length
                max_len = max(len(row) for row in data_matrix)
                padded_matrix = [row + [np.nan]*(max_len-len(row)) for row in data_matrix]
                
                sns.heatmap(padded_matrix,
                           xticklabels=[f'R{i+1}' for i in range(max_len)],
                           yticklabels=agent_names,
                           cmap='YlOrRd',
                           cbar_kws={'label': 'Utility'},
                           ax=ax,
                           linewidths=0.5)
                
                ax.set_title('Utility Heatmap', fontsize=14, fontweight='bold')
    
    def _plot_action_sunburst_on_ax(self, ax):
        """Action distribution as sunburst/donut chart."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        all_actions = {}
        for agent_file in agent_files:
            data = self.load_agent_data(agent_file.stem.replace("agent_", ""))
            for action in data['actions']:
                action_type = action['action_type']
                all_actions[action_type] = all_actions.get(action_type, 0) + 1
        
        if all_actions:
            # Create donut chart
            wedges, texts, autotexts = ax.pie(all_actions.values(), 
                                              labels=all_actions.keys(),
                                              autopct='%1.1f%%',
                                              colors=self.colors[:len(all_actions)],
                                              wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            
            # Create donut hole
            centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=2, edgecolor='gray')
            ax.add_artist(centre_circle)
            
            ax.set_title('Action Distribution', fontsize=14, fontweight='bold')
    
    def _plot_metrics_cards_on_ax(self, ax):
        """Display key metrics as cards."""
        ax.axis('off')
        
        metrics = self.load_metrics()
        
        # Create metric cards
        card_y = 0.9
        for metric_name, stats in list(metrics.items())[:3]:
            if isinstance(stats, dict) and 'mean' in stats:
                # Card background
                rect = FancyBboxPatch((0.05, card_y-0.25), 0.9, 0.2,
                                     boxstyle="round,pad=0.02",
                                     facecolor='lightblue',
                                     edgecolor='darkblue',
                                     alpha=0.3)
                ax.add_patch(rect)
                
                # Metric text
                ax.text(0.5, card_y-0.05, metric_name,
                       ha='center', fontsize=12, fontweight='bold')
                ax.text(0.5, card_y-0.15, f"{stats['mean']:.3f}",
                       ha='center', fontsize=16, color='darkblue')
                
                card_y -= 0.3
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Key Metrics', fontsize=14, fontweight='bold')
    
    def _plot_utility_distribution_on_ax(self, ax):
        """Plot utility value distributions."""
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        all_utilities = []
        labels = []
        
        for agent_file in agent_files:
            agent_name = agent_file.stem.replace("agent_", "")
            data = self.load_agent_data(agent_name)
            
            if data['utility_history']:
                utilities = [u['utility_value'] for u in data['utility_history']]
                all_utilities.append(utilities)
                labels.append(agent_name)
        
        if all_utilities:
            # Create violin plot
            parts = ax.violinplot(all_utilities, positions=range(len(all_utilities)),
                                 showmeans=True, showmedians=True)
            
            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(self.colors[i % len(self.colors)])
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Utility Value', fontsize=12)
            ax.set_title('Utility Distributions', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_performance_gauges_on_ax(self, ax):
        """Create performance gauge indicators."""
        ax.axis('off')
        
        # Calculate overall performance score
        messages = self.load_messages()
        agent_files = list(self.log_dir.glob("agent_*.json"))
        
        total_messages = len(messages)
        total_agents = len(agent_files)
        
        # Simple performance metric (example)
        performance_score = min(total_messages / (total_agents * 10), 1.0)
        
        # Draw gauge
        theta = np.linspace(np.pi, 0, 100)
        r = 0.9
        
        # Background arc
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, color='lightgray', linewidth=20)
        
        # Performance arc
        performance_theta = theta[:int(performance_score * 100)]
        x_perf = r * np.cos(performance_theta)
        y_perf = r * np.sin(performance_theta)
        
        color = self.colors[0] if performance_score > 0.7 else self.colors[1]
        ax.plot(x_perf, y_perf, color=color, linewidth=20)
        
        # Center text
        ax.text(0, -0.2, f'{performance_score:.0%}',
               ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0, -0.4, 'Performance',
               ha='center', va='center', fontsize=14)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.set_title('Overall Performance', fontsize=14, fontweight='bold')
    
    def _plot_conversation_stats_on_ax(self, ax):
        """Plot detailed conversation statistics."""
        messages = self.load_messages()
        
        if messages:
            # Calculate statistics
            rounds = [msg.get('round', 1) for msg in messages]
            round_counts = {}
            for r in rounds:
                round_counts[r] = round_counts.get(r, 0) + 1
            
            sorted_rounds = sorted(round_counts.keys())
            counts = [round_counts[r] for r in sorted_rounds]
            
            # Create bar chart with gradient
            bars = ax.bar(sorted_rounds, counts, width=0.8)
            
            # Apply gradient coloring
            for i, bar in enumerate(bars):
                color_intensity = i / len(bars)
                bar.set_facecolor(plt.cm.viridis(color_intensity))
                bar.set_edgecolor('white')
                bar.set_linewidth(1.5)
            
            # Add trend line
            if len(sorted_rounds) > 1:
                try:
                    z = np.polyfit(sorted_rounds, counts, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(min(sorted_rounds), max(sorted_rounds), 100)
                    ax.plot(x_smooth, p(x_smooth), 'r--', linewidth=2, alpha=0.7, label='Trend')
                except np.linalg.LinAlgError:
                    # Skip trend line if polyfit fails
                    pass
            
            ax.set_xlabel('Round Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Messages per Round', fontsize=12, fontweight='bold')
            ax.set_title('Conversation Intensity Analysis', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
    
    def save_all_enhanced_visualizations(self, output_dir: Optional[Path] = None):
        """Generate and save all enhanced visualizations."""
        output_dir = output_dir or self.log_dir / "enhanced_visualizations"
        output_dir.mkdir(exist_ok=True)
        
        # Generate all enhanced plots
        self.plot_enhanced_utility_trends(output_dir / "enhanced_utility_trends.png")
        self.plot_sankey_message_flow(output_dir / "sankey_message_flow.png")
        self.plot_utility_journey_map(output_dir / "utility_journey_map.png")
        self.plot_negotiation_dynamics_heatmap(output_dir / "negotiation_dynamics.png")
        self.plot_agent_strategy_radar(output_dir / "strategy_radar.png")
        self.create_enhanced_dashboard(output_dir / "enhanced_dashboard.png")
        
        # Close all figures to free memory
        for fig in self.figures:
            plt.close(fig)
        
        return output_dir
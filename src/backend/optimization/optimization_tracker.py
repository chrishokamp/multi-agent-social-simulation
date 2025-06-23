"""
Optimization tracking and convergence analysis.

Provides comprehensive tracking of optimization progress and
convergence detection for stopping criteria.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import json
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStep:
    """Single step in the optimization process."""
    step_number: int
    timestamp: datetime
    prompt: str
    utility: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceCriteria:
    """Criteria for determining convergence."""
    min_steps: int = 10
    window_size: int = 5
    utility_threshold: float = 0.95  # Stop if utility exceeds this
    improvement_threshold: float = 0.01  # Minimum improvement to continue
    stability_threshold: float = 0.02  # Maximum variance for stability
    patience: int = 5  # Steps without improvement before stopping


class OptimizationTracker:
    """Track optimization progress and provide analytics."""
    
    def __init__(self, experiment_name: str, save_dir: Optional[str] = None):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir or "optimization_results")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.steps: List[OptimizationStep] = []
        self.best_step: Optional[OptimizationStep] = None
        self.convergence_criteria = ConvergenceCriteria()
        
        # Performance tracking
        self.utility_history: List[float] = []
        self.improvement_history: List[float] = []
        self.moving_averages: Dict[int, List[float]] = {5: [], 10: []}
        
    def add_step(self, prompt: str, utility: float, 
                 metrics: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Add a new optimization step."""
        step = OptimizationStep(
            step_number=len(self.steps),
            timestamp=datetime.now(),
            prompt=prompt,
            utility=utility,
            metrics=metrics or {},
            metadata=metadata or {}
        )
        
        self.steps.append(step)
        self.utility_history.append(utility)
        
        # Update best step
        if self.best_step is None or utility > self.best_step.utility:
            self.best_step = step
        
        # Calculate improvement
        if len(self.steps) > 1:
            improvement = utility - self.utility_history[-2]
            self.improvement_history.append(improvement)
        
        # Update moving averages
        for window, avg_list in self.moving_averages.items():
            if len(self.utility_history) >= window:
                avg = np.mean(self.utility_history[-window:])
                avg_list.append(avg)
    
    def check_convergence(self) -> Tuple[bool, str]:
        """Check if optimization has converged."""
        if len(self.steps) < self.convergence_criteria.min_steps:
            return False, "Insufficient steps"
        
        # Check utility threshold
        if self.utility_history[-1] >= self.convergence_criteria.utility_threshold:
            return True, "Utility threshold reached"
        
        # Check improvement over window
        window = self.convergence_criteria.window_size
        if len(self.improvement_history) >= window:
            recent_improvements = self.improvement_history[-window:]
            avg_improvement = np.mean(recent_improvements)
            
            if avg_improvement < self.convergence_criteria.improvement_threshold:
                # Check if we've been patient enough
                patience_window = self.improvement_history[-self.convergence_criteria.patience:]
                if all(imp < self.convergence_criteria.improvement_threshold 
                       for imp in patience_window):
                    return True, "No significant improvement"
        
        # Check stability (low variance)
        if len(self.utility_history) >= window:
            recent_utilities = self.utility_history[-window:]
            variance = np.var(recent_utilities)
            
            if variance < self.convergence_criteria.stability_threshold:
                return True, "Optimization stabilized"
        
        return False, "Optimization continuing"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        if not self.steps:
            return {}
        
        convergence_result = self.check_convergence()
        stats = {
            'total_steps': len(self.steps),
            'best_utility': float(self.best_step.utility) if self.best_step else 0.0,
            'best_step_number': int(self.best_step.step_number) if self.best_step else 0,
            'final_utility': float(self.utility_history[-1]),
            'total_improvement': float(self.utility_history[-1] - self.utility_history[0]),
            'average_utility': float(np.mean(self.utility_history)),
            'utility_std': float(np.std(self.utility_history)),
            'convergence_status': {
                'converged': bool(convergence_result[0]),
                'reason': str(convergence_result[1])
            }
        }
        
        # Add improvement statistics
        if self.improvement_history:
            trend_coeff = np.polyfit(
                range(len(self.improvement_history)), 
                self.improvement_history, 1
            )[0]  # Linear trend coefficient
            stats.update({
                'average_improvement': float(np.mean(self.improvement_history)),
                'max_improvement': float(max(self.improvement_history)),
                'min_improvement': float(min(self.improvement_history)),
                'improvement_trend': float(trend_coeff)
            })
        
        return stats
    
    def plot_optimization_progress(self, save_path: Optional[str] = None):
        """Plot optimization progress."""
        if not self.steps:
            logger.warning("No optimization steps to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Optimization Progress: {self.experiment_name}', fontsize=16)
        
        # Plot 1: Utility over steps
        ax1 = axes[0, 0]
        ax1.plot(self.utility_history, 'b-', label='Utility', linewidth=2)
        if self.best_step:
            ax1.axhline(y=self.best_step.utility, color='g', linestyle='--', 
                       label=f'Best: {self.best_step.utility:.3f}')
            ax1.plot(self.best_step.step_number, self.best_step.utility, 
                    'g*', markersize=15)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Utility')
        ax1.set_title('Utility Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement per step
        ax2 = axes[0, 1]
        if self.improvement_history:
            ax2.plot(self.improvement_history, 'r-', label='Step Improvement')
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.axhline(y=self.convergence_criteria.improvement_threshold, 
                       color='orange', linestyle='--', 
                       label=f'Threshold: {self.convergence_criteria.improvement_threshold}')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Improvement')
        ax2.set_title('Step-wise Improvement')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Moving averages
        ax3 = axes[1, 0]
        ax3.plot(self.utility_history, 'b-', alpha=0.3, label='Raw Utility')
        for window, avg_list in self.moving_averages.items():
            if avg_list:
                x_offset = window - 1
                ax3.plot(range(x_offset, len(avg_list) + x_offset), 
                        avg_list, label=f'{window}-step MA', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Utility')
        ax3.set_title('Moving Averages')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Convergence indicators
        ax4 = axes[1, 1]
        if len(self.utility_history) > self.convergence_criteria.window_size:
            # Calculate rolling variance
            window = self.convergence_criteria.window_size
            rolling_var = [
                np.var(self.utility_history[i:i+window])
                for i in range(len(self.utility_history) - window + 1)
            ]
            ax4.plot(range(window-1, len(self.utility_history)), 
                    rolling_var, 'purple', label='Rolling Variance')
            ax4.axhline(y=self.convergence_criteria.stability_threshold,
                       color='orange', linestyle='--',
                       label=f'Stability Threshold: {self.convergence_criteria.stability_threshold}')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Variance')
        ax4.set_title('Convergence Indicators')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.save_dir / f"{self.experiment_name}_progress.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Saved optimization plot to {save_path}")
    
    def save_results(self):
        """Save optimization results to disk."""
        results = {
            'experiment_name': self.experiment_name,
            'statistics': self.get_statistics(),
            'best_prompt': self.best_step.prompt if self.best_step else None,
            'convergence_criteria': {
                'min_steps': self.convergence_criteria.min_steps,
                'window_size': self.convergence_criteria.window_size,
                'utility_threshold': self.convergence_criteria.utility_threshold,
                'improvement_threshold': self.convergence_criteria.improvement_threshold,
                'stability_threshold': self.convergence_criteria.stability_threshold,
                'patience': self.convergence_criteria.patience
            },
            'steps': [
                {
                    'step_number': step.step_number,
                    'timestamp': step.timestamp.isoformat(),
                    'utility': step.utility,
                    'metrics': step.metrics,
                    'metadata': step.metadata
                }
                for step in self.steps
            ],
            'utility_history': self.utility_history,
            'improvement_history': self.improvement_history
        }
        
        save_path = self.save_dir / f"{self.experiment_name}_results.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved optimization results to {save_path}")
        
        # Also save the best prompt separately
        if self.best_step:
            best_prompt_path = self.save_dir / f"{self.experiment_name}_best_prompt.txt"
            with open(best_prompt_path, 'w') as f:
                f.write(f"# Best Prompt (Utility: {self.best_step.utility:.4f})\n\n")
                f.write(self.best_step.prompt)
            
            logger.info(f"Saved best prompt to {best_prompt_path}")


class ConvergenceAnalyzer:
    """Advanced convergence analysis and stopping criteria."""
    
    @staticmethod
    def analyze_convergence_rate(utility_history: List[float]) -> Dict[str, float]:
        """Analyze the rate of convergence."""
        if len(utility_history) < 2:
            return {}
        
        # Fit exponential decay model: u(t) = a * (1 - exp(-b * t)) + c
        from scipy.optimize import curve_fit
        
        def exp_model(t, a, b, c):
            return a * (1 - np.exp(-b * t)) + c
        
        t = np.arange(len(utility_history))
        
        try:
            # Initial guess
            a_guess = max(utility_history) - min(utility_history)
            c_guess = min(utility_history)
            b_guess = 0.1
            
            popt, _ = curve_fit(exp_model, t, utility_history, 
                               p0=[a_guess, b_guess, c_guess],
                               maxfev=5000)
            
            a, b, c = popt
            
            # Calculate convergence metrics
            asymptotic_value = a + c
            half_life = np.log(2) / b if b > 0 else float('inf')
            time_to_90_percent = np.log(10) / b if b > 0 else float('inf')
            
            # Calculate R-squared
            y_pred = exp_model(t, a, b, c)
            ss_res = np.sum((utility_history - y_pred) ** 2)
            ss_tot = np.sum((utility_history - np.mean(utility_history)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'asymptotic_value': asymptotic_value,
                'convergence_rate': b,
                'half_life': half_life,
                'time_to_90_percent': time_to_90_percent,
                'r_squared': r_squared,
                'model_fit': 'exponential'
            }
            
        except:
            # Fallback to linear analysis
            slope, intercept = np.polyfit(t, utility_history, 1)
            
            return {
                'linear_slope': slope,
                'linear_intercept': intercept,
                'model_fit': 'linear'
            }
    
    @staticmethod
    def detect_plateaus(utility_history: List[float], 
                       window_size: int = 5,
                       threshold: float = 0.01) -> List[Tuple[int, int]]:
        """Detect plateaus in the optimization curve."""
        plateaus = []
        
        if len(utility_history) < window_size:
            return plateaus
        
        i = 0
        while i < len(utility_history) - window_size:
            window = utility_history[i:i + window_size]
            
            if np.std(window) < threshold:
                # Found start of plateau
                start = i
                
                # Find end of plateau
                j = i + window_size
                while j < len(utility_history):
                    if abs(utility_history[j] - np.mean(window)) > threshold:
                        break
                    j += 1
                
                end = j - 1
                plateaus.append((start, end))
                i = j
            else:
                i += 1
        
        return plateaus
    
    @staticmethod
    def suggest_parameter_adjustment(tracker: OptimizationTracker) -> Dict[str, Any]:
        """Suggest parameter adjustments based on optimization progress."""
        suggestions = {}
        
        stats = tracker.get_statistics()
        
        # Check if we're making progress
        if 'average_improvement' in stats:
            avg_improvement = stats['average_improvement']
            
            if avg_improvement < 0.001:
                suggestions['exploration'] = "Increase exploration - optimization is stuck"
                suggestions['mutation_rate'] = 1.5  # Increase by 50%
            elif avg_improvement > 0.05:
                suggestions['exploration'] = "Decrease exploration - good progress"
                suggestions['mutation_rate'] = 0.7  # Decrease by 30%
        
        # Check for plateaus
        plateaus = ConvergenceAnalyzer.detect_plateaus(tracker.utility_history)
        if plateaus:
            recent_plateau = plateaus[-1]
            plateau_length = recent_plateau[1] - recent_plateau[0]
            
            if plateau_length > 10:
                suggestions['action'] = "Break out of plateau"
                suggestions['restart_with_perturbation'] = True
        
        # Analyze convergence rate
        conv_analysis = ConvergenceAnalyzer.analyze_convergence_rate(
            tracker.utility_history
        )
        
        if 'convergence_rate' in conv_analysis:
            rate = conv_analysis['convergence_rate']
            
            if rate < 0.05:
                suggestions['convergence'] = "Slow convergence detected"
                suggestions['increase_learning_rate'] = True
            elif rate > 0.5:
                suggestions['convergence'] = "Fast convergence - might miss optima"
                suggestions['decrease_learning_rate'] = True
        
        return suggestions
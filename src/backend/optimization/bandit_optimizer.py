"""
Multi-armed bandit algorithms for prompt optimization.

Implements UCB (Upper Confidence Bound) and Thompson Sampling
for balancing exploration and exploitation in prompt selection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import json
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class ArmStatistics:
    """Track statistics for each arm (prompt variant)."""
    prompt: str
    pulls: int = 0
    total_reward: float = 0.0
    rewards: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mean_reward(self) -> float:
        """Calculate mean reward."""
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0
    
    @property
    def variance(self) -> float:
        """Calculate variance of rewards."""
        if self.pulls < 2:
            return 0.0
        mean = self.mean_reward
        return sum((r - mean) ** 2 for r in self.rewards) / (self.pulls - 1)
    
    @property
    def std_dev(self) -> float:
        """Calculate standard deviation."""
        return math.sqrt(self.variance)


class BanditOptimizer(ABC):
    """Base class for multi-armed bandit optimizers."""
    
    def __init__(self, exploration_factor: float = 1.0):
        self.exploration_factor = exploration_factor
        self.arms: Dict[str, ArmStatistics] = {}
        self.total_pulls = 0
        self.history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def select_arm(self) -> Tuple[str, ArmStatistics]:
        """Select which arm (prompt) to pull next."""
        pass
    
    def add_arm(self, arm_id: str, prompt: str, metadata: Optional[Dict] = None):
        """Add a new arm to the bandit."""
        if arm_id not in self.arms:
            self.arms[arm_id] = ArmStatistics(
                prompt=prompt,
                metadata=metadata or {}
            )
    
    def update(self, arm_id: str, reward: float, context: Optional[Dict] = None):
        """Update arm statistics after observing reward."""
        if arm_id not in self.arms:
            raise ValueError(f"Unknown arm: {arm_id}")
            
        arm = self.arms[arm_id]
        arm.pulls += 1
        arm.total_reward += reward
        arm.rewards.append(reward)
        self.total_pulls += 1
        
        # Record history
        self.history.append({
            'arm_id': arm_id,
            'reward': reward,
            'pull_number': self.total_pulls,
            'context': context or {},
            'mean_reward': arm.mean_reward,
            'std_dev': arm.std_dev
        })
    
    def get_best_arm(self) -> Tuple[str, ArmStatistics]:
        """Get the arm with highest mean reward."""
        if not self.arms:
            raise ValueError("No arms available")
        
        best_id = max(self.arms.keys(), key=lambda x: self.arms[x].mean_reward)
        return best_id, self.arms[best_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all arms."""
        stats = {}
        for arm_id, arm in self.arms.items():
            stats[arm_id] = {
                'pulls': arm.pulls,
                'mean_reward': arm.mean_reward,
                'std_dev': arm.std_dev,
                'total_reward': arm.total_reward,
                'confidence_interval': self._confidence_interval(arm)
            }
        return stats
    
    def _confidence_interval(self, arm: ArmStatistics, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for arm's mean reward."""
        if arm.pulls == 0:
            return (0.0, 1.0)
        
        # Use normal approximation
        z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99% CI
        margin = z_score * arm.std_dev / math.sqrt(arm.pulls)
        
        return (
            max(0.0, arm.mean_reward - margin),
            min(1.0, arm.mean_reward + margin)
        )
    
    def save_state(self, filepath: str):
        """Save optimizer state to file."""
        state = {
            'exploration_factor': self.exploration_factor,
            'total_pulls': self.total_pulls,
            'arms': {
                arm_id: {
                    'prompt': arm.prompt,
                    'pulls': arm.pulls,
                    'total_reward': arm.total_reward,
                    'rewards': arm.rewards,
                    'metadata': arm.metadata
                }
                for arm_id, arm in self.arms.items()
            },
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load optimizer state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.exploration_factor = state['exploration_factor']
        self.total_pulls = state['total_pulls']
        self.history = state['history']
        
        self.arms = {}
        for arm_id, arm_data in state['arms'].items():
            arm = ArmStatistics(
                prompt=arm_data['prompt'],
                pulls=arm_data['pulls'],
                total_reward=arm_data['total_reward'],
                rewards=arm_data['rewards'],
                metadata=arm_data.get('metadata', {})
            )
            self.arms[arm_id] = arm


class UCBOptimizer(BanditOptimizer):
    """Upper Confidence Bound (UCB) optimizer.
    
    Balances exploration and exploitation by selecting arms based on
    their upper confidence bound.
    """
    
    def select_arm(self) -> Tuple[str, ArmStatistics]:
        """Select arm using UCB algorithm."""
        if not self.arms:
            raise ValueError("No arms available")
        
        # If any arm hasn't been pulled, pull it first
        for arm_id, arm in self.arms.items():
            if arm.pulls == 0:
                return arm_id, arm
        
        # Calculate UCB for each arm
        ucb_scores = {}
        for arm_id, arm in self.arms.items():
            # UCB = mean + sqrt(2 * ln(total_pulls) / arm_pulls) * exploration_factor
            exploration_term = math.sqrt(2 * math.log(self.total_pulls) / arm.pulls)
            ucb_scores[arm_id] = arm.mean_reward + self.exploration_factor * exploration_term
        
        # Select arm with highest UCB
        best_arm_id = max(ucb_scores.keys(), key=lambda x: ucb_scores[x])
        return best_arm_id, self.arms[best_arm_id]


class ThompsonSamplingOptimizer(BanditOptimizer):
    """Thompson Sampling optimizer using Beta distribution.
    
    Maintains a Beta distribution for each arm and samples from it
    to make decisions.
    """
    
    def __init__(self, exploration_factor: float = 1.0, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        super().__init__(exploration_factor)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        # Track successes and failures for Beta distribution
        self.successes: Dict[str, float] = defaultdict(lambda: prior_alpha)
        self.failures: Dict[str, float] = defaultdict(lambda: prior_beta)
    
    def update(self, arm_id: str, reward: float, context: Optional[Dict] = None):
        """Update arm statistics and Beta distribution parameters."""
        super().update(arm_id, reward, context)
        
        # Interpret reward as success/failure for Beta distribution
        # Assume reward is in [0, 1]
        self.successes[arm_id] += reward
        self.failures[arm_id] += (1 - reward)
    
    def select_arm(self) -> Tuple[str, ArmStatistics]:
        """Select arm using Thompson Sampling."""
        if not self.arms:
            raise ValueError("No arms available")
        
        # Sample from Beta distribution for each arm
        samples = {}
        for arm_id in self.arms.keys():
            alpha = self.successes[arm_id]
            beta = self.failures[arm_id]
            # Apply exploration factor by scaling the variance
            if self.exploration_factor != 1.0:
                # Reduce parameters to increase variance (more exploration)
                scale = 1 / self.exploration_factor
                alpha = max(1.0, alpha * scale)
                beta = max(1.0, beta * scale)
            
            samples[arm_id] = np.random.beta(alpha, beta)
        
        # Select arm with highest sample
        best_arm_id = max(samples.keys(), key=lambda x: samples[x])
        return best_arm_id, self.arms[best_arm_id]
    
    def get_distribution_params(self, arm_id: str) -> Tuple[float, float]:
        """Get Beta distribution parameters for an arm."""
        return self.successes[arm_id], self.failures[arm_id]
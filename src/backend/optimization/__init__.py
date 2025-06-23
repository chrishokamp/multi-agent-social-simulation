"""
Enhanced self-optimization framework for multi-agent simulations.

This module provides advanced optimization strategies including:
- Multi-armed bandit algorithms (UCB, Thompson Sampling)
- Structured prompt templates and mutations
- Bayesian optimization
- Meta-learning capabilities
"""

from .bandit_optimizer import BanditOptimizer, UCBOptimizer, ThompsonSamplingOptimizer
from .prompt_templates import PromptTemplate, PromptMutator, PromptCrossover
from .meta_learner import MetaLearner, TransferLearning
from .optimization_tracker import OptimizationTracker, ConvergenceAnalyzer

__all__ = [
    'BanditOptimizer',
    'UCBOptimizer', 
    'ThompsonSamplingOptimizer',
    'PromptTemplate',
    'PromptMutator',
    'PromptCrossover',
    'MetaLearner',
    'TransferLearning',
    'OptimizationTracker',
    'ConvergenceAnalyzer'
]
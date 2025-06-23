"""
Meta-learning capabilities for cross-simulation optimization.

Learns transferable optimization strategies and maintains a knowledge base
of effective prompt patterns across different simulation types.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class PromptPattern:
    """Represents a reusable prompt pattern."""
    pattern_id: str
    content: str
    domain_tags: Set[str] = field(default_factory=set)
    performance_scores: Dict[str, float] = field(default_factory=dict)  # simulation_type -> score
    usage_count: int = 0
    success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_performance(self, simulation_type: str, score: float):
        """Update performance score for a simulation type."""
        if simulation_type not in self.performance_scores:
            self.performance_scores[simulation_type] = score
        else:
            # Exponential moving average
            alpha = 0.3
            self.performance_scores[simulation_type] = (
                alpha * score + (1 - alpha) * self.performance_scores[simulation_type]
            )
        
        # Update overall success rate
        self.usage_count += 1
        success_threshold = 0.7
        successes = sum(1 for s in self.performance_scores.values() if s > success_threshold)
        self.success_rate = successes / len(self.performance_scores) if self.performance_scores else 0


@dataclass 
class SimulationContext:
    """Context information for a simulation."""
    simulation_type: str
    domain: str
    objectives: List[str]
    constraints: List[str]
    agent_types: List[str]
    output_variables: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_feature_vector(self) -> str:
        """Convert context to text for feature extraction."""
        parts = [
            f"Type: {self.simulation_type}",
            f"Domain: {self.domain}",
            f"Objectives: {' '.join(self.objectives)}",
            f"Constraints: {' '.join(self.constraints)}",
            f"Agents: {' '.join(self.agent_types)}",
            f"Outputs: {' '.join(self.output_variables)}"
        ]
        return " ".join(parts)


class MetaLearner:
    """Meta-learner for cross-simulation optimization."""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path or "meta_learning_kb.pkl"
        self.prompt_patterns: Dict[str, PromptPattern] = {}
        self.context_history: List[SimulationContext] = []
        self.pattern_clusters: Optional[Dict[int, List[str]]] = None
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.is_fitted = False
        
        # Load existing knowledge base if available
        self.load_knowledge_base()
    
    def learn_from_simulation(self, context: SimulationContext, 
                            prompt: str, performance: float):
        """Learn from a simulation run."""
        # Extract patterns from successful prompts
        if performance > 0.7:  # Success threshold
            patterns = self._extract_patterns(prompt)
            
            for pattern_content in patterns:
                pattern_id = self._hash_pattern(pattern_content)
                
                if pattern_id not in self.prompt_patterns:
                    self.prompt_patterns[pattern_id] = PromptPattern(
                        pattern_id=pattern_id,
                        content=pattern_content,
                        domain_tags={context.domain, context.simulation_type}
                    )
                else:
                    # Update existing pattern
                    self.prompt_patterns[pattern_id].domain_tags.update({
                        context.domain, context.simulation_type
                    })
                
                # Update performance
                self.prompt_patterns[pattern_id].update_performance(
                    context.simulation_type, performance
                )
        
        # Store context for similarity matching
        self.context_history.append(context)
        
        # Rebuild clusters periodically
        if len(self.context_history) % 10 == 0:
            self._rebuild_clusters()
    
    def suggest_prompt_patterns(self, context: SimulationContext, 
                              top_k: int = 5) -> List[PromptPattern]:
        """Suggest relevant prompt patterns for a new simulation."""
        # Find similar contexts
        similar_contexts = self._find_similar_contexts(context, k=10)
        
        # Collect patterns used in similar contexts
        candidate_patterns = set()
        for sim_context, similarity in similar_contexts:
            # Get patterns that performed well in this context
            for pattern_id, pattern in self.prompt_patterns.items():
                if sim_context.simulation_type in pattern.performance_scores:
                    score = pattern.performance_scores[sim_context.simulation_type]
                    if score > 0.6:  # Performance threshold
                        candidate_patterns.add(pattern_id)
        
        # Rank patterns by expected performance
        ranked_patterns = []
        for pattern_id in candidate_patterns:
            pattern = self.prompt_patterns[pattern_id]
            expected_score = self._estimate_performance(pattern, context, similar_contexts)
            ranked_patterns.append((pattern, expected_score))
        
        # Sort by expected score
        ranked_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, _ in ranked_patterns[:top_k]]
    
    def _extract_patterns(self, prompt: str) -> List[str]:
        """Extract reusable patterns from a prompt."""
        patterns = []
        
        # Split into sentences
        sentences = prompt.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Extract different types of patterns
            
            # 1. Objective patterns (containing goal-oriented keywords)
            objective_keywords = ['maximize', 'minimize', 'optimize', 'achieve', 'reach', 'ensure']
            if any(keyword in sentence.lower() for keyword in objective_keywords):
                patterns.append(sentence)
            
            # 2. Strategy patterns (containing strategy keywords) 
            strategy_keywords = ['strategy', 'approach', 'method', 'technique', 'focus on', 'prioritize']
            if any(keyword in sentence.lower() for keyword in strategy_keywords):
                patterns.append(sentence)
            
            # 3. Constraint patterns
            constraint_keywords = ['must', 'should', 'cannot', 'avoid', 'never', 'always']
            if any(keyword in sentence.lower() for keyword in constraint_keywords):
                patterns.append(sentence)
        
        # Also extract structural patterns (e.g., "If X then Y")
        if_then_pattern = r'[Ii]f\s+(.+?)\s+then\s+(.+)'
        import re
        for match in re.finditer(if_then_pattern, prompt):
            patterns.append(match.group(0))
        
        return patterns
    
    def _hash_pattern(self, pattern: str) -> str:
        """Generate a hash ID for a pattern."""
        import hashlib
        return hashlib.md5(pattern.encode()).hexdigest()[:8]
    
    def _find_similar_contexts(self, context: SimulationContext, 
                              k: int = 5) -> List[Tuple[SimulationContext, float]]:
        """Find similar simulation contexts."""
        if not self.context_history:
            return []
        
        # Convert contexts to feature vectors
        all_contexts = self.context_history + [context]
        context_texts = [ctx.to_feature_vector() for ctx in all_contexts]
        
        # Fit vectorizer if needed
        if not self.is_fitted:
            self.vectorizer.fit(context_texts)
            self.is_fitted = True
        
        # Transform to vectors
        try:
            context_vectors = self.vectorizer.transform(context_texts)
        except:
            # Re-fit if vocabulary changed
            self.vectorizer.fit(context_texts)
            self.is_fitted = True
            context_vectors = self.vectorizer.transform(context_texts)
        
        # Calculate similarities
        target_vector = context_vectors[-1]
        similarities = cosine_similarity(target_vector, context_vectors[:-1])[0]
        
        # Get top k similar contexts
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        similar_contexts = [
            (self.context_history[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return similar_contexts
    
    def _estimate_performance(self, pattern: PromptPattern, context: SimulationContext,
                            similar_contexts: List[Tuple[SimulationContext, float]]) -> float:
        """Estimate expected performance of a pattern in a new context."""
        if not similar_contexts:
            return pattern.success_rate
        
        # Weighted average based on context similarity
        weighted_sum = 0.0
        weight_total = 0.0
        
        for sim_context, similarity in similar_contexts:
            if sim_context.simulation_type in pattern.performance_scores:
                score = pattern.performance_scores[sim_context.simulation_type]
                weighted_sum += score * similarity
                weight_total += similarity
        
        if weight_total > 0:
            estimated_score = weighted_sum / weight_total
        else:
            estimated_score = pattern.success_rate
        
        # Adjust for domain match
        domain_bonus = 0.1 if context.domain in pattern.domain_tags else 0
        
        return min(1.0, estimated_score + domain_bonus)
    
    def _rebuild_clusters(self):
        """Rebuild pattern clusters for better organization."""
        if len(self.prompt_patterns) < 5:
            return
        
        # Extract pattern texts
        pattern_ids = list(self.prompt_patterns.keys())
        pattern_texts = [self.prompt_patterns[pid].content for pid in pattern_ids]
        
        # Vectorize patterns
        try:
            pattern_vectors = self.vectorizer.transform(pattern_texts)
        except:
            # Re-fit if needed
            self.vectorizer.fit(pattern_texts)
            pattern_vectors = self.vectorizer.transform(pattern_texts)
        
        # Cluster patterns
        n_clusters = min(5, len(pattern_ids) // 3)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(pattern_vectors)
            
            # Store clusters
            self.pattern_clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                self.pattern_clusters[label].append(pattern_ids[idx])
    
    def save_knowledge_base(self):
        """Save the knowledge base to disk."""
        kb_data = {
            'prompt_patterns': self.prompt_patterns,
            'context_history': self.context_history,
            'pattern_clusters': self.pattern_clusters,
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted
        }
        
        Path(self.knowledge_base_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.knowledge_base_path, 'wb') as f:
            pickle.dump(kb_data, f)
        
        logger.info(f"Saved knowledge base with {len(self.prompt_patterns)} patterns")
    
    def load_knowledge_base(self):
        """Load knowledge base from disk."""
        if Path(self.knowledge_base_path).exists():
            try:
                with open(self.knowledge_base_path, 'rb') as f:
                    kb_data = pickle.load(f)
                
                self.prompt_patterns = kb_data.get('prompt_patterns', {})
                self.context_history = kb_data.get('context_history', [])
                self.pattern_clusters = kb_data.get('pattern_clusters')
                self.vectorizer = kb_data.get('vectorizer', TfidfVectorizer(max_features=100))
                self.is_fitted = kb_data.get('is_fitted', False)
                
                logger.info(f"Loaded knowledge base with {len(self.prompt_patterns)} patterns")
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learned patterns."""
        stats = {
            'total_patterns': len(self.prompt_patterns),
            'total_contexts': len(self.context_history),
            'domains': set(),
            'simulation_types': set(),
            'avg_success_rate': 0.0,
            'top_patterns': []
        }
        
        success_rates = []
        for pattern in self.prompt_patterns.values():
            stats['domains'].update(pattern.domain_tags)
            for sim_type in pattern.performance_scores.keys():
                stats['simulation_types'].add(sim_type)
            
            if pattern.success_rate > 0:
                success_rates.append(pattern.success_rate)
        
        if success_rates:
            stats['avg_success_rate'] = np.mean(success_rates)
        
        # Get top performing patterns
        sorted_patterns = sorted(
            self.prompt_patterns.values(),
            key=lambda p: p.success_rate * p.usage_count,
            reverse=True
        )
        
        stats['top_patterns'] = [
            {
                'content': p.content,
                'success_rate': p.success_rate,
                'usage_count': p.usage_count,
                'domains': list(p.domain_tags)
            }
            for p in sorted_patterns[:5]
        ]
        
        # Convert sets to lists for JSON serialization
        stats['domains'] = list(stats['domains'])
        stats['simulation_types'] = list(stats['simulation_types'])
        
        return stats


class TransferLearning:
    """Transfer learning utilities for applying knowledge across domains."""
    
    @staticmethod
    def adapt_pattern_to_context(pattern: PromptPattern, 
                               source_context: SimulationContext,
                               target_context: SimulationContext) -> str:
        """Adapt a pattern from one context to another."""
        adapted_content = pattern.content
        
        # Replace domain-specific terms
        domain_mappings = TransferLearning._get_domain_mappings(
            source_context.domain, target_context.domain
        )
        
        for source_term, target_term in domain_mappings.items():
            adapted_content = adapted_content.replace(source_term, target_term)
        
        # Adjust objectives if needed
        if source_context.objectives != target_context.objectives:
            adapted_content = TransferLearning._adapt_objectives(
                adapted_content, source_context.objectives, target_context.objectives
            )
        
        return adapted_content
    
    @staticmethod
    def _get_domain_mappings(source_domain: str, target_domain: str) -> Dict[str, str]:
        """Get term mappings between domains."""
        # This would be expanded with a comprehensive domain ontology
        mappings = {
            ('negotiation', 'auction'): {
                'buyer': 'bidder',
                'seller': 'auctioneer',
                'price': 'bid',
                'deal': 'winning bid'
            },
            ('debate', 'negotiation'): {
                'argument': 'offer',
                'position': 'price point',
                'convince': 'persuade',
                'opponent': 'counterparty'
            }
        }
        
        key = (source_domain.lower(), target_domain.lower())
        return mappings.get(key, {})
    
    @staticmethod
    def _adapt_objectives(content: str, source_objectives: List[str], 
                         target_objectives: List[str]) -> str:
        """Adapt objectives in the content."""
        # Simple keyword replacement - would be more sophisticated in production
        for source_obj in source_objectives:
            if source_obj in content:
                # Find most similar target objective
                similarities = [
                    TransferLearning._text_similarity(source_obj, target_obj)
                    for target_obj in target_objectives
                ]
                
                if similarities:
                    best_match_idx = np.argmax(similarities)
                    if similarities[best_match_idx] > 0.3:
                        content = content.replace(source_obj, target_objectives[best_match_idx])
        
        return content
    
    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
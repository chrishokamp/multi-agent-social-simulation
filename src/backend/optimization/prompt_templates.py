"""
Structured prompt templates and mutation strategies.

Provides a framework for creating, mutating, and combining prompts
in a systematic way to explore the prompt space effectively.
"""

import random
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from enum import Enum


class ComponentType(Enum):
    """Types of prompt components."""
    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"
    STRATEGY = "strategy"
    PERSONALITY = "personality"
    CONTEXT = "context"
    OUTPUT_FORMAT = "output_format"
    EXAMPLE = "example"


@dataclass
class PromptComponent:
    """Individual component of a prompt."""
    type: ComponentType
    content: str
    weight: float = 1.0  # Importance weight
    metadata: Dict[str, Any] = field(default_factory=dict)
    variations: List[str] = field(default_factory=list)  # Alternative phrasings
    
    def mutate(self, mutation_rate: float = 0.1) -> 'PromptComponent':
        """Create a mutated version of this component."""
        if random.random() < mutation_rate and self.variations:
            # Use a variation
            new_content = random.choice(self.variations)
        else:
            # Apply text mutations
            new_content = self._mutate_text(self.content, mutation_rate)
        
        return PromptComponent(
            type=self.type,
            content=new_content,
            weight=self.weight * (1 + random.uniform(-0.1, 0.1)),  # Slight weight adjustment
            metadata=self.metadata.copy(),
            variations=self.variations.copy()
        )
    
    def _mutate_text(self, text: str, mutation_rate: float) -> str:
        """Apply various text mutations."""
        mutations = []
        
        # Synonym replacement
        if random.random() < mutation_rate:
            mutations.append(self._synonym_replacement)
        
        # Emphasis addition/removal
        if random.random() < mutation_rate:
            mutations.append(self._emphasis_mutation)
        
        # Sentence reordering
        if random.random() < mutation_rate:
            mutations.append(self._sentence_reorder)
        
        # Apply mutations
        result = text
        for mutation in mutations:
            result = mutation(result)
        
        return result
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        # Simple synonym map - in production, use a proper NLP library
        synonyms = {
            "must": ["should", "need to", "have to", "are required to"],
            "maximize": ["optimize", "increase", "improve", "enhance"],
            "minimize": ["reduce", "decrease", "lower", "diminish"],
            "ensure": ["make sure", "guarantee", "verify", "confirm"],
            "important": ["crucial", "essential", "vital", "key"],
            "quickly": ["rapidly", "swiftly", "promptly", "efficiently"]
        }
        
        result = text
        for word, syns in synonyms.items():
            if word in result.lower():
                replacement = random.choice(syns)
                result = re.sub(r'\b' + word + r'\b', replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _emphasis_mutation(self, text: str) -> str:
        """Add or remove emphasis."""
        emphasis_words = ["very", "extremely", "really", "absolutely", "definitely"]
        
        if any(word in text.lower() for word in emphasis_words):
            # Remove emphasis
            for word in emphasis_words:
                text = re.sub(r'\b' + word + r'\s+', '', text, flags=re.IGNORECASE)
        else:
            # Add emphasis
            words = text.split()
            if len(words) > 3:
                # Find adjectives or verbs to emphasize
                insert_pos = random.randint(1, len(words) - 1)
                emphasis = random.choice(emphasis_words)
                words.insert(insert_pos, emphasis)
                text = ' '.join(words)
        
        return text
    
    def _sentence_reorder(self, text: str) -> str:
        """Reorder sentences if multiple exist."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '. '.join(sentences) + '.'
        
        return text


@dataclass
class PromptTemplate:
    """Structured prompt template with composable components."""
    
    components: List[PromptComponent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_component(self, component: PromptComponent):
        """Add a component to the template."""
        self.components.append(component)
    
    def remove_component(self, component_type: ComponentType):
        """Remove all components of a specific type."""
        self.components = [c for c in self.components if c.type != component_type]
    
    def get_components_by_type(self, component_type: ComponentType) -> List[PromptComponent]:
        """Get all components of a specific type."""
        return [c for c in self.components if c.type == component_type]
    
    def generate_prompt(self, component_separator: str = "\n\n") -> str:
        """Generate the full prompt from components."""
        # Sort components by type order
        type_order = [
            ComponentType.CONTEXT,
            ComponentType.OBJECTIVE,
            ComponentType.PERSONALITY,
            ComponentType.CONSTRAINT,
            ComponentType.STRATEGY,
            ComponentType.EXAMPLE,
            ComponentType.OUTPUT_FORMAT
        ]
        
        sorted_components = sorted(
            self.components,
            key=lambda c: type_order.index(c.type) if c.type in type_order else len(type_order)
        )
        
        # Weight-based filtering (include components based on their weights)
        included_components = []
        for component in sorted_components:
            if random.random() < component.weight:
                included_components.append(component)
        
        # Generate prompt text
        prompt_parts = [component.content for component in included_components]
        return component_separator.join(prompt_parts)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'PromptTemplate':
        """Create a mutated version of this template."""
        new_template = PromptTemplate(metadata=self.metadata.copy())
        
        # Mutate existing components
        for component in self.components:
            if random.random() > mutation_rate * 0.5:  # Component removal
                new_component = component.mutate(mutation_rate)
                new_template.add_component(new_component)
        
        # Potentially add new components
        if random.random() < mutation_rate * 0.3:
            new_template._add_random_component()
        
        return new_template
    
    def _add_random_component(self):
        """Add a random new component."""
        # This would be expanded with a library of component templates
        new_components = [
            PromptComponent(
                ComponentType.STRATEGY,
                "Focus on finding mutually beneficial solutions.",
                variations=["Seek win-win outcomes.", "Look for collaborative solutions."]
            ),
            PromptComponent(
                ComponentType.CONSTRAINT,
                "Be respectful and professional in all interactions.",
                variations=["Maintain a courteous tone.", "Communicate professionally."]
            ),
            PromptComponent(
                ComponentType.PERSONALITY,
                "Be analytical and data-driven in your approach.",
                variations=["Use logical reasoning.", "Base decisions on evidence."]
            )
        ]
        
        self.add_component(random.choice(new_components))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            'components': [
                {
                    'type': c.type.value,
                    'content': c.content,
                    'weight': c.weight,
                    'metadata': c.metadata,
                    'variations': c.variations
                }
                for c in self.components
            ],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create template from dictionary."""
        template = cls(metadata=data.get('metadata', {}))
        
        for comp_data in data.get('components', []):
            component = PromptComponent(
                type=ComponentType(comp_data['type']),
                content=comp_data['content'],
                weight=comp_data.get('weight', 1.0),
                metadata=comp_data.get('metadata', {}),
                variations=comp_data.get('variations', [])
            )
            template.add_component(component)
        
        return template


class PromptMutator:
    """Advanced prompt mutation strategies."""
    
    @staticmethod
    def crossover(template1: PromptTemplate, template2: PromptTemplate, 
                  crossover_rate: float = 0.5) -> Tuple[PromptTemplate, PromptTemplate]:
        """Perform crossover between two templates."""
        child1 = PromptTemplate(metadata={'parents': ['template1', 'template2']})
        child2 = PromptTemplate(metadata={'parents': ['template1', 'template2']})
        
        # Collect all unique component types
        all_types = set()
        for template in [template1, template2]:
            all_types.update(c.type for c in template.components)
        
        # For each component type, randomly assign to children
        for comp_type in all_types:
            comps1 = template1.get_components_by_type(comp_type)
            comps2 = template2.get_components_by_type(comp_type)
            
            if random.random() < crossover_rate:
                # Swap components
                for comp in comps2:
                    child1.add_component(comp)
                for comp in comps1:
                    child2.add_component(comp)
            else:
                # Keep original
                for comp in comps1:
                    child1.add_component(comp)
                for comp in comps2:
                    child2.add_component(comp)
        
        return child1, child2
    
    @staticmethod
    def guided_mutation(template: PromptTemplate, performance_history: List[float],
                       mutation_rate: float = 0.1) -> PromptTemplate:
        """Mutate based on performance history."""
        # Adjust mutation rate based on performance trend
        if len(performance_history) >= 3:
            recent_trend = performance_history[-1] - performance_history[-3]
            if recent_trend < 0:  # Performance declining
                mutation_rate *= 1.5  # Increase exploration
            elif recent_trend > 0.1:  # Performance improving significantly
                mutation_rate *= 0.5  # Reduce exploration
        
        return template.mutate(mutation_rate)
    
    @staticmethod
    def component_importance_update(template: PromptTemplate, 
                                  component_performance: Dict[str, float]) -> PromptTemplate:
        """Update component weights based on their individual performance."""
        new_template = PromptTemplate(metadata=template.metadata.copy())
        
        for component in template.components:
            # Check if we have performance data for similar content
            perf_score = 0.5  # Default neutral score
            for content, score in component_performance.items():
                similarity = PromptMutator._text_similarity(component.content, content)
                if similarity > 0.8:
                    perf_score = score
                    break
            
            # Update weight based on performance
            new_weight = component.weight * (0.5 + perf_score)
            new_component = PromptComponent(
                type=component.type,
                content=component.content,
                weight=min(1.0, max(0.1, new_weight)),  # Keep weights in [0.1, 1.0]
                metadata=component.metadata,
                variations=component.variations
            )
            new_template.add_component(new_component)
        
        return new_template
    
    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Simple text similarity measure."""
        # In production, use proper NLP similarity measures
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class PromptCrossover:
    """Different crossover strategies for prompt templates."""
    
    @staticmethod
    def uniform_crossover(template1: PromptTemplate, template2: PromptTemplate) -> PromptTemplate:
        """Each component is randomly selected from either parent."""
        child = PromptTemplate(metadata={'crossover_type': 'uniform'})
        
        all_components = template1.components + template2.components
        
        # Group by type
        by_type = {}
        for comp in all_components:
            if comp.type not in by_type:
                by_type[comp.type] = []
            by_type[comp.type].append(comp)
        
        # Select one component per type
        for comp_type, components in by_type.items():
            selected = random.choice(components)
            child.add_component(selected)
        
        return child
    
    @staticmethod
    def weighted_crossover(template1: PromptTemplate, template2: PromptTemplate,
                          weight1: float = 0.5) -> PromptTemplate:
        """Crossover with weighted probability of selecting from each parent."""
        child = PromptTemplate(metadata={'crossover_type': 'weighted', 'weight1': weight1})
        
        # Collect components by type from both parents
        types1 = {c.type for c in template1.components}
        types2 = {c.type for c in template2.components}
        all_types = types1.union(types2)
        
        for comp_type in all_types:
            comps1 = template1.get_components_by_type(comp_type)
            comps2 = template2.get_components_by_type(comp_type)
            
            # Select based on weight
            if comps1 and comps2:
                if random.random() < weight1:
                    child.add_component(random.choice(comps1))
                else:
                    child.add_component(random.choice(comps2))
            elif comps1:
                child.add_component(random.choice(comps1))
            elif comps2:
                child.add_component(random.choice(comps2))
        
        return child
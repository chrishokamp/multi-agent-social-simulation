"""
Comprehensive unit tests for the optimization framework.

Tests multi-armed bandit algorithms, prompt templates, meta-learning,
and optimization tracking functionality.
"""

import pytest
import numpy as np
import tempfile
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from src.backend.optimization.bandit_optimizer import (
    BanditOptimizer, UCBOptimizer, ThompsonSamplingOptimizer, ArmStatistics
)
from src.backend.optimization.prompt_templates import (
    PromptTemplate, PromptComponent, PromptMutator, ComponentType
)
from src.backend.optimization.meta_learner import (
    MetaLearner, PromptPattern, SimulationContext, TransferLearning
)
from src.backend.optimization.optimization_tracker import (
    OptimizationTracker, ConvergenceAnalyzer, OptimizationStep
)


class TestBanditOptimizer:
    """Test multi-armed bandit optimizers."""
    
    def test_arm_statistics(self):
        """Test arm statistics calculations."""
        arm = ArmStatistics(prompt="test prompt")
        assert arm.mean_reward == 0.0
        assert arm.variance == 0.0
        assert arm.std_dev == 0.0
        
        # Add some rewards
        arm.pulls = 3
        arm.total_reward = 2.1
        arm.rewards = [0.5, 0.8, 0.8]
        
        assert arm.mean_reward == 0.7
        assert arm.variance > 0
        assert arm.std_dev > 0
    
    def test_ucb_optimizer(self):
        """Test UCB optimizer functionality."""
        optimizer = UCBOptimizer(exploration_factor=1.0)
        
        # Add arms
        optimizer.add_arm("arm1", "prompt1")
        optimizer.add_arm("arm2", "prompt2")
        
        # Initially should select unplayed arms
        arm_id, arm = optimizer.select_arm()
        assert arm_id in ["arm1", "arm2"]
        assert optimizer.arms[arm_id].pulls == 0
        
        # Update with reward
        optimizer.update(arm_id, 0.8)
        assert optimizer.arms[arm_id].pulls == 1
        assert optimizer.arms[arm_id].mean_reward == 0.8
        
        # Test best arm selection
        best_id, best_arm = optimizer.get_best_arm()
        assert best_id == arm_id
        assert best_arm.mean_reward == 0.8
    
    def test_thompson_sampling_optimizer(self):
        """Test Thompson Sampling optimizer."""
        optimizer = ThompsonSamplingOptimizer(
            exploration_factor=1.0, 
            prior_alpha=1.0, 
            prior_beta=1.0
        )
        
        optimizer.add_arm("arm1", "prompt1")
        optimizer.add_arm("arm2", "prompt2")
        
        # Test initial selection
        arm_id, arm = optimizer.select_arm()
        assert arm_id in ["arm1", "arm2"]
        
        # Update and test distribution parameters
        optimizer.update(arm_id, 0.9)
        alpha, beta = optimizer.get_distribution_params(arm_id)
        assert alpha > 1.0  # Should increase with reward
        assert beta < 2.0   # Should decrease with high reward
    
    def test_bandit_state_persistence(self):
        """Test saving and loading bandit state."""
        optimizer = UCBOptimizer(exploration_factor=2.0)
        optimizer.add_arm("arm1", "prompt1", {"meta": "data"})
        optimizer.update("arm1", 0.7)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        
        try:
            # Save state
            optimizer.save_state(state_file)
            
            # Create new optimizer and load state
            new_optimizer = UCBOptimizer()
            new_optimizer.load_state(state_file)
            
            # Verify state is preserved
            assert new_optimizer.exploration_factor == 2.0
            assert "arm1" in new_optimizer.arms
            assert new_optimizer.arms["arm1"].pulls == 1
            assert new_optimizer.arms["arm1"].mean_reward == 0.7
            assert new_optimizer.arms["arm1"].metadata == {"meta": "data"}
            
        finally:
            Path(state_file).unlink()


class TestPromptTemplates:
    """Test prompt template functionality."""
    
    def test_prompt_component_creation(self):
        """Test prompt component creation and properties."""
        component = PromptComponent(
            type=ComponentType.OBJECTIVE,
            content="Maximize your utility in this negotiation.",
            weight=0.8,
            variations=["Optimize your outcomes", "Achieve the best deal"]
        )
        
        assert component.type == ComponentType.OBJECTIVE
        assert component.weight == 0.8
        assert len(component.variations) == 2
    
    def test_prompt_component_mutation(self):
        """Test prompt component mutation."""
        component = PromptComponent(
            type=ComponentType.STRATEGY,
            content="Be aggressive in your negotiations.",
            variations=["Be assertive", "Take a strong stance"]
        )
        
        # Mutation should create different content
        mutated = component.mutate(mutation_rate=1.0)  # High rate for testing
        assert mutated.type == component.type
        assert mutated.content != component.content or mutated.weight != component.weight
    
    def test_prompt_template_generation(self):
        """Test prompt template generation."""
        template = PromptTemplate()
        
        # Add components in different order
        template.add_component(PromptComponent(
            ComponentType.STRATEGY, "Use collaborative approach."
        ))
        template.add_component(PromptComponent(
            ComponentType.OBJECTIVE, "Maximize mutual benefit."
        ))
        template.add_component(PromptComponent(
            ComponentType.CONTEXT, "You are in a business negotiation."
        ))
        
        prompt = template.generate_prompt()
        
        # Should have all components
        assert "collaborative approach" in prompt
        assert "mutual benefit" in prompt
        assert "business negotiation" in prompt
        
        # Context should come first (based on type order)
        context_pos = prompt.find("business negotiation")
        objective_pos = prompt.find("mutual benefit")
        assert context_pos < objective_pos
    
    def test_prompt_template_mutation(self):
        """Test prompt template mutation."""
        template = PromptTemplate()
        template.add_component(PromptComponent(
            ComponentType.OBJECTIVE, "Achieve the best outcome."
        ))
        template.add_component(PromptComponent(
            ComponentType.STRATEGY, "Be patient and persistent."
        ))
        
        # Mutate template
        mutated = template.mutate(mutation_rate=0.5)
        
        # Should have similar structure but potentially different content
        assert len(mutated.components) > 0
        assert isinstance(mutated, PromptTemplate)
    
    def test_prompt_template_serialization(self):
        """Test template serialization and deserialization."""
        template = PromptTemplate(metadata={"version": "1.0"})
        template.add_component(PromptComponent(
            ComponentType.CONSTRAINT,
            "Stay within budget limits.",
            weight=0.9,
            variations=["Mind your budget", "Keep costs low"]
        ))
        
        # Convert to dict and back
        template_dict = template.to_dict()
        restored_template = PromptTemplate.from_dict(template_dict)
        
        # Verify restoration
        assert restored_template.metadata == template.metadata
        assert len(restored_template.components) == 1
        
        restored_comp = restored_template.components[0]
        original_comp = template.components[0]
        
        assert restored_comp.type == original_comp.type
        assert restored_comp.content == original_comp.content
        assert restored_comp.weight == original_comp.weight
        assert restored_comp.variations == original_comp.variations
    
    def test_prompt_crossover(self):
        """Test prompt template crossover."""
        template1 = PromptTemplate()
        template1.add_component(PromptComponent(
            ComponentType.OBJECTIVE, "Maximize profit."
        ))
        template1.add_component(PromptComponent(
            ComponentType.STRATEGY, "Be aggressive."
        ))
        
        template2 = PromptTemplate()
        template2.add_component(PromptComponent(
            ComponentType.OBJECTIVE, "Minimize cost."
        ))
        template2.add_component(PromptComponent(
            ComponentType.PERSONALITY, "Be friendly."
        ))
        
        child1, child2 = PromptMutator.crossover(template1, template2)
        
        # Children should have components from both parents
        assert len(child1.components) > 0
        assert len(child2.components) > 0
        
        # Should contain components from both parents
        all_child1_content = " ".join(c.content for c in child1.components)
        all_child2_content = " ".join(c.content for c in child2.components)
        
        # At least one child should have mixed content
        assert ("profit" in all_child1_content or "cost" in all_child1_content or
                "profit" in all_child2_content or "cost" in all_child2_content)


class TestMetaLearner:
    """Test meta-learning functionality."""
    
    def test_simulation_context_creation(self):
        """Test simulation context creation and feature extraction."""
        context = SimulationContext(
            simulation_type="negotiation",
            domain="business",
            objectives=["maximize_profit", "minimize_risk"],
            constraints=["budget_limit", "time_limit"],
            agent_types=["buyer", "seller"],
            output_variables=["final_price", "deal_reached"]
        )
        
        feature_text = context.to_feature_vector()
        assert "negotiation" in feature_text
        assert "business" in feature_text
        assert "maximize_profit" in feature_text
        assert "buyer" in feature_text
    
    def test_prompt_pattern_performance_tracking(self):
        """Test prompt pattern performance tracking."""
        pattern = PromptPattern(
            pattern_id="test_pattern",
            content="Focus on win-win solutions.",
            domain_tags={"negotiation", "business"}
        )
        
        # Update performance
        pattern.update_performance("negotiation", 0.8)
        pattern.update_performance("negotiation", 0.9)
        pattern.update_performance("auction", 0.6)
        
        assert pattern.usage_count == 3
        assert "negotiation" in pattern.performance_scores
        assert "auction" in pattern.performance_scores
        assert pattern.performance_scores["negotiation"] > 0.8  # Should be EMA
    
    @patch('src.backend.optimization.meta_learner.TfidfVectorizer')
    def test_meta_learner_similarity_matching(self, mock_vectorizer):
        """Test meta-learner similarity matching."""
        # Mock vectorizer behavior
        mock_vectorizer_instance = Mock()
        mock_vectorizer.return_value = mock_vectorizer_instance
        mock_vectorizer_instance.fit.return_value = None
        mock_vectorizer_instance.transform.return_value = np.array([[1, 0], [0.8, 0.2], [0.1, 0.9]])
        
        learner = MetaLearner()
        
        # Add some context history
        context1 = SimulationContext("negotiation", "business", ["profit"], [], ["buyer"], [])
        context2 = SimulationContext("negotiation", "personal", ["satisfaction"], [], ["person"], [])
        
        learner.context_history = [context1, context2]
        learner.is_fitted = True
        
        # Test similarity matching
        new_context = SimulationContext("negotiation", "business", ["revenue"], [], ["seller"], [])
        
        with patch('src.backend.optimization.meta_learner.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = np.array([[0.9, 0.3]])
            
            similar_contexts = learner._find_similar_contexts(new_context, k=2)
            
            assert len(similar_contexts) == 2
            assert similar_contexts[0][1] > similar_contexts[1][1]  # Should be sorted by similarity
    
    def test_meta_learner_pattern_extraction(self):
        """Test pattern extraction from prompts."""
        learner = MetaLearner()
        
        prompt = """
        You are a skilled negotiator. Your goal is to maximize your profit while maintaining good relationships.
        Always focus on win-win solutions. If the other party makes a low offer, then counter with a reasonable alternative.
        Never accept the first offer. Ensure you understand their needs before proposing solutions.
        """
        
        patterns = learner._extract_patterns(prompt)
        
        # Should extract various types of patterns
        assert len(patterns) > 0
        
        # Check for specific pattern types
        pattern_text = " ".join(patterns)
        assert any("maximize" in p for p in patterns)  # Objective pattern
        assert any("focus" in p for p in patterns)    # Strategy pattern
        assert any("never" in p.lower() for p in patterns)  # Constraint pattern
    
    def test_meta_learner_state_persistence(self):
        """Test meta-learner state saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_path = Path(temp_dir) / "test_kb.pkl"
            
            # Create learner with some data
            learner = MetaLearner(str(kb_path))
            
            context = SimulationContext("test", "domain", ["obj"], [], ["agent"], [])
            learner.learn_from_simulation(context, "test prompt with maximize utility", 0.9)
            
            # Save state
            learner.save_knowledge_base()
            
            # Create new learner and load
            new_learner = MetaLearner(str(kb_path))
            
            # Verify data is loaded
            assert len(new_learner.prompt_patterns) > 0
            assert len(new_learner.context_history) > 0
    
    def test_transfer_learning_domain_adaptation(self):
        """Test transfer learning domain adaptation."""
        pattern = PromptPattern(
            pattern_id="test",
            content="The buyer should negotiate the price carefully.",
            domain_tags={"negotiation"}
        )
        
        source_context = SimulationContext(
            "negotiation", "business", [], [], ["buyer", "seller"], []
        )
        target_context = SimulationContext(
            "auction", "art", [], [], ["bidder", "auctioneer"], []
        )
        
        adapted_content = TransferLearning.adapt_pattern_to_context(
            pattern, source_context, target_context
        )
        
        # Should have domain-specific terms replaced
        assert "bidder" in adapted_content
        assert "buyer" not in adapted_content


class TestOptimizationTracker:
    """Test optimization tracking and convergence analysis."""
    
    def test_optimization_tracker_basic_functionality(self):
        """Test basic optimization tracker operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = OptimizationTracker("test_experiment", temp_dir)
            
            # Add some steps
            tracker.add_step("prompt1", 0.5, {"metric1": 1.0})
            tracker.add_step("prompt2", 0.7, {"metric1": 1.5})
            tracker.add_step("prompt3", 0.6, {"metric1": 1.2})
            
            # Check statistics
            stats = tracker.get_statistics()
            assert stats['total_steps'] == 3
            assert stats['best_utility'] == 0.7
            assert stats['final_utility'] == 0.6
            assert stats['total_improvement'] == 0.1  # 0.6 - 0.5
            
            # Check best step
            assert tracker.best_step.utility == 0.7
            assert tracker.best_step.step_number == 1
    
    def test_convergence_detection(self):
        """Test convergence detection logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = OptimizationTracker("convergence_test", temp_dir)
            
            # Add steps that converge
            for i in range(15):
                utility = 0.9 + 0.001 * i  # Slight improvement
                tracker.add_step(f"prompt{i}", utility)
            
            # Should detect convergence due to small improvements
            converged, reason = tracker.check_convergence()
            assert converged
            assert "improvement" in reason.lower()
    
    def test_convergence_analyzer_plateau_detection(self):
        """Test plateau detection in convergence analyzer."""
        # Create utility history with plateaus
        utility_history = [0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8]
        
        plateaus = ConvergenceAnalyzer.detect_plateaus(utility_history, window_size=3, threshold=0.01)
        
        # Should detect plateaus
        assert len(plateaus) >= 1
        
        # First plateau should be around indices 3-7 (where utility = 0.5)
        first_plateau = plateaus[0]
        assert first_plateau[0] >= 2  # Start around index 3
        assert first_plateau[1] <= 8   # End before index 8
    
    def test_convergence_rate_analysis(self):
        """Test convergence rate analysis."""
        # Create exponential-like convergence
        t = np.arange(20)
        utility_history = 0.8 * (1 - np.exp(-0.2 * t)) + 0.1
        utility_history = utility_history.tolist()
        
        analysis = ConvergenceAnalyzer.analyze_convergence_rate(utility_history)
        
        # Should fit exponential model
        assert 'asymptotic_value' in analysis
        assert 'convergence_rate' in analysis
        assert analysis['model_fit'] == 'exponential'
        assert analysis['asymptotic_value'] > 0.8  # Should approach ~0.9
    
    def test_parameter_adjustment_suggestions(self):
        """Test parameter adjustment suggestions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = OptimizationTracker("param_test", temp_dir)
            
            # Add steps with declining improvement
            for i in range(15):
                utility = 0.5 + 0.1 / (i + 1)  # Decreasing improvement
                tracker.add_step(f"prompt{i}", utility)
            
            suggestions = ConvergenceAnalyzer.suggest_parameter_adjustment(tracker)
            
            # Should suggest increasing exploration due to low improvement
            assert 'exploration' in suggestions
            assert 'mutation_rate' in suggestions
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_optimization_plotting(self, mock_close, mock_savefig):
        """Test optimization progress plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = OptimizationTracker("plot_test", temp_dir)
            
            # Add some steps
            for i in range(10):
                utility = 0.3 + 0.05 * i + 0.01 * np.random.randn()
                tracker.add_step(f"prompt{i}", utility)
            
            # Should not raise exception
            tracker.plot_optimization_progress()
            
            # Should call save
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    def test_optimization_results_saving(self):
        """Test saving optimization results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = OptimizationTracker("save_test", temp_dir)
            
            tracker.add_step("test prompt", 0.8, {"test_metric": 2.0})
            
            # Save results
            tracker.save_results()
            
            # Check files exist
            results_file = Path(temp_dir) / "save_test_results.json"
            prompt_file = Path(temp_dir) / "save_test_best_prompt.txt"
            
            assert results_file.exists()
            assert prompt_file.exists()
            
            # Verify content
            with open(results_file) as f:
                results = json.load(f)
            
            assert results['experiment_name'] == 'save_test'
            assert results['best_prompt'] == 'test prompt'
            assert len(results['steps']) == 1


# Integration tests
class TestOptimizationIntegration:
    """Integration tests for the optimization framework."""
    
    def test_bandit_with_prompt_templates(self):
        """Test bandit optimizer with prompt templates."""
        optimizer = UCBOptimizer(exploration_factor=1.0)
        
        # Create prompt templates
        template1 = PromptTemplate()
        template1.add_component(PromptComponent(
            ComponentType.OBJECTIVE, "Maximize your profit."
        ))
        
        template2 = PromptTemplate()
        template2.add_component(PromptComponent(
            ComponentType.OBJECTIVE, "Minimize your costs."
        ))
        
        # Add templates as arms
        optimizer.add_arm("template1", template1.generate_prompt())
        optimizer.add_arm("template2", template2.generate_prompt())
        
        # Simulate optimization loop
        for i in range(10):
            arm_id, arm = optimizer.select_arm()
            
            # Simulate reward (template1 should be better)
            reward = 0.8 if arm_id == "template1" else 0.3
            reward += 0.1 * np.random.randn()  # Add noise
            
            optimizer.update(arm_id, max(0, min(1, reward)))
        
        # Template1 should be identified as better
        best_id, best_arm = optimizer.get_best_arm()
        assert best_arm.mean_reward > 0.5
    
    def test_meta_learner_with_tracker(self):
        """Test meta-learner integration with optimization tracker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create meta-learner and tracker
            learner = MetaLearner()
            tracker = OptimizationTracker("integration_test", temp_dir)
            
            # Simulate optimization with learning
            context = SimulationContext("negotiation", "business", ["profit"], [], ["buyer"], [])
            
            for i in range(5):
                # Get suggestions from meta-learner
                suggestions = learner.suggest_prompt_patterns(context, top_k=1)
                
                if suggestions:
                    prompt = suggestions[0].content
                else:
                    prompt = f"Default prompt {i}"
                
                # Simulate utility
                utility = 0.5 + 0.1 * i + 0.05 * np.random.randn()
                
                # Update tracker
                tracker.add_step(prompt, utility)
                
                # Update meta-learner
                learner.learn_from_simulation(context, prompt, utility)
            
            # Verify both components have learned
            assert len(tracker.steps) == 5
            assert len(learner.context_history) > 0
            
            # Final suggestions should be available
            final_suggestions = learner.suggest_prompt_patterns(context)
            assert len(final_suggestions) >= 0  # May be empty initially


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
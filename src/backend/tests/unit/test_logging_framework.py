"""
Unit tests for the logging framework.
"""
import json
import tempfile
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from logging_framework.core import (
    AgentAction, 
    UtilitySnapshot, 
    AgentLogger, 
    MetricsCollector, 
    SimulationLogger
)
from logging_framework.visualization import SimulationVisualizer
from logging_framework.reporters import SimulationReporter, HTMLReporter


class TestAgentAction:
    """Test AgentAction dataclass."""
    
    def test_agent_action_creation(self):
        """Test creating an AgentAction."""
        timestamp = datetime.now()
        action = AgentAction(
            timestamp=timestamp,
            agent_name="TestAgent",
            action_type="message",
            content="Hello world",
            metadata={"round": 1},
            utility_before=0.5,
            utility_after=0.6
        )
        
        assert action.timestamp == timestamp
        assert action.agent_name == "TestAgent"
        assert action.action_type == "message"
        assert action.content == "Hello world"
        assert action.metadata == {"round": 1}
        assert action.utility_before == 0.5
        assert action.utility_after == 0.6
    
    def test_agent_action_to_dict(self):
        """Test AgentAction serialization."""
        timestamp = datetime.now()
        action = AgentAction(
            timestamp=timestamp,
            agent_name="TestAgent",
            action_type="message",
            content="Hello world"
        )
        
        data = action.to_dict()
        assert data['timestamp'] == timestamp.isoformat()
        assert data['agent_name'] == "TestAgent"
        assert data['action_type'] == "message"
        assert data['content'] == "Hello world"


class TestUtilitySnapshot:
    """Test UtilitySnapshot dataclass."""
    
    def test_utility_snapshot_creation(self):
        """Test creating a UtilitySnapshot."""
        timestamp = datetime.now()
        snapshot = UtilitySnapshot(
            timestamp=timestamp,
            round_number=5,
            agent_name="TestAgent",
            utility_value=0.75,
            environment_state={"final_price": 100}
        )
        
        assert snapshot.timestamp == timestamp
        assert snapshot.round_number == 5
        assert snapshot.agent_name == "TestAgent"
        assert snapshot.utility_value == 0.75
        assert snapshot.environment_state == {"final_price": 100}
    
    def test_utility_snapshot_to_dict(self):
        """Test UtilitySnapshot serialization."""
        timestamp = datetime.now()
        snapshot = UtilitySnapshot(
            timestamp=timestamp,
            round_number=5,
            agent_name="TestAgent",
            utility_value=0.75
        )
        
        data = snapshot.to_dict()
        assert data['timestamp'] == timestamp.isoformat()
        assert data['round_number'] == 5
        assert data['agent_name'] == "TestAgent"
        assert data['utility_value'] == 0.75


class TestAgentLogger:
    """Test AgentLogger class."""
    
    def test_agent_logger_creation(self):
        """Test creating an AgentLogger."""
        logger = AgentLogger("TestAgent", "sim123")
        
        assert logger.agent_name == "TestAgent"
        assert logger.simulation_id == "sim123"
        assert len(logger.actions) == 0
        assert len(logger.utility_history) == 0
    
    def test_log_action(self):
        """Test logging an action."""
        logger = AgentLogger("TestAgent", "sim123")
        
        logger.log_action(
            "message",
            "Hello world",
            metadata={"round": 1},
            utility_before=0.5,
            utility_after=0.6
        )
        
        assert len(logger.actions) == 1
        action = logger.actions[0]
        assert action.action_type == "message"
        assert action.content == "Hello world"
        assert action.metadata == {"round": 1}
        assert action.utility_before == 0.5
        assert action.utility_after == 0.6
    
    def test_log_utility(self):
        """Test logging utility values."""
        logger = AgentLogger("TestAgent", "sim123")
        
        logger.log_utility(1, 0.5, {"final_price": 100})
        logger.log_utility(2, 0.7, {"final_price": 90})
        
        assert len(logger.utility_history) == 2
        assert logger.utility_history[0].utility_value == 0.5
        assert logger.utility_history[1].utility_value == 0.7
    
    def test_get_utility_trend(self):
        """Test getting utility trend."""
        logger = AgentLogger("TestAgent", "sim123")
        
        logger.log_utility(1, 0.5)
        logger.log_utility(2, 0.7)
        logger.log_utility(3, 0.6)
        
        trend = logger.get_utility_trend()
        assert trend == [(1, 0.5), (2, 0.7), (3, 0.6)]
    
    def test_get_actions_by_type(self):
        """Test filtering actions by type."""
        logger = AgentLogger("TestAgent", "sim123")
        
        logger.log_action("message", "Hello")
        logger.log_action("decision", "Choose option A")
        logger.log_action("message", "Goodbye")
        
        messages = logger.get_actions_by_type("message")
        decisions = logger.get_actions_by_type("decision")
        
        assert len(messages) == 2
        assert len(decisions) == 1
        assert messages[0].content == "Hello"
        assert messages[1].content == "Goodbye"
        assert decisions[0].content == "Choose option A"


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_metrics_collector_creation(self):
        """Test creating a MetricsCollector."""
        collector = MetricsCollector()
        assert len(collector.metrics) == 0
    
    def test_record_metric(self):
        """Test recording metrics."""
        collector = MetricsCollector()
        
        collector.record("price", 100)
        collector.record("price", 90)
        collector.record("satisfaction", 0.8)
        
        assert len(collector.metrics) == 2
        assert len(collector.metrics["price"]) == 2
        assert len(collector.metrics["satisfaction"]) == 1
    
    def test_get_metric(self):
        """Test getting metric values."""
        collector = MetricsCollector()
        
        collector.record("price", 100)
        collector.record("price", 90)
        
        price_values = collector.get_metric("price")
        assert len(price_values) == 2
        assert price_values[0][1] == 100
        assert price_values[1][1] == 90
    
    def test_get_summary(self):
        """Test getting metrics summary."""
        collector = MetricsCollector()
        
        collector.record("price", 100)
        collector.record("price", 90)
        collector.record("price", 110)
        collector.record("deal_reached", True)
        
        summary = collector.get_summary()
        
        price_stats = summary["price"]
        assert price_stats["count"] == 3
        assert price_stats["mean"] == 100.0
        assert price_stats["min"] == 90
        assert price_stats["max"] == 110
        assert price_stats["last"] == 110
        
        deal_stats = summary["deal_reached"]
        assert deal_stats["count"] == 1
        assert deal_stats["last"] == True


class TestSimulationLogger:
    """Test SimulationLogger class."""
    
    def test_simulation_logger_creation(self):
        """Test creating a SimulationLogger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            logger = SimulationLogger("sim123", log_dir)
            
            assert logger.simulation_id == "sim123"
            assert logger.log_dir == log_dir
            assert log_dir.exists()
    
    def test_get_agent_logger(self):
        """Test getting agent loggers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            sim_logger = SimulationLogger("sim123", log_dir)
            
            agent_logger1 = sim_logger.get_agent_logger("Agent1")
            agent_logger2 = sim_logger.get_agent_logger("Agent2")
            agent_logger1_again = sim_logger.get_agent_logger("Agent1")
            
            assert agent_logger1.agent_name == "Agent1"
            assert agent_logger2.agent_name == "Agent2"
            assert agent_logger1 is agent_logger1_again  # Same instance
    
    def test_log_message(self):
        """Test logging messages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            sim_logger = SimulationLogger("sim123", log_dir)
            
            sim_logger.log_message("Agent1", "Hello world", {"round": 1})
            
            assert len(sim_logger.messages) == 1
            message = sim_logger.messages[0]
            assert message['agent'] == "Agent1"
            assert message['message'] == "Hello world"
            assert message['metadata'] == {"round": 1}
    
    def test_increment_round(self):
        """Test round increment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            sim_logger = SimulationLogger("sim123", log_dir)
            
            assert sim_logger._round_number == 0
            sim_logger.increment_round()
            assert sim_logger._round_number == 1
            sim_logger.increment_round()
            assert sim_logger._round_number == 2
    
    def test_log_utility_update(self):
        """Test logging utility updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            sim_logger = SimulationLogger("sim123", log_dir)
            
            sim_logger.log_utility_update("Agent1", 0.75, {"final_price": 100})
            
            agent_logger = sim_logger.get_agent_logger("Agent1")
            assert len(agent_logger.utility_history) == 1
            assert agent_logger.utility_history[0].utility_value == 0.75
    
    def test_save_logs(self):
        """Test saving logs to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            sim_logger = SimulationLogger("sim123", log_dir)
            
            # Add some data
            sim_logger.log_message("Agent1", "Hello")
            sim_logger.log_utility_update("Agent1", 0.75)
            sim_logger.metrics.record("price", 100)
            
            # Save logs
            sim_logger.save_logs()
            
            # Check files exist
            assert (log_dir / "messages.json").exists()
            assert (log_dir / "agent_Agent1.json").exists()
            assert (log_dir / "metrics.json").exists()
            
            # Check content
            with open(log_dir / "messages.json") as f:
                messages = json.load(f)
                assert len(messages) == 1
                assert messages[0]["agent"] == "Agent1"
    
    def test_get_summary(self):
        """Test getting simulation summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            sim_logger = SimulationLogger("sim123", log_dir)
            
            sim_logger.log_message("Agent1", "Hello")
            sim_logger.increment_round()
            sim_logger.log_utility_update("Agent1", 0.75)
            
            summary = sim_logger.get_summary()
            
            assert summary["simulation_id"] == "sim123"
            assert summary["total_rounds"] == 1
            assert summary["total_messages"] == 1
            assert "Agent1" in summary["agents"]


class TestSimulationReporter:
    """Test SimulationReporter class."""
    
    def setup_test_logs(self, log_dir):
        """Create test log files."""
        # Create messages.json
        messages = [
            {"timestamp": "2023-01-01T10:00:00", "round": 1, "agent": "Agent1", "message": "Hello", "metadata": {}},
            {"timestamp": "2023-01-01T10:01:00", "round": 1, "agent": "Agent2", "message": "Hi there", "metadata": {}},
        ]
        with open(log_dir / "messages.json", 'w') as f:
            json.dump(messages, f)
        
        # Create metrics.json
        metrics = {
            "final_price": {"count": 1, "mean": 100.0, "min": 100, "max": 100, "last": 100},
            "deal_reached": {"count": 1, "last": True}
        }
        with open(log_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)
        
        # Create agent files
        agent_data = {
            "agent_name": "Agent1",
            "actions": [
                {
                    "timestamp": "2023-01-01T10:00:00",
                    "agent_name": "Agent1",
                    "action_type": "message",
                    "content": "Hello",
                    "metadata": {}
                }
            ],
            "utility_history": [
                {
                    "timestamp": "2023-01-01T10:00:00",
                    "round_number": 1,
                    "agent_name": "Agent1",
                    "utility_value": 0.75,
                    "environment_state": {}
                }
            ]
        }
        with open(log_dir / "agent_Agent1.json", 'w') as f:
            json.dump(agent_data, f)
    
    def test_reporter_initialization(self):
        """Test reporter initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            self.setup_test_logs(log_dir)
            
            reporter = SimulationReporter(log_dir)
            
            assert len(reporter.messages) == 2
            assert len(reporter.agent_data) == 1
            assert "Agent1" in reporter.agent_data
    
    def test_generate_summary(self):
        """Test generating simulation summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            self.setup_test_logs(log_dir)
            
            reporter = SimulationReporter(log_dir)
            summary = reporter.generate_summary()
            
            assert summary["total_messages"] == 2
            assert summary["total_rounds"] == 1
            assert "Agent1" in summary["agents"]
            assert "Agent1" in summary["agent_summaries"]
            
            agent_summary = summary["agent_summaries"]["Agent1"]
            assert agent_summary["total_actions"] == 1
            assert agent_summary["final_utility"] == 0.75
    
    def test_generate_markdown_report(self):
        """Test generating markdown report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            self.setup_test_logs(log_dir)
            
            reporter = SimulationReporter(log_dir)
            markdown = reporter.generate_markdown_report()
            
            assert "# Simulation Report" in markdown
            assert "Agent1" in markdown
            assert "0.7500" in markdown  # Final utility


@pytest.mark.skipif(
    not all(module in sys.modules for module in ['matplotlib', 'seaborn', 'pandas']),
    reason="Visualization dependencies not available"
)
class TestSimulationVisualizer:
    """Test SimulationVisualizer class."""
    
    def setup_test_logs(self, log_dir):
        """Create test log files for visualization."""
        # Create messages.json
        messages = [
            {"timestamp": "2023-01-01T10:00:00", "round": 1, "agent": "Agent1", "message": "Hello", "metadata": {}},
            {"timestamp": "2023-01-01T10:01:00", "round": 1, "agent": "Agent2", "message": "Hi there", "metadata": {}},
        ]
        with open(log_dir / "messages.json", 'w') as f:
            json.dump(messages, f)
        
        # Create metrics.json
        metrics = {
            "Agent1_utility": {"count": 2, "mean": 0.6, "min": 0.5, "max": 0.7, "last": 0.7},
            "Agent2_utility": {"count": 2, "mean": 0.4, "min": 0.3, "max": 0.5, "last": 0.5}
        }
        with open(log_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)
        
        # Create agent files
        for agent_name, utilities in [("Agent1", [0.5, 0.7]), ("Agent2", [0.3, 0.5])]:
            agent_data = {
                "agent_name": agent_name,
                "actions": [
                    {
                        "timestamp": "2023-01-01T10:00:00",
                        "agent_name": agent_name,
                        "action_type": "message",
                        "content": "Hello",
                        "metadata": {}
                    }
                ],
                "utility_history": [
                    {
                        "timestamp": "2023-01-01T10:00:00",
                        "round_number": i+1,
                        "agent_name": agent_name,
                        "utility_value": util,
                        "environment_state": {}
                    } for i, util in enumerate(utilities)
                ]
            }
            with open(log_dir / f"agent_{agent_name}.json", 'w') as f:
                json.dump(agent_data, f)
    
    @patch('matplotlib.pyplot.show')
    def test_visualizer_initialization(self, mock_show):
        """Test visualizer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            self.setup_test_logs(log_dir)
            
            visualizer = SimulationVisualizer(log_dir)
            assert visualizer.log_dir == log_dir
    
    @patch('matplotlib.pyplot.show')
    def test_plot_utility_trends(self, mock_show):
        """Test plotting utility trends."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            self.setup_test_logs(log_dir)
            
            visualizer = SimulationVisualizer(log_dir)
            fig = visualizer.plot_utility_trends()
            
            assert fig is not None
            assert len(fig.axes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
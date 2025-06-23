#!/usr/bin/env python
"""
Test script for the logging framework - demonstrates functionality without requiring OpenAI API.
"""
import json
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock
import sys

# Add backend to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src" / "backend"))

from logging_framework.core import SimulationLogger, AgentLogger, MetricsCollector
from logging_framework.visualization import SimulationVisualizer
from logging_framework.reporters import HTMLReporter


def create_mock_simulation_data(log_dir: Path):
    """Create mock simulation data for testing."""
    print("Creating mock simulation data...")
    
    # Initialize logger
    sim_logger = SimulationLogger("test_simulation_123", log_dir)
    
    # Simulate agents
    agents = ["BuyerAlice", "SellerBob"]
    
    # Simulate conversation
    conversation = [
        ("BuyerAlice", "Hi! I'm interested in your bike. What are you asking for it?"),
        ("SellerBob", "Hello! I'm asking $900 for it. It's a great bike in excellent condition."),
        ("BuyerAlice", "That seems a bit high. The bike looks good, but I was thinking more like $700."),
        ("SellerBob", "I can't go that low. This bike is worth more than that. How about $850?"),
        ("BuyerAlice", "I could do $800. That's really the most I can afford."),
        ("SellerBob", "Alright, $800 works for me. It's a deal!"),
    ]
    
    # Log conversation with utility evolution
    initial_utilities = {"BuyerAlice": 0.0, "SellerBob": 0.0}
    utility_progression = {
        "BuyerAlice": [0.0, 0.1, 0.3, 0.5, 0.7, 0.75],  # Increasing as price drops
        "SellerBob": [0.0, 0.2, 0.4, 0.6, 0.8, 0.85]     # Negotiation progress
    }
    
    for round_num, (speaker, message) in enumerate(conversation, 1):
        sim_logger.increment_round()
        sim_logger.log_message(speaker, message, {"round": round_num})
        
        # Log agent actions
        for agent_name in agents:
            agent_logger = sim_logger.get_agent_logger(agent_name)
            
            if agent_name == speaker:
                agent_logger.log_action("message", message, {"round": round_num})
                agent_logger.log_action("strategy_decision", 
                                      f"Decided to {'speak' if agent_name == speaker else 'listen'}",
                                      {"active": True})
            else:
                agent_logger.log_action("listening", "Processing opponent's message", {"round": round_num})
            
            # Update utilities
            utility = utility_progression[agent_name][round_num - 1]
            agent_logger.log_utility(round_num, utility, {"current_offer": "negotiating"})
    
    # Final negotiation outcome
    final_environment = {
        "outputs": {
            "final_price": 800,
            "deal_reached": True,
            "buyer_satisfaction": 0.75,
            "negotiation_rounds": 6
        }
    }
    
    # Log final utilities based on outcome
    buyer_final_utility = 1.0 - (800 / 1200)  # BuyerAgent formula: 1 - (price/max_price)
    seller_final_utility = min(800 / 850, 1.0)  # SellerAgent formula: min(price/target, 1.0)
    
    sim_logger.log_utility_update("BuyerAlice", buyer_final_utility, final_environment)
    sim_logger.log_utility_update("SellerBob", seller_final_utility, final_environment)
    
    # Log some additional metrics
    sim_logger.metrics.record("final_price", 800)
    sim_logger.metrics.record("deal_reached", True)
    sim_logger.metrics.record("negotiation_duration", 6)
    sim_logger.metrics.record("buyer_satisfaction", 0.75)
    sim_logger.metrics.record("seller_satisfaction", 0.85)
    
    # Add some final actions
    for agent_name in agents:
        agent_logger = sim_logger.get_agent_logger(agent_name)
        agent_logger.log_action("final_evaluation", 
                               f"Final utility: {buyer_final_utility if agent_name == 'BuyerAlice' else seller_final_utility:.3f}",
                               {"deal_reached": True, "final_price": 800})
    
    # Save all logs
    sim_logger.save_logs()
    
    # Create simulation info
    simulation_info = {
        "simulation_id": "test_simulation_123",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "agents": [
                {"name": "BuyerAlice", "utility_class": "BuyerAgent", "strategy": {"max_price": 1200}},
                {"name": "SellerBob", "utility_class": "SellerAgent", "strategy": {"target_price": 850}}
            ],
            "model": "test-model"
        },
        "max_messages": 10,
        "min_messages": 5
    }
    
    with open(log_dir / "simulation_info.json", 'w') as f:
        json.dump(simulation_info, f, indent=2)
    
    print(f"Mock data created in: {log_dir}")
    return sim_logger


def test_visualizations(log_dir: Path):
    """Test visualization generation."""
    print("\nTesting visualizations...")
    
    try:
        visualizer = SimulationVisualizer(log_dir)
        viz_dir = visualizer.save_all_visualizations()
        print(f"Visualizations saved to: {viz_dir}")
        
        # List generated files
        viz_files = list(viz_dir.glob("*.png"))
        print(f"Generated {len(viz_files)} visualization files:")
        for file in viz_files:
            print(f"  - {file.name}")
        
        return True
    except ImportError as e:
        print(f"Visualization dependencies not available: {e}")
        print("Install with: pip install matplotlib seaborn pandas")
        return False
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return False


def test_reports(log_dir: Path):
    """Test report generation."""
    print("\nTesting report generation...")
    
    try:
        # Generate HTML report
        html_reporter = HTMLReporter(log_dir)
        html_path = html_reporter.generate_report()
        print(f"HTML report generated: {html_path}")
        
        # Check if file exists and has content
        if html_path.exists():
            size = html_path.stat().st_size
            print(f"Report size: {size} bytes")
            return True
        else:
            print("ERROR: HTML report file was not created")
            return False
            
    except Exception as e:
        print(f"Error generating reports: {e}")
        return False


def test_data_integrity(log_dir: Path):
    """Test that logged data is complete and valid."""
    print("\nTesting data integrity...")
    
    required_files = [
        "simulation.log",
        "messages.json", 
        "agent_BuyerAlice.json",
        "agent_SellerBob.json",
        "metrics.json",
        "simulation_info.json"
    ]
    
    missing_files = []
    for filename in required_files:
        if not (log_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"ERROR: Missing files: {missing_files}")
        return False
    
    # Test JSON file validity
    try:
        # Test messages
        with open(log_dir / "messages.json") as f:
            messages = json.load(f)
            assert len(messages) == 6, f"Expected 6 messages, got {len(messages)}"
            print(f"✓ Messages file valid: {len(messages)} messages")
        
        # Test agent data
        for agent in ["BuyerAlice", "SellerBob"]:
            with open(log_dir / f"agent_{agent}.json") as f:
                agent_data = json.load(f)
                assert "actions" in agent_data
                assert "utility_history" in agent_data
                assert len(agent_data["utility_history"]) >= 6
                print(f"✓ Agent {agent} data valid: {len(agent_data['actions'])} actions, {len(agent_data['utility_history'])} utility points")
        
        # Test metrics
        with open(log_dir / "metrics.json") as f:
            metrics = json.load(f)
            assert "final_price" in metrics
            print(f"✓ Metrics file valid: {len(metrics)} metric types")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Data integrity check failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Testing Rich Logging Framework")
    print("=" * 60)
    
    # Create test directory
    test_dir = Path("test_logs") / f"framework_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Test 1: Create mock data
        sim_logger = create_mock_simulation_data(test_dir)
        
        # Test 2: Data integrity
        data_ok = test_data_integrity(test_dir)
        
        # Test 3: Visualizations
        viz_ok = test_visualizations(test_dir)
        
        # Test 4: Reports  
        report_ok = test_reports(test_dir)
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Data Creation:     {'✓ PASS' if sim_logger else '✗ FAIL'}")
        print(f"Data Integrity:    {'✓ PASS' if data_ok else '✗ FAIL'}")
        print(f"Visualizations:    {'✓ PASS' if viz_ok else '✗ FAIL'}")
        print(f"Reports:           {'✓ PASS' if report_ok else '✗ FAIL'}")
        print(f"\nTest artifacts saved to: {test_dir}")
        
        if all([data_ok, report_ok]):
            print("\n🎉 Core logging framework is working correctly!")
            if viz_ok:
                print("🎨 Visualizations are also working perfectly!")
            else:
                print("⚠️  Visualizations need dependency installation")
        else:
            print("\n❌ Some tests failed - check error messages above")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
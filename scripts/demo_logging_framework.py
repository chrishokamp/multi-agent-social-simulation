#!/usr/bin/env python
"""
Demo script showing the rich logging framework in action.
This creates realistic simulation logs without requiring API calls.
"""
import asyncio
from pathlib import Path
import sys

# Add backend to path
BASE_DIR = Path(__file__).resolve().parents[1] 
sys.path.append(str(BASE_DIR / "src" / "backend"))

def main():
    """Run the logging framework demo."""
    print("🎬 Rich Logging Framework Demo")
    print("=" * 50)
    print()
    print("This demo shows what 'make run-simulation' generates:")
    print("• Agent action logs and utility tracking")
    print("• Beautiful visualizations and charts") 
    print("• HTML/PDF reports with analysis")
    print("• Consolidated multi-run reports")
    print()
    print("🚀 Running test without API calls...")
    print()
    
    # Add scripts to path and run the test framework
    sys.path.append(str(BASE_DIR))
    from scripts.test_logging_framework import main as test_main
    test_main()

if __name__ == "__main__":
    main()
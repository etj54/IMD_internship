#!/usr/bin/env python3
"""
Test script for the real-time RVR prediction system
This script runs one update cycle to verify everything works correctly
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_rvr_system import RealTimeRVRSystem

def test_single_update():
    """Test a single update cycle of the real-time system"""
    print("=" * 60)
    print("🧪 TESTING REAL-TIME RVR PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize the system with shorter interval for testing
    system = RealTimeRVRSystem(update_interval=30)
    
    # Show initial status
    print(f"\n📊 Initial System Status:")
    status = system.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Perform one update cycle
    print(f"\n🔄 Running single update cycle...")
    system.update_system()
    
    # Show final status
    print(f"\n📊 Final System Status:")
    status = system.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Check if output files were created
    output_dir = Path("real_time_predictions")
    if output_dir.exists():
        print(f"\n📁 Output Directory Contents:")
        for file in output_dir.glob("*.csv"):
            print(f"   📄 {file.name} ({file.stat().st_size} bytes)")
    
    print(f"\n✅ Test completed successfully!")
    print(f"   Check the 'real_time_predictions' directory for generated CSV files")
    print(f"   Run 'python scripts/generate_rvr_map.py' to create the map with real-time data")

if __name__ == "__main__":
    test_single_update() 
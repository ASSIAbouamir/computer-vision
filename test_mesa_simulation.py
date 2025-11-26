#!/usr/bin/env python
"""Test MESA simulation in isolation"""

from simulation.model import CrowdModel
import sys

print("🧪 Testing MESA Simulation...")

try:
    # Create a simple model
    print("  Creating model with 10 agents...")
    model = CrowdModel(N=10, width=100, height=100, scenario='normal')
    
    # Run a few steps
    print("  Running 5 simulation steps...")
    for i in range(5):
        model.step()
        density = model.datacollector.model_vars['Density'][-1]
        avg_speed = model.datacollector.model_vars['AvgSpeed'][-1]
        print(f"    Step {i+1}: Density={density:.4f}, AvgSpeed={avg_speed:.4f}")
    
    print("\n✅ MESA simulation test PASSED!")
    print(f"   Final agents: {len(list(model.agents))}")
    print(f"   Total steps: {model.steps_count}")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ MESA simulation test FAILED!")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

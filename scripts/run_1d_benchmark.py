"""
Execution entry point for the 1D Poisson quantum linear solver benchmark.

This script imports and triggers the primary benchmark orchestrator, ensuring 
systematic execution of all simulated sweeps, CSV data generation, and 
Matplotlib visualisations without exposing internal module architecture.
"""
import sys
from pathlib import Path

# Dynamically resolve the project root directory (one level up from this script)
# and append it to the system path to enable absolute imports.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from benchmark.runner import main

if __name__ == "__main__":
    main()
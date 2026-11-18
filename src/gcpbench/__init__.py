"""gcpbench: leakage-free, audited benchmark for LLM-generated graph-coloring heuristics.

Revision infrastructure for "LLM-Aided Heuristics for the Graph Coloring
Problem" (AlMubaraki, Ahmad, Salman; Computation, MDPI). Each module names
the revision-plan items it implements (e.g. S0.1, C1, E1).

The original implementation in ``src/heuristics/sequential_heuristics.py``
is kept unchanged as the *reference* the new code is checked against.
"""

FEATURES = ("sat", "deg", "ucn", "cc", "twohop")
DYNAMIC_FEATURES = ("sat", "ucn")
STATIC_FEATURES = ("deg", "cc", "twohop")

__version__ = "2.0.0"

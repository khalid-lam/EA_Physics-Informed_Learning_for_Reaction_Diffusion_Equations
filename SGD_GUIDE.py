#!/usr/bin/env python
"""
MINI-BATCH SGD IMPLEMENTATION REFERENCE
========================================

Simplified SGD Support for Physics-Informed Neural Networks
"""

# ============================================================================
# QUICK START
# ============================================================================

# Full-batch training (default, unchanged from original):
#   history = train(model, equation, x_in, steps=2000, batch_size=None)
#   → Does 2000 gradient updates using all interior points each time

# Mini-batch SGD training (NEW):
#   history = train(model, equation, x_in, steps=5, batch_size=64)
#   → Does 5 epochs, each epoch shuffles interior points into batches of 64
#   → Total gradient updates = 5 * ceil(n_interior / 64)

# ============================================================================
# API SUMMARY
# ============================================================================

print("""
PARAMETER SEMANTICS
-------------------

batch_size = None (default):
  • Full-batch training
  • steps parameter = number of gradient updates
  • Behavior identical to original implementation
  • Example: steps=2000, batch_size=None → 2000 gradient steps

batch_size = 64 (or any int < n_interior):
  • Mini-batch SGD
  • steps parameter = number of EPOCHS (not steps!)
  • Each epoch shuffles interior points into batches of 64
  • Boundary and supervised data always full-batch
  • Example: steps=5, batch_size=64 → 5 epochs through dataset

KEY DESIGN DECISIONS
--------------------
✓ Interior points ONLY: Mini-batch sampling applied only to PDE residual terms
✓ Boundary/data FULL-BATCH: BC and supervised terms use all points (more stable)
✓ Semantic clarity: Parameter names match their meaning in each mode
✓ Backward compatible: Existing code with batch_size=None works unchanged
✓ Shuffle: Interior points shuffled each epoch for better convergence

USAGE EXAMPLES
--------------
""")

# Example 1: Command-line usage
print("1. CLI with mini-batch:")
print("   python scripts/run_experiment.py --batch-size 64 --steps 10")
print("   → Trains for 10 epochs, batch size 64 on interior points")
print()

# Example 2: In code with ExperimentConfig
print("2. In code with ExperimentConfig:")
print("   config = ExperimentConfig(")
print("       name='Fisher MLP+minibatch',")
print("       batch_size=64,")
print("       steps=20,  # 20 epochs")
print("   )")
print()

# Example 3: Direct train() call
print("3. Direct train() call:")
print("   history = train(")
print("       model, equation, x_interior,")
print("       batch_size=64,")
print("       steps=5,  # 5 epochs")
print("   )")
print()

print("\n" + "="*70)
print("For full details, see:")
print("  • src/solvers.py: train() docstring")
print("  • src/experiment_config.py: ExperimentConfig class")
print("  • test_sgd.py: Validation examples")
print("="*70)

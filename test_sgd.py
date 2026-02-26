#!/usr/bin/env python
"""Quick validation that the simplified SGD API works correctly."""

import sys
from pathlib import Path

# Add src to path
SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

import torch
from equations import PoissonEquation
from models import MLPModel
from sampling import sample_uniform_interior, sample_uniform_boundary
from solvers import train

torch.manual_seed(0)
x_in = sample_uniform_interior(1000, 2)
x_b = sample_uniform_boundary(200, 2)
u_b = torch.zeros(200, 1)

model = MLPModel(input_dim=2, hidden_dim=32, num_hidden_layers=2)
eq = PoissonEquation(f=lambda x: -4.0 * torch.ones(x.shape[0], 1))

# Test 1: Full-batch (batch_size=None)
print('Testing full-batch (batch_size=None):')
h1 = train(model, eq, x_in, x_boundary=x_b, u_boundary=u_b, steps=10, batch_size=None, print_every=0)
print(f'  History length: {len(h1["loss_total"])} (expected: 10)')
assert len(h1["loss_total"]) == 10, "Full-batch: steps should equal history length"

# Test 2: Mini-batch (batch_size=64, steps as epochs)
print('\nTesting mini-batch (batch_size=64, steps=5 epochs):')
model2 = MLPModel(input_dim=2, hidden_dim=32, num_hidden_layers=2)
h2 = train(model2, eq, x_in, x_boundary=x_b, u_boundary=u_b, steps=5, batch_size=64, print_every=0)
expected = 5 * ((1000 + 64 - 1) // 64)  # 5 epochs * 16 batches per epoch
print(f'  History length: {len(h2["loss_total"])} (expected: {expected})')
assert len(h2["loss_total"]) == expected, f"Mini-batch: expected {expected} steps, got {len(h2['loss_total'])}"

print('\n✅ All tests passed!')
print('\nAPI Summary:')
print('  • batch_size=None:      steps = number of gradient updates')
print('  • batch_size=64:        steps = number of epochs through dataset')
print('  • Boundary/data terms:  always full-batch (not affected by batch_size)')

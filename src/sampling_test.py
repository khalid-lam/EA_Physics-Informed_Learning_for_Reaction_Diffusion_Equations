# src/sampling_test.py
import torch
from sampling import sample_uniform_interior, sample_uniform_boundary, make_dirichlet_bc


def run_reference_checks():
    torch.manual_seed(0)

    n, d = 1000, 3

    x_in = sample_uniform_interior(n, d, dtype=torch.float32)
    assert x_in.shape == (n, d)
    assert torch.all(x_in >= 0.0)
    assert torch.all(x_in <= 1.0)

    x_b = sample_uniform_boundary(n, d, dtype=torch.float32)
    assert x_b.shape == (n, d)
    assert torch.all(x_b >= 0.0)
    assert torch.all(x_b <= 1.0)

    # For each boundary point, at least one coordinate must be exactly 0 or 1
    on_boundary = ((x_b == 0.0) | (x_b == 1.0)).any(dim=1)
    frac = on_boundary.float().mean().item()
    print("=== Boundary sampling check ===")
    print(f"Fraction on boundary: {frac:.4f} (expected 1.0)")
    assert frac == 1.0

    # Test boundary condition function
    # Example: u(x) = x0 + 2*x1 - x2
    def bc_fn(x):
        return x[:, 0:1] + 2.0 * x[:, 1:2] - x[:, 2:3]

    u_b = make_dirichlet_bc(x_b, bc_fn)
    assert u_b.shape == (n, 1)

    # Spot-check a few indices
    idx = torch.tensor([0, 1, 2, 10, 999])
    u_ref = bc_fn(x_b[idx])
    max_err = (u_b[idx] - u_ref).abs().max().item()
    print("=== Dirichlet BC function check ===")
    print(f"max|u_b - ref| = {max_err:.6e} (expected 0)")
    assert max_err == 0.0

    print("\nAll sampling checks passed ✅")


if __name__ == "__main__":
    run_reference_checks()

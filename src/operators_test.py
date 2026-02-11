# src/operators_test.py
import torch
from operators import gradient, laplacian


def run_reference_checks():
    """
    Reference checks using analytic functions.
    """
    torch.manual_seed(0)

    # Use fixed points for deterministic checks
    x = torch.tensor(
        [[0.0, 0.0],
         [0.25, 0.5],
         [0.5, 0.25],
         [1.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    X = x[:, 0:1]
    Y = x[:, 1:2]

    # u(x,y) = x^2 + y^2
    u = X**2 + Y**2  # (N,1)

    # Expected gradient: (2x, 2y)
    grad_ref = torch.cat([2 * X, 2 * Y], dim=1)  # (N,2)
    grad_u = gradient(u, x)
    print("=== Gradient reference check ===")
    print(f"max|grad - ref| = {(grad_u - grad_ref).abs().max().item():.6e}")
    assert grad_u.shape == grad_ref.shape
    assert torch.allclose(grad_u, grad_ref, atol=1e-6, rtol=0.0)

    # Expected Laplacian: 4
    lap_ref = torch.full((x.shape[0], 1), 4.0, dtype=x.dtype)
    lap_u = laplacian(u, x)
    print("=== Laplacian reference check ===")
    print(f"max|lap - ref| = {(lap_u - lap_ref).abs().max().item():.6e}")
    assert lap_u.shape == lap_ref.shape
    assert torch.allclose(lap_u, lap_ref, atol=1e-6, rtol=0.0)

    # Extra sanity: a constant function has zero grad and zero laplacian
    u_const = torch.ones_like(u) * 3.0
    grad_c = gradient(u_const, x)
    lap_c = laplacian(u_const, x)
    print("=== Constant function check ===")
    print(f"max|grad| = {grad_c.abs().max().item():.6e} (expected 0)")
    print(f"max|lap|  = {lap_c.abs().max().item():.6e} (expected 0)")
    assert torch.allclose(grad_c, torch.zeros_like(grad_c), atol=1e-6, rtol=0.0)
    assert torch.allclose(lap_c, torch.zeros_like(lap_c), atol=1e-6, rtol=0.0)

    print("\nAll operator checks passed ✅")


if __name__ == "__main__":
    run_reference_checks()

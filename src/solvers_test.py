# src/solvers_test.py
import torch

from equations import PoissonEquation
from models import MLPModel
from sampling import sample_uniform_interior, sample_uniform_boundary, make_dirichlet_bc
from solvers import train


def run_reference_checks():
    """
    End-to-end sanity check:
    Train a small MLP PINN on Poisson with an analytic solution and Dirichlet BC,
    then verify that the grid MSE improves after training.

    Domain: [0,1]^2
    Exact solution: u(x,y) = x^2 + y^2
    Then Δu = 4, so for -Δu = f we use f = -4.
    """
    torch.manual_seed(0)
    dtype = torch.float32

    def u_exact(x: torch.Tensor) -> torch.Tensor:
        X = x[:, 0:1]
        Y = x[:, 1:2]
        return X**2 + Y**2

    # Poisson: -Δu = f, with Δu=4 => f=-4
    f = lambda x: -4.0 * torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    eq = PoissonEquation(f=f)

    # Training points
    x_in = sample_uniform_interior(2000, 2, dtype=dtype)
    x_b = sample_uniform_boundary(400, 2, dtype=dtype)
    u_b = make_dirichlet_bc(x_b, u_exact)

    # Model
    model = MLPModel(input_dim=2, hidden_dim=32, num_hidden_layers=3, activation="tanh")

    # Evaluation grid
    xs = torch.linspace(0.0, 1.0, 25, dtype=dtype)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).reshape(-1, 2)

    # Pre-train MSE
    with torch.no_grad():
        mse0 = torch.mean((model(grid) - u_exact(grid)) ** 2).item()

    print("=== Pre-train check ===")
    print(f"Grid MSE: {mse0:.6e}")

    # Train
    _ = train(
        model,
        eq,
        x_in,
        x_boundary=x_b,
        u_boundary=u_b,
        use_energy=False,
        w_pde=1.0,
        w_bc=10.0,
        lr=1e-3,
        steps=1500,
        print_every=300,
    )

    # Post-train MSE
    with torch.no_grad():
        mse1 = torch.mean((model(grid) - u_exact(grid)) ** 2).item()

    print("=== Post-train check ===")
    print(f"Grid MSE: {mse1:.6e}")

    # Basic requirement: should improve
    assert mse1 < mse0

    print("\nSolver training check passed ✅")


if __name__ == "__main__":
    run_reference_checks()

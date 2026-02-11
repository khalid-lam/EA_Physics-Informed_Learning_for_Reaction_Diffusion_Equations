# src/losses_test.py
import torch

from equations import PoissonEquation, FisherKPPStationaryEquation
from losses import pde_residual_loss, dirichlet_bc_loss


class AnalyticModel(torch.nn.Module):
    """
    A tiny analytic "model" that returns a known u(x).
    Useful to validate operators + equation + loss wiring.
    """
    def __init__(self, kind: str = "quad"):
        super().__init__()
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = x[:, 0:1]
        Y = x[:, 1:2]
        if self.kind == "quad":
            return X**2 + Y**2
        if self.kind == "zero":
            return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if self.kind == "one":
            return torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        raise ValueError("Unknown kind")


def run_reference_checks():
    torch.manual_seed(0)

    # Interior points in 2D
    x_in = torch.rand(64, 2)

    # --- Poisson residual should be zero for u=x^2+y^2 and f=-4 ---
    f = lambda x: -4.0 * torch.ones(x.shape[0], 1)
    poisson = PoissonEquation(f=f)
    model_quad = AnalyticModel(kind="quad")

    lp = pde_residual_loss(model_quad, poisson, x_in)
    print("=== Poisson residual loss reference ===")
    print(f"loss = {lp.item():.6e} (expected ~0)")
    assert lp.item() < 1e-10

    # --- Fisher residual should be zero for constant solutions u=0 and u=1 (with lap=0) ---
    fisher = FisherKPPStationaryEquation(D=2.0, r=3.0)

    model_zero = AnalyticModel(kind="zero")
    lf0 = pde_residual_loss(model_zero, fisher, x_in)
    print("=== Fisher residual loss reference (u=0) ===")
    print(f"loss = {lf0.item():.6e} (expected ~0)")
    assert lf0.item() < 1e-10

    model_one = AnalyticModel(kind="one")
    lf1 = pde_residual_loss(model_one, fisher, x_in)
    print("=== Fisher residual loss reference (u=1) ===")
    print(f"loss = {lf1.item():.6e} (expected ~0)")
    assert lf1.item() < 1e-10

    # --- Dirichlet BC loss reference ---
    x_b = torch.rand(32, 2)
    u_b = model_quad(x_b).detach()
    lbc = dirichlet_bc_loss(model_quad, x_b, u_b)
    print("=== Dirichlet BC loss reference ===")
    print(f"loss = {lbc.item():.6e} (expected 0)")
    assert lbc.item() < 1e-12

    print("\nAll loss checks passed ✅")


if __name__ == "__main__":
    run_reference_checks()

# tests/test_equations.py
import torch
from equations import PoissonEquation, FisherKPPStationaryEquation


def test_poisson_residual_shapes_and_values():
    torch.manual_seed(0)
    N, d = 10, 2
    x = torch.rand(N, d)
    u = torch.zeros(N, 1)
    lap_u = torch.zeros(N, 1)

    # f(x) = 1 everywhere
    f = lambda x: torch.ones(x.shape[0], 1)
    eq = PoissonEquation(f=f)

    r = eq.residual(x, u, lap_u)
    assert r.shape == (N, 1)
    assert torch.allclose(r, -torch.ones(N, 1))


def test_poisson_energy_density_shape():
    torch.manual_seed(0)
    N, d = 10, 2
    x = torch.rand(N, d)
    u = torch.zeros(N, 1)
    grad_u = torch.zeros(N, d)

    f = lambda x: torch.ones(x.shape[0], 1)
    eq = PoissonEquation(f=f)

    e = eq.energy_density(x, u, grad_u)
    assert e.shape == (N, 1)
    assert torch.allclose(e, torch.zeros(N, 1))


def test_fisher_residual_shapes_and_values():
    torch.manual_seed(0)
    N, d = 10, 2
    x = torch.rand(N, d)
    u = torch.zeros(N, 1)
    lap_u = torch.zeros(N, 1)

    eq = FisherKPPStationaryEquation(D=2.0, r=3.0)
    r = eq.residual(x, u, lap_u)

    assert r.shape == (N, 1)
    assert torch.allclose(r, torch.zeros(N, 1))


def run_basic_checks():
    """
    Basic printable sanity checks.
    This is useful for quick debugging before integrating more modules.
    """
    torch.manual_seed(0)

    N, d = 5, 2
    x = torch.rand(N, d)
    u = torch.randn(N, 1)
    lap_u = torch.randn(N, 1)
    grad_u = torch.randn(N, d)

    # Poisson
    f = lambda x: torch.ones(x.shape[0], 1)
    poisson = PoissonEquation(f=f)

    r_poisson = poisson.residual(x, u, lap_u)
    e_poisson = poisson.energy_density(x, u, grad_u)

    print("=== Poisson basic checks ===")
    print(f"x.shape      = {tuple(x.shape)}")
    print(f"u.shape      = {tuple(u.shape)}")
    print(f"lap_u.shape  = {tuple(lap_u.shape)}")
    print(f"grad_u.shape = {tuple(grad_u.shape)}")
    print(f"residual.shape = {tuple(r_poisson.shape)}")
    print(f"residual stats: mean={r_poisson.mean().item():.4f}, "
          f"min={r_poisson.min().item():.4f}, max={r_poisson.max().item():.4f}")
    print(f"energy.shape   = {tuple(e_poisson.shape)}")
    print(f"energy stats:   mean={e_poisson.mean().item():.4f}, "
          f"min={e_poisson.min().item():.4f}, max={e_poisson.max().item():.4f}")
    print()

    # Fisher-KPP
    fisher = FisherKPPStationaryEquation(D=2.0, r=3.0)
    r_fisher = fisher.residual(x, u, lap_u)

    print("=== Fisher–KPP (stationary) basic checks ===")
    print(f"residual.shape = {tuple(r_fisher.shape)}")
    print(f"residual stats: mean={r_fisher.mean().item():.4f}, "
          f"min={r_fisher.min().item():.4f}, max={r_fisher.max().item():.4f}")
    print()

    # Deterministic zero case checks (should be exact)
    u0 = torch.zeros(N, 1)
    lap0 = torch.zeros(N, 1)
    r_fisher0 = fisher.residual(x, u0, lap0)
    print("=== Deterministic checks ===")
    print(f"Fisher residual with u=0 and lap_u=0 -> max abs = {r_fisher0.abs().max().item():.6e}")
    assert torch.allclose(r_fisher0, torch.zeros_like(r_fisher0))

def run_reference_checks():
    """
    Reference-based checks using analytic u, grad(u), lap(u).
    This validates the PDE residual and Poisson energy density exactly.
    """
    torch.manual_seed(0)

    # Use a small deterministic set of points in 2D
    x = torch.tensor(
        [[0.0, 0.0],
         [0.25, 0.5],
         [0.5, 0.25],
         [1.0, 1.0]],
        dtype=torch.float32
    )
    N = x.shape[0]
    X = x[:, 0:1]
    Y = x[:, 1:2]

    # ---- Poisson reference: u = x^2 + y^2 ----
    u = X**2 + Y**2                      # (N,1)
    grad_u = torch.cat([2*X, 2*Y], dim=1) # (N,2)
    lap_u = torch.full((N, 1), 4.0)       # Δu = 4 in 2D

    # For -Δu = f, we need f = -4
    f = lambda x_in: -4.0 * torch.ones(x_in.shape[0], 1)
    poisson = PoissonEquation(f=f)

    r = poisson.residual(x, u, lap_u)
    max_abs_r = r.abs().max().item()

    # Expected residual is exactly 0
    print("=== Reference check: Poisson residual ===")
    print(f"max|residual| = {max_abs_r:.6e} (expected 0)")
    assert torch.allclose(r, torch.zeros_like(r), atol=0.0, rtol=0.0)

    # Energy density expected: e = 6*(x^2 + y^2)
    e = poisson.energy_density(x, u, grad_u)
    e_ref = 6.0 * (X**2 + Y**2)
    max_abs_e = (e - e_ref).abs().max().item()

    print("=== Reference check: Poisson energy density ===")
    print(f"max|e - e_ref| = {max_abs_e:.6e} (expected 0)")
    assert torch.allclose(e, e_ref, atol=0.0, rtol=0.0)

    # ---- Fisher-KPP reference: constant solutions ----
    fisher = FisherKPPStationaryEquation(D=2.0, r=3.0)

    u0 = torch.zeros(N, 1)
    lap0 = torch.zeros(N, 1)
    r0 = fisher.residual(x, u0, lap0)
    print("=== Reference check: Fisher residual (u=0) ===")
    print(f"max|residual| = {r0.abs().max().item():.6e} (expected 0)")
    assert torch.allclose(r0, torch.zeros_like(r0), atol=0.0, rtol=0.0)

    u1 = torch.ones(N, 1)
    r1 = fisher.residual(x, u1, lap0)
    print("=== Reference check: Fisher residual (u=1) ===")
    print(f"max|residual| = {r1.abs().max().item():.6e} (expected 0)")
    assert torch.allclose(r1, torch.zeros_like(r1), atol=0.0, rtol=0.0)

    print("\nAll reference checks passed ✅")


if __name__ == "__main__":
    # Running this file directly prints the sanity checks.
    # Unit tests can still be executed via: python -m tests.test_equations
    run_basic_checks()
    run_reference_checks()

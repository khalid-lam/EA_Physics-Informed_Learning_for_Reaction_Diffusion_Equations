# src/equations_test.py
import torch
from src.equations import PoissonEquation, FisherKPPStationaryEquation


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
    # residual = -0 - 1 = -1
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
    # grad_u=0 and u=0 -> energy density should be 0
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
    # u=0 and lap_u=0 -> residual should be 0
    assert torch.allclose(r, torch.zeros(N, 1))

# src/EA_Physics-Informed_Learning_for_Reaction_Diffusion_Equations/equations.py

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PoissonEquation:
    """
    Poisson equation on a domain Ω (boundary conditions handled elsewhere):

        -Δu(x) = f(x)

    Parameters
    ----------
    f : callable
        Function f(x) returning a tensor of shape (N, 1) or (N,).
    """

    f: object  # kept generic to avoid importing typing
    name: str = "poisson"

    def residual(self, x: torch.Tensor, u: torch.Tensor, lap_u: torch.Tensor) -> torch.Tensor:
        """
        Strong-form residual:
            R(x) = -Δu(x) - f(x)
        """
        fx = self.f(x)
        fx = fx.reshape_as(u)
        return -lap_u - fx

    def energy_density(self, x: torch.Tensor, u: torch.Tensor, grad_u: torch.Tensor) -> torch.Tensor:
        """
        Pointwise energy density for Poisson:
            e(x) = 0.5 * |∇u(x)|^2 - f(x) * u(x)

        Notes
        -----
        The full energy is typically approximated as an average over sampled points.
        """
        fx = self.f(x).reshape_as(u)
        grad_sq = torch.sum(grad_u ** 2, dim=1, keepdim=True)
        return 0.5 * grad_sq - fx * u


@dataclass(frozen=True)
class FisherKPPStationaryEquation:
    """
    Stationary Fisher–KPP reaction–diffusion equation:

        -D Δu(x) - r u(x) (1 - u(x)) = 0

    Notes
    -----
    - Positivity constraints (u >= 0) should be enforced via the loss or the model.
    - Boundary conditions are handled elsewhere.
    """

    D: float
    r: float
    name: str = "fisher_kpp_stationary"

    def residual(self, x: torch.Tensor, u: torch.Tensor, lap_u: torch.Tensor) -> torch.Tensor:
        """
        Strong-form residual:
            R(x) = -D Δu(x) - r u(x) (1 - u(x))
        """
        D = torch.as_tensor(self.D, dtype=u.dtype, device=u.device)
        r = torch.as_tensor(self.r, dtype=u.dtype, device=u.device)
        return -D * lap_u - r * u * (1.0 - u)

    def energy_density(self, x: torch.Tensor, u: torch.Tensor, grad_u: torch.Tensor) -> torch.Tensor:
        """
        Energy density whose Euler–Lagrange equation is:
            -D Δu - r u(1-u) = 0

        e(x) = (D/2)*|∇u|^2 - r*(u^2/2 - u^3/3)
        """
        D = torch.as_tensor(self.D, dtype=u.dtype, device=u.device)
        r = torch.as_tensor(self.r, dtype=u.dtype, device=u.device)

        grad_sq = torch.sum(grad_u ** 2, dim=1, keepdim=True)
        potential = (u ** 2) / 2.0 - (u ** 3) / 3.0
        return 0.5 * D * grad_sq - r * potential

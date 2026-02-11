# src/losses.py
import torch
from operators import gradient, laplacian


def mse(x: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error helper: mean(x^2).
    """
    return torch.mean(x ** 2)


def pde_residual_loss(model, equation, x_interior: torch.Tensor) -> torch.Tensor:
    """
    Strong-form PINN loss on interior points:
        L_pde = mean( R(x)^2 )

    where R(x) is the PDE residual.

    Parameters
    ----------
    model : torch.nn.Module
        Maps x -> u, expects output shape (N,1).
    equation : object
        Must implement residual(x, u, lap_u) -> (N,1).
    x_interior : torch.Tensor
        Interior collocation points, shape (N,d).

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    x = x_interior.clone().detach().requires_grad_(True)
    u = model(x)
    lap_u = laplacian(u, x)
    r = equation.residual(x, u, lap_u)
    return mse(r)


def energy_loss(model, equation, x_interior: torch.Tensor) -> torch.Tensor:
    x = x_interior.clone().detach().requires_grad_(True)
    u = model(x)
    grad_u = gradient(u, x)
    e = equation.energy_density(x, u, grad_u)
    return torch.mean(e)


def dirichlet_bc_loss(model, x_boundary: torch.Tensor, u_boundary: torch.Tensor) -> torch.Tensor:
    """
    Dirichlet boundary condition loss:
        L_bc = mean( (u(x_b) - u_b)^2 )

    Parameters
    ----------
    x_boundary : (Nb, d)
    u_boundary : (Nb, 1) or (Nb,)
    """
    if u_boundary.ndim == 1:
        u_boundary = u_boundary.unsqueeze(1)

    pred = model(x_boundary)
    return mse(pred - u_boundary)


def total_loss(
    model,
    equation,
    x_interior: torch.Tensor,
    x_boundary: torch.Tensor | None = None,
    u_boundary: torch.Tensor | None = None,
    w_pde: float = 1.0,
    w_bc: float = 1.0,
    use_energy: bool = False,
) -> torch.Tensor:
    """
    Total training loss aggregator.

    If use_energy=True, uses poisson_energy_loss instead of pde_residual_loss.

    Notes
    -----
    - For Fisher–KPP, use_energy should remain False (unless you add a variational form later).
    """
    if use_energy:
        lpde = poisson_energy_loss(model, equation, x_interior)
    else:
        lpde = pde_residual_loss(model, equation, x_interior)

    if x_boundary is None or u_boundary is None:
        return w_pde * lpde

    lbc = dirichlet_bc_loss(model, x_boundary, u_boundary)
    return w_pde * lpde + w_bc * lbc

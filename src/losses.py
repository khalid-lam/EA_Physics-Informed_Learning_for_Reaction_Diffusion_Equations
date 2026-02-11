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

def data_loss(model, x_data: torch.Tensor, u_data: torch.Tensor) -> torch.Tensor:
    """
    Data mismatch loss:
        L_data = mean( (u(x_data) - u_data)^2 )
    """
    if u_data.ndim == 1:
        u_data = u_data.unsqueeze(1)
    pred = model(x_data)
    return mse(pred - u_data)

def l2_regularization(model) -> torch.Tensor:
    """
    L2 weight decay regularization: sum(||theta||^2) / n_params
    (normalized so the scale is not crazy).
    """
    s = torch.tensor(0.0)
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            s = s + torch.sum(p ** 2)
            n += p.numel()
    if n == 0:
        return torch.tensor(0.0)
    return s / float(n)

def total_loss(
    model,
    equation,
    x_interior: torch.Tensor | None = None,
    x_boundary: torch.Tensor | None = None,
    u_boundary: torch.Tensor | None = None,
    x_data: torch.Tensor | None = None,
    u_data: torch.Tensor | None = None,
    *,
    use_energy: bool = False,
    w_pde: float = 1.0,
    w_bc: float = 1.0,
    w_data: float = 1.0,
    w_reg: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Total loss with up to 4 weighted terms:
        L = w_pde*L_pde + w_bc*L_bc + w_data*L_data + w_reg*L_reg

    Returns
    -------
    total : scalar tensor
    parts : dict with each term (for logging)
    """
    parts = {}

    total = torch.tensor(0.0)
    # Keep device/dtype consistent if possible
    ref = None
    for t in [x_interior, x_boundary, x_data]:
        if t is not None:
            ref = t
            break
    if ref is not None:
        total = total.to(device=ref.device, dtype=ref.dtype)

    # PDE / energy term
    if x_interior is not None:
        if use_energy:
            lpde = energy_loss(model, equation, x_interior)
        else:
            lpde = pde_residual_loss(model, equation, x_interior)
    else:
        lpde = total * 0.0
    parts["pde"] = lpde
    total = total + w_pde * lpde

    # BC term
    if x_boundary is not None and u_boundary is not None:
        lbc = dirichlet_bc_loss(model, x_boundary, u_boundary)
    else:
        lbc = total * 0.0
    parts["bc"] = lbc
    total = total + w_bc * lbc

    # Data term
    if x_data is not None and u_data is not None:
        ldata = data_loss(model, x_data, u_data)
    else:
        ldata = total * 0.0
    parts["data"] = ldata
    total = total + w_data * ldata

    # Regularization
    if w_reg != 0.0:
        lreg = l2_regularization(model).to(device=total.device, dtype=total.dtype)
    else:
        lreg = total * 0.0
    parts["reg"] = lreg
    total = total + w_reg * lreg

    return total, parts

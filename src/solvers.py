# src/solvers.py
import torch

from losses import pde_residual_loss, energy_loss, dirichlet_bc_loss, total_loss


def train(
    model,
    equation,
    x_interior: torch.Tensor,
    x_boundary: torch.Tensor | None = None,
    u_boundary: torch.Tensor | None = None,
    *,
    use_energy: bool = False,
    w_pde: float = 1.0,
    w_bc: float = 1.0,
    lr: float = 1e-3,
    steps: int = 2000,
    print_every: int = 200,
) -> dict:
    """
    Train a model with Adam.

    Returns a history dict with losses.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "loss_total": [],
        "loss_pde": [],
        "loss_bc": [],
    }

    for step in range(1, steps + 1):
        opt.zero_grad()

        # Compute components for logging
        if use_energy:
            lpde = energy_loss(model, equation, x_interior)
        else:
            lpde = pde_residual_loss(model, equation, x_interior)

        if x_boundary is None or u_boundary is None:
            lbc = torch.tensor(0.0, device=x_interior.device, dtype=x_interior.dtype)
            ltot = w_pde * lpde
        else:
            lbc = dirichlet_bc_loss(model, x_boundary, u_boundary)
            ltot = w_pde * lpde + w_bc * lbc

        ltot.backward()
        opt.step()

        history["loss_total"].append(ltot.detach().item())
        history["loss_pde"].append(lpde.detach().item())
        history["loss_bc"].append(lbc.detach().item())

        if print_every > 0 and (step % print_every == 0 or step == 1 or step == steps):
            print(
                f"[step {step:5d}] "
                f"total={history['loss_total'][-1]:.6e} "
                f"pde={history['loss_pde'][-1]:.6e} "
                f"bc={history['loss_bc'][-1]:.6e}"
            )

    return history


def evaluate_model(model, x: torch.Tensor) -> torch.Tensor:
    """
    Simple forward evaluation helper.
    """
    model.eval()
    with torch.no_grad():
        return model(x)

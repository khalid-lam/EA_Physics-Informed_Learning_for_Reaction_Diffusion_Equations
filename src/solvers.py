# src/solvers.py
import torch

from losses import pde_residual_loss, energy_loss, dirichlet_bc_loss
from losses import l2_regularization, data_loss as data_mse_loss

def train(
    model,
    equation,
    x_interior: torch.Tensor,
    x_boundary: torch.Tensor | None = None,
    u_boundary: torch.Tensor | None = None,
    *,
    # Optional supervised data term
    x_data: torch.Tensor | None = None,
    u_data: torch.Tensor | None = None,
    # Objective choice
    use_energy: bool = False,
    # Weights
    w_pde: float = 1.0,
    w_bc: float = 1.0,
    w_data: float = 0.0,
    w_reg: float = 0.0,
    # Optim
    lr: float = 1e-3,
    steps: int = 2000,
    print_every: int = 200,
    # Mini-batch / SGD controls
    batch_size: int | None = None,
    shuffle: bool = True,
) -> dict:
    """
    Train a model with Adam.

    Supports:
    - PDE residual loss OR energy (variational) loss
    - optional Dirichlet BC loss
    - optional supervised data loss
    - optional L2 regularization
    - mini-batch stochastic gradient descent via batch_size

    Parameters
    ----------
    batch_size : int or None
        If None (default): full-batch training; `steps` = number of gradient updates.
        If set and < number of interior points: mini-batch SGD; `steps` = number of epochs.
        Boundary and data terms are always computed on full batch (more stable).
    shuffle : bool
        Whether to shuffle interior points each epoch (only used when batch_size is set).

    Returns
    -------
    history : dict
        Contains loss_total and all components.
        Length = steps if batch_size is None, else steps * ceil(n_interior/batch_size).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "loss_total": [],
        "loss_pde": [],
        "loss_bc": [],
        "loss_data": [],
        "loss_reg": [],
    }

    # Determine whether we're doing full-batch or mini-batch SGD.
    # When batch_size is set and is less than n_interior, interpret steps as epochs.
    n_in = x_interior.size(0)
    is_minibatch = batch_size is not None and batch_size < n_in

    global_step = 0

    if not is_minibatch:
        # Full-batch mode: steps = number of gradient updates (original behaviour)
        for step in range(1, steps + 1):
            global_step = step
            opt.zero_grad()

            # PDE part (full batch)
            if use_energy:
                lpde = energy_loss(model, equation, x_interior)
            else:
                lpde = pde_residual_loss(model, equation, x_interior)

            # BC part
            if x_boundary is None or u_boundary is None:
                lbc = torch.tensor(0.0, device=x_interior.device, dtype=x_interior.dtype)
            else:
                lbc = dirichlet_bc_loss(model, x_boundary, u_boundary)

            # Data part
            if (x_data is None) or (u_data is None) or (w_data == 0.0):
                ldata = torch.tensor(0.0, device=x_interior.device, dtype=x_interior.dtype)
            else:
                ldata = data_mse_loss(model, x_data, u_data)

            # Reg part
            if w_reg == 0.0:
                lreg = torch.tensor(0.0, device=x_interior.device, dtype=x_interior.dtype)
            else:
                lreg = l2_regularization(model)

            # Total
            ltot = w_pde * lpde + w_bc * lbc + w_data * ldata + w_reg * lreg

            ltot.backward()
            opt.step()

            # Log
            history["loss_total"].append(ltot.detach().item())
            history["loss_pde"].append(lpde.detach().item())
            history["loss_bc"].append(lbc.detach().item())
            history["loss_data"].append(ldata.detach().item())
            history["loss_reg"].append(lreg.detach().item())

            if print_every > 0 and (global_step % print_every == 0 or global_step == 1):
                print(
                    f"[step {global_step:5d}] "
                    f"total={history['loss_total'][-1]:.6e} "
                    f"pde={history['loss_pde'][-1]:.6e} "
                    f"bc={history['loss_bc'][-1]:.6e} "
                    f"data={history['loss_data'][-1]:.6e} "
                    f"reg={history['loss_reg'][-1]:.6e} "
                    f"epoch={global_step}/{steps}"
                )

    else:
        # Mini-batch mode: steps = number of epochs
        for epoch in range(1, steps + 1):
            # Create permutation for this epoch
            if shuffle:
                perm = torch.randperm(n_in, device=x_interior.device)
            else:
                perm = torch.arange(n_in, device=x_interior.device)
            
            # Split into batches
            for batch_start in range(0, n_in, batch_size):
                batch_end = min(batch_start + batch_size, n_in)
                idx = perm[batch_start:batch_end]

                global_step += 1
                opt.zero_grad()

                # PDE part (mini-batch)
                xin = x_interior[idx]
                if use_energy:
                    lpde = energy_loss(model, equation, xin)
                else:
                    lpde = pde_residual_loss(model, equation, xin)

                # BC part (full-batch)
                if x_boundary is None or u_boundary is None:
                    lbc = torch.tensor(0.0, device=xin.device, dtype=xin.dtype)
                else:
                    lbc = dirichlet_bc_loss(model, x_boundary, u_boundary)

                # Data part (full-batch)
                if (x_data is None) or (u_data is None) or (w_data == 0.0):
                    ldata = torch.tensor(0.0, device=xin.device, dtype=xin.dtype)
                else:
                    ldata = data_mse_loss(model, x_data, u_data)

                # Reg part
                if w_reg == 0.0:
                    lreg = torch.tensor(0.0, device=xin.device, dtype=xin.dtype)
                else:
                    lreg = l2_regularization(model)

                # Total
                ltot = w_pde * lpde + w_bc * lbc + w_data * ldata + w_reg * lreg

                ltot.backward()
                opt.step()

                # Log
                history["loss_total"].append(ltot.detach().item())
                history["loss_pde"].append(lpde.detach().item())
                history["loss_bc"].append(lbc.detach().item())
                history["loss_data"].append(ldata.detach().item())
                history["loss_reg"].append(lreg.detach().item())

                if print_every > 0 and (global_step % print_every == 0 or global_step == 1):
                    print(
                        f"[step {global_step:5d}] "
                        f"total={history['loss_total'][-1]:.6e} "
                        f"pde={history['loss_pde'][-1]:.6e} "
                        f"bc={history['loss_bc'][-1]:.6e} "
                        f"data={history['loss_data'][-1]:.6e} "
                        f"reg={history['loss_reg'][-1]:.6e} "
                        f"epoch={epoch}/{steps}"
                    )

    return history


def evaluate_model(model, x: torch.Tensor) -> torch.Tensor:
    """
    Simple forward evaluation helper.
    """
    model.eval()
    with torch.no_grad():
        return model(x)

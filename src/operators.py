# src/operators.py
import torch


def gradient(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute grad(u) with respect to x using autograd.

    Parameters
    ----------
    u : torch.Tensor
        Shape (N, 1) (or (N,) but (N,1) is recommended).
    x : torch.Tensor
        Shape (N, d), must have requires_grad=True.

    Returns
    -------
    grad_u : torch.Tensor
        Shape (N, d).
    """
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True to compute gradients.")

    if u.ndim == 1:
        u = u.unsqueeze(1)

    if u.shape[0] != x.shape[0]:
        raise ValueError("u and x must have the same batch size.")

    # If u is not connected to the autograd graph, its gradient w.r.t x is zero.
    if not u.requires_grad and u.grad_fn is None:
        return torch.zeros_like(x)

    ones = torch.ones_like(u)
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]

    # If u does not depend on x, autograd returns None (allow_unused=True).
    if grad_u is None:
        grad_u = torch.zeros_like(x)

    return grad_u


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True to compute derivatives.")

    if u.ndim == 1:
        u = u.unsqueeze(1)

    if u.shape[0] != x.shape[0]:
        raise ValueError("u and x must have the same batch size.")

    # If u is not connected to the autograd graph, its Laplacian is zero.
    if (not u.requires_grad) and (u.grad_fn is None):
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

    grad_u = gradient(u, x)  # (N, d)
    N, d = grad_u.shape

    lap = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
    for i in range(d):
        gi = grad_u[:, i : i + 1]  # (N,1)

        # If gi is not connected to the graph, its derivative is zero.
        if (not gi.requires_grad) and (gi.grad_fn is None):
            continue

        ones = torch.ones_like(gi)
        dgi = torch.autograd.grad(
            outputs=gi,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]

        if dgi is None:
            continue

        lap = lap + dgi[:, i : i + 1]

    return lap

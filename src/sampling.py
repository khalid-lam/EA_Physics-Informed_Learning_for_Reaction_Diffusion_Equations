# src/sampling.py
import torch


def sample_uniform_interior(n: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """
    Sample interior points uniformly in the hypercube [0, 1]^dim.

    Returns
    -------
    x : (n, dim)
    """
    return torch.rand(n, dim, device=device, dtype=dtype)


def sample_uniform_boundary(n: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """
    Sample boundary points uniformly on the boundary of [0, 1]^dim.

    Strategy
    --------
    - Sample x ~ U([0,1]^dim)
    - Pick a random coordinate j per point
    - Set x[j] to either 0 or 1 (random side)

    Returns
    -------
    x_b : (n, dim)
    """
    x = torch.rand(n, dim, device=device, dtype=dtype)

    # Choose which coordinate is fixed to the boundary for each point
    j = torch.randint(low=0, high=dim, size=(n,), device=device)

    # Choose side 0 or 1
    side = torch.randint(low=0, high=2, size=(n,), device=device)
    side = side.to(dtype=torch.float32)  # 0.0 or 1.0
    if dtype is not None:
        side = side.to(dtype=dtype)
    if device is not None:
        side = side.to(device=device)

    x[torch.arange(n, device=device), j] = side
    return x


def make_dirichlet_bc(
    x_boundary: torch.Tensor,
    bc_fn,
) -> torch.Tensor:
    """
    Compute Dirichlet boundary values u_b = bc_fn(x_b).

    Parameters
    ----------
    x_boundary : (Nb, d)
    bc_fn : callable
        bc_fn(x_b) -> (Nb,1) or (Nb,)

    Returns
    -------
    u_boundary : (Nb,1)
    """
    u_b = bc_fn(x_boundary)
    if u_b.ndim == 1:
        u_b = u_b.unsqueeze(1)
    return u_b

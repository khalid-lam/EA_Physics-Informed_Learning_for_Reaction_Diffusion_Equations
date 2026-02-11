# src/data_loader.py
import torch
from dataclasses import dataclass


@dataclass
class Dataset:
    """
    Simple container for supervised PDE data.

    Attributes
    ----------
    x : torch.Tensor
        Inputs, shape (N, d)
    u : torch.Tensor
        Targets, shape (N, 1)
    """
    x: torch.Tensor
    u: torch.Tensor


def ensure_column(u: torch.Tensor) -> torch.Tensor:
    """
    Ensure u has shape (N,1).
    """
    if u.ndim == 1:
        return u.unsqueeze(1)
    return u


def normalize_coords_to_unit_interval(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    Normalize each coordinate independently to [0, 1] using min-max scaling.

    This is robust to datasets whose coordinates are in [-0.5, 0.5), [0, 1], etc.

    Parameters
    ----------
    x : (N, d)
    eps : float
        Numerical stability for degenerate dimensions.

    Returns
    -------
    x01 : (N, d)
        Normalized coordinates.
    info : dict
        Contains per-dimension min/max used for scaling.
    """
    x_min = torch.min(x, dim=0).values
    x_max = torch.max(x, dim=0).values
    scale = (x_max - x_min).clamp_min(eps)
    x01 = (x - x_min) / scale

    info = {
        "x_min": x_min,
        "x_max": x_max,
    }
    return x01, info


def load_torch_dataset(
    path: str,
    *,
    normalize_to_unit: bool = True,
    eps: float = 1e-12,
) -> tuple[Dataset, dict]:
    """
    Load a dataset saved with torch.save({...}, path).

    Expected keys in the saved dict
    -------------------------------
    - "x": (N, d)
    - "u": (N, 1) or (N,)

    Optional keys are ignored.

    Parameters
    ----------
    path : str
        Path to the .pt file.
    normalize_to_unit : bool
        If True, normalize x to [0,1]^d using per-dimension min-max.
        If False, keep x as-is.
    eps : float
        Stability for degenerate dimensions in min-max scaling.

    Returns
    -------
    dataset : Dataset
    info : dict
        Contains normalization metadata and basic stats.
    """
    obj = torch.load(path, map_location="cpu")

    if "x" not in obj or "u" not in obj:
        raise ValueError('Expected keys "x" and "u" in saved dataset.')

    x = obj["x"]
    u = ensure_column(obj["u"])

    if not torch.is_tensor(x) or not torch.is_tensor(u):
        raise ValueError('"x" and "u" must be torch tensors.')

    info = {
        "path": path,
        "original_min": torch.min(x, dim=0).values,
        "original_max": torch.max(x, dim=0).values,
        "normalized": False,
    }

    if normalize_to_unit:
        x01, norm_info = normalize_coords_to_unit_interval(x, eps=eps)
        x = x01
        info["normalized"] = True
        info["x_min_used"] = norm_info["x_min"]
        info["x_max_used"] = norm_info["x_max"]
        info["new_min"] = torch.min(x, dim=0).values
        info["new_max"] = torch.max(x, dim=0).values

        # Basic sanity check: should be within [0,1] (up to tiny numerical noise)
        if torch.any(x < -1e-6) or torch.any(x > 1.0 + 1e-6):
            raise RuntimeError("Normalization failed: x is not within [0,1] bounds.")

    return Dataset(x=x, u=u), info

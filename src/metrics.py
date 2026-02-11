# src/metrics.py
import torch


def mse(u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error between two tensors of same shape.
    """
    return torch.mean((u_pred - u_true) ** 2)


def rmse(u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    """
    Root mean squared error.
    """
    return torch.sqrt(mse(u_pred, u_true))


def l2_error(u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    """
    L2 error: ||u_pred - u_true||_2
    """
    diff = u_pred - u_true
    return torch.sqrt(torch.sum(diff ** 2))


def rel_l2_error(u_pred: torch.Tensor, u_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Relative L2 error: ||u_pred - u_true||_2 / (||u_true||_2 + eps)
    """
    num = l2_error(u_pred, u_true)
    den = torch.sqrt(torch.sum(u_true ** 2)) + eps
    return num / den


def linf_error(u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    """
    L-infinity error: max |u_pred - u_true|
    """
    return torch.max(torch.abs(u_pred - u_true))


def summarize_errors(u_pred: torch.Tensor, u_true: torch.Tensor) -> dict:
    """
    Return a dict of common scalar error metrics.
    """
    out = {
        "mse": mse(u_pred, u_true).item(),
        "rmse": rmse(u_pred, u_true).item(),
        "l2": l2_error(u_pred, u_true).item(),
        "rel_l2": rel_l2_error(u_pred, u_true).item(),
        "linf": linf_error(u_pred, u_true).item(),
    }
    return out

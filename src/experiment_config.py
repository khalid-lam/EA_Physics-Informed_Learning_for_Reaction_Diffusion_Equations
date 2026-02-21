"""
Experiment configuration and runners for PINN studies.

Defines:
- ExperimentConfig: dataclass capturing all experiment settings
- Model factories: reusable model instantiation
- Experiment creators: canned configs for Poisson, Fisher-KPP
- run_single_experiment: unified training loop
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import torch
from metrics import summarize_errors


@dataclass
class ExperimentConfig:
    """Encapsulates all settings for a single experiment run."""
    
    # Identity
    name: str
    
    # Equation & model
    equation_name: str  # "poisson" or "fisher"
    model_name: str     # "mlp" or "fourier"
    
    # Loss objective
    use_energy: bool = False
    
    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 0.0
    w_data: float = 0.0
    w_reg: float = 0.0
    
    # Collocation points
    n_interior: int = 5000
    n_boundary: int = 1000
    
    # Data configuration
    n_data_points: int = 200
    use_noisy_data: bool = True
    noise_level: float = 0.05
    
    # Training hyperparameters
    lr: float = 1e-4
    steps: int = 2000
    print_every: int = 200


def create_model_factory(device: torch.device, dtype: torch.dtype) -> Dict[str, Callable]:
    """
    Returns a dict of model factory functions.
    Each factory takes no arguments and returns an initialized model on the given device/dtype.
    """
    from models import MLPModel, LinearFourierModel
    
    return {
        'mlp': lambda: MLPModel(
            input_dim=2,
            hidden_dim=64,
            num_hidden_layers=4,
            activation="tanh"
        ).to(device),
        'fourier': lambda: LinearFourierModel(
            input_dim=2,
            max_freq=20
        ).to(device),
    }


def create_poisson_configs() -> list[ExperimentConfig]:
    """
    Curated experiment configs for Poisson equation.
    
    Returns:
        List of ExperimentConfig objects for Poisson experiments
    """
    return [
        # No regularization variants
        ExperimentConfig(
            name="MLP residual no-reg",
            equation_name="poisson",
            model_name="mlp",
            use_energy=False,
            w_pde=1.0,
            w_bc=10, 
            w_data=0.0,
            w_reg=0.0,
            n_interior=6000,
            n_boundary=1200,
            steps=2000,
            lr=1e-4,
            print_every=400,
        ),
        ExperimentConfig(
            name="MLP energy no-reg",
            equation_name="poisson",
            model_name="mlp",
            use_energy=True,
            w_pde=1.0,
            w_bc=10,
            w_data=0.0,
            w_reg=0.0,
            n_interior=6000,
            n_boundary=1200,
            steps=2000,
            lr=1e-4,
            print_every=400,
        ),
        ExperimentConfig(
            name="Fourier residual no-reg",
            equation_name="poisson",
            model_name="fourier",
            use_energy=False,
            w_pde=1.0,
            w_bc=10,
            w_data=0.0,
            w_reg=0.0,
            n_interior=6000,
            n_boundary=1200,
            steps=2000,
            lr=1e-4,
            print_every=400,
        ),
        ExperimentConfig(
            name="Fourier energy no-reg",
            equation_name="poisson",
            model_name="fourier",
            use_energy=True,
            w_pde=1.0,
            w_bc=10,
            w_data=0.0,
            w_reg=0.0,
            n_interior=6000,
            n_boundary=1200,
            steps=2000,
            lr=1e-4,
            print_every=400,
        ),
        # Regularization variants
        ExperimentConfig(
            name="MLP residual reg",
            equation_name="poisson",
            model_name="mlp",
            use_energy=False,
            w_pde=1.0,
            w_bc=10,
            w_data=0.0,
            w_reg=1e-3,
            n_interior=6000,
            n_boundary=1200,
            steps=2000,
            lr=1e-4,
            print_every=400,
        ),
        ExperimentConfig(
            name="Fourier residual reg",
            equation_name="poisson",
            model_name="fourier",
            use_energy=False,
            w_pde=1.0,
            w_bc=10,
            w_data=0.0,
            w_reg=1e-3,
            n_interior=6000,
            n_boundary=1200,
            steps=2000,
            lr=1e-4,
            print_every=400,
        ),
    ]


def create_fisher_configs() -> list[ExperimentConfig]:
    """
    Curated experiment configs for Fisher-KPP / Logistic dataset.
    
    Returns:
        List of ExperimentConfig objects for Fisher experiments
    """
    return [
        # With noisy data variants
        ExperimentConfig(
            name="MLP residual + noisy data",
            equation_name="fisher",
            model_name="mlp",
            use_energy=False,
            w_pde=1.0,
            w_bc=1.0,
            w_data=0.5,
            w_reg=0.0,
            n_interior=8000,
            n_boundary=0,
            n_data_points=500,
            use_noisy_data=True,
            noise_level=0.05,
            lr=1e-3,
            steps=2500,
            print_every=500,
        ),
        ExperimentConfig(
            name="MLP residual + noisy data + reg",
            equation_name="fisher",
            model_name="mlp",
            use_energy=False,
            w_pde=1.0,
            w_bc=1.0,
            w_data=0.5,
            w_reg=1e-6,
            n_interior=8000,
            n_boundary=0,
            n_data_points=500,
            use_noisy_data=True,
            noise_level=0.05,
            lr=1e-3,
            steps=2500,
            print_every=500,
        ),
        ExperimentConfig(
            name="MLP energy + noisy data",
            equation_name="fisher",
            model_name="mlp",
            use_energy=True,
            w_pde=1.0,
            w_bc=1.0,
            w_data=1.0,
            w_reg=0.0,
            n_interior=8000,
            n_boundary=0,
            n_data_points=500,
            use_noisy_data=True,
            noise_level=0.05,
            lr=1e-3,
            steps=2500,
            print_every=500,
        ),
        ExperimentConfig(
            name="Fourier residual + noisy data",
            equation_name="fisher",
            model_name="fourier",
            use_energy=False,
            w_pde=1.0,
            w_bc=1.0,
            w_data=1.0,
            w_reg=0.0,
            n_interior=8000,
            n_boundary=0,
            n_data_points=500,
            use_noisy_data=True,
            noise_level=0.05,
            lr=1e-3,
            steps=2500,
            print_every=500,
        ),
        ExperimentConfig(
            name="Fourier residual + noisy data + reg",
            equation_name="fisher",
            model_name="fourier",
            use_energy=False,
            w_pde=1.0,
            w_bc=1.0,
            w_data=1.0,
            w_reg=1e-6,
            n_interior=8000,
            n_boundary=0,
            n_data_points=500,
            use_noisy_data=True,
            noise_level=0.05,
            lr=1e-3,
            steps=2500,
            print_every=500,
        ),
        ExperimentConfig(
            name="Fourier energy + noisy data",
            equation_name="fisher",
            model_name="fourier",
            use_energy=True,
            w_pde=1.0,
            w_bc=1.0,
            w_data=1.0,
            w_reg=0.0,
            n_interior=8000,
            n_boundary=0,
            n_data_points=500,
            use_noisy_data=True,
            noise_level=0.05,
            lr=1e-3,
            steps=2500,
            print_every=500,
        ),
        # PDE-only variants (no data fitting)
        ExperimentConfig(
            name="MLP residual PDE-only",
            equation_name="fisher",
            model_name="mlp",
            use_energy=False,
            w_pde=1.0,
            w_bc=1.0,
            w_data=0.0,
            w_reg=0.0,
            n_interior=8000,
            n_boundary=0,
            lr=1e-3,
            steps=2500,
            print_every=500,
        ),
        ExperimentConfig(
            name="Fourier residual PDE-only",
            equation_name="fisher",
            model_name="fourier",
            use_energy=False,
            w_pde=1.0,
            w_bc=1.0,
            w_data=0.0,
            w_reg=0.0,
            n_interior=8000,
            n_boundary=0,
            lr=1e-3,
            steps=2500,
            print_every=500,
        ),
    ]


def run_single_experiment(
    config: ExperimentConfig,
    equation,
    model_factory: Callable,
    x_interior: torch.Tensor,
    x_boundary: Optional[torch.Tensor] = None,
    u_boundary: Optional[torch.Tensor] = None,
    x_data: Optional[torch.Tensor] = None,
    u_data_clean: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    grid_xs: Optional[torch.Tensor] = None,
    grid_points: Optional[torch.Tensor] = None,
    u_true_fn: Optional[Callable] = None,
    val_ds = None,
) -> dict:
    """
    Run a single experiment: train model and evaluate.
    
    Args:
        config: ExperimentConfig with all settings
        equation: PDE equation object
        model_factory: Callable() -> model on device/dtype
        x_interior, x_boundary, u_boundary: collocation points & BCs
        x_data, u_data_clean: training data (will add noise if config.use_noisy_data)
        device, dtype: torch device and dtype
        grid_xs, grid_points: evaluation grid coords
        u_true_fn: optional callable for exact solution on grid
        val_ds: optional validation dataset (for Fisher)
    
    Returns:
        Dict with keys: name, history, xs, grid, u_pred_grid, metrics_grid, model, etc.
    """
    from solvers import train
    
    # Prepare data with noise if needed
    u_data = None
    if x_data is not None and u_data_clean is not None:
        u_data = u_data_clean.clone()
        if config.use_noisy_data:
            u_range = u_data.max() - u_data.min()
            noise = torch.randn_like(u_data) * config.noise_level * u_range
            u_data = u_data + noise
    
    # Create and train model
    model = model_factory()
    
    history = train(
        model,
        equation,
        x_interior,
        x_boundary=x_boundary,
        u_boundary=u_boundary,
        x_data=x_data,
        u_data=u_data,
        use_energy=config.use_energy,
        w_pde=config.w_pde,
        w_bc=config.w_bc,
        w_data=config.w_data,
        w_reg=config.w_reg,
        lr=config.lr,
        steps=config.steps,
        print_every=config.print_every,
    )
    
    # Evaluate on grid
    out = {
        "name": config.name,
        "config": config,
        "history": history,
        "model": model,
    }
    
    if grid_xs is not None and grid_points is not None:
        with torch.no_grad():
            u_pred_grid = model(grid_points).detach()
        out["xs"] = grid_xs
        out["grid"] = grid_points
        out["u_pred_grid"] = u_pred_grid
        
        # Metrics on grid if true solution available
        if u_true_fn is not None:
            with torch.no_grad():
                u_true_grid = u_true_fn(grid_points).detach()
            out["u_true_grid"] = u_true_grid
            out["metrics_grid"] = summarize_errors(u_pred_grid, u_true_grid)
    
    # Validation metrics (for Fisher)
    if val_ds is not None:
        model.eval()
        with torch.no_grad():
            u_pred_val = model(val_ds.x)
        out["metrics_val"] = summarize_errors(u_pred_val, val_ds.u)
    
    return out

# Provide a module-level default mapping for convenience (e.g., notebooks)
try:
    model_factories = create_model_factory(torch.device("cpu"), torch.float32)
except Exception:
    model_factories = {}

# scripts/run_experiment.py
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from data_loader import load_torch_dataset, train_val_split
from sampling import sample_uniform_interior, sample_uniform_boundary, make_dirichlet_bc
from equations import PoissonEquation, FisherKPPStationaryEquation
from models import MLPModel, LinearFourierModel
from solvers import train
from metrics import summarize_errors


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    # -------------------------
    # Experiment configuration
    # -------------------------
    data_path = ROOT / "data" / "logistic_mu0.100_sx50_sy50_coef2.667.pt"
    normalize_to_unit = True

    equation_name = "fisher"  # "poisson" or "fisher"
    model_name = "mlp"        # "mlp" or "fourier"

    # ===============================
    # Loss weights
    # ===============================
    use_energy = False
    w_pde = 1.0               # PDE residual weight
    w_bc = 1.0                # Boundary condition weight
    w_data = 0.5              # Data fitting weight
    w_reg = 0.01              # L2 regularization weight
    
    # ===============================
    # Data configuration
    # ===============================
    n_data_points = 200       # EXPLICIT: how many data points to use
    use_noisy_data = True     # Add noise to data?
    noise_level = 0.05        # Noise std (% of range)
    
    # ===============================
    # Collocation point configuration
    # ===============================
    n_interior = 5000         # Interior collocation points
    n_boundary = 1000         # Boundary collocation points

    # ===============================
    # Training configuration
    # ===============================
    lr = 1e-3
    steps = 2000
    print_every = 200

    # -------------------------
    # Load and prepare data
    # -------------------------
    ds, info = load_torch_dataset(data_path, normalize_to_unit=normalize_to_unit)
    ds = type(ds)(x=ds.x.to(device=device, dtype=dtype), u=ds.u.to(device=device, dtype=dtype))

    train_ds, val_ds = train_val_split(ds, val_ratio=0.2, shuffle=True, seed=0)

    print("=== Data Loaded ===")
    print(f"Total points: {ds.x.shape[0]}")
    print(f"Training pool: {train_ds.x.shape[0]} | Validation pool: {val_ds.x.shape[0]}")

    # Select data points
    n_data_points = min(n_data_points, train_ds.x.shape[0])
    x_data = train_ds.x[:n_data_points]
    u_data = train_ds.u[:n_data_points].clone()
    
    # Add noise
    if use_noisy_data:
        u_range = u_data.max() - u_data.min()
        noise = torch.randn_like(u_data) * noise_level * u_range
        u_data = u_data + noise
        print(f"Noise added: σ = {(noise_level * u_range):.6f}")
    
    print("\n=== Configuration Summary ===")
    print(f"Data points (noisy):     {n_data_points}")
    print(f"Interior collocation:    {n_interior}")
    print(f"Boundary collocation:    {n_boundary}")
    print(f"Validation points:       {val_ds.x.shape[0]} (clean)")
    print(f"Data/Collocation ratio:  {n_data_points / (n_interior + n_boundary):.1%}")

    # Domain bounds for sampling:
    d = train_ds.x.shape[1]
    if normalize_to_unit:
        low, high = 0.0, 1.0
    else:
        low = float(train_ds.x.min().item())
        high = float(train_ds.x.max().item())

    # -------------------------
    # Sample collocation points
    # -------------------------
    x_in = sample_uniform_interior(n_interior, d, device=device, dtype=dtype)
    x_b = sample_uniform_boundary(n_boundary, d, device=device, dtype=dtype)

    if not normalize_to_unit:
        x_in = low + (high - low) * x_in
        x_b = low + (high - low) * x_b

    # -------------------------
    # Define equation
    # -------------------------
    if equation_name == "poisson":
        f = lambda x: torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        equation = PoissonEquation(f=f)
    elif equation_name == "fisher":
        equation = FisherKPPStationaryEquation(D=1.0, r=1.0)
    else:
        raise ValueError(f"Unknown equation_name: {equation_name}")

    # -------------------------
    # Define model
    # -------------------------
    if model_name == "mlp":
        model = MLPModel(input_dim=d, hidden_dim=64, num_hidden_layers=4, activation="tanh").to(device)
        params = None  # train all parameters
    elif model_name == "fourier":
        model = LinearFourierModel(input_dim=d, max_frequency=5).to(device)
        params = None  # (for now) or [model.coeffs] when you update train()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # -------------------------
    # Boundary condition (optional)
    # -------------------------
    # For torus datasets, Dirichlet BC may not be appropriate.
    # Here we provide a placeholder BC: u_boundary = 0.
    # You should replace this by your true BC or periodic constraints.
    u_b = torch.zeros(x_b.shape[0], 1, device=device, dtype=dtype)

    # -------------------------
    # Train
    # -------------------------
    history = train(
        model,
        equation,
        x_in,
        x_boundary=x_b,
        u_boundary=u_b,
        x_data=x_data,
        u_data=u_data,
        use_energy=use_energy,
        w_pde=w_pde,
        w_bc=w_bc,
        w_data=w_data,
        w_reg=w_reg,
        lr=lr,
        steps=steps,
        print_every=print_every,
    )

    # -------------------------
    # Evaluate on validation data
    # -------------------------
    model.eval()
    with torch.no_grad():
        u_pred = model(val_ds.x)
    errs = summarize_errors(u_pred, val_ds.u)

    print("\n=== Validation metrics ===")
    for k, v in errs.items():
        print(f"{k:>10s}: {v:.6e}")

    print("\nDone ✅")


if __name__ == "__main__":
    main()

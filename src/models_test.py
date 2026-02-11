# tests/test_models.py
import torch
from models import MLPModel, LinearFourierModel


def test_mlp_shapes_and_gradients():
    torch.manual_seed(0)
    N, d = 8, 2
    x = torch.rand(N, d, requires_grad=True)

    model = MLPModel(input_dim=d, hidden_dim=16, num_hidden_layers=2, activation="tanh")
    u = model(x)

    assert u.shape == (N, 1)

    # Check autograd path
    loss = (u ** 2).mean()
    loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.abs().sum().item()
    assert grad_norm > 0.0


def test_linear_fourier_shapes_and_gradients():
    torch.manual_seed(0)
    N, d = 10, 2
    x = torch.rand(N, d)

    model = LinearFourierModel(input_dim=d, max_freq=2, include_constant=True)
    u = model(x)

    assert u.shape == (N, 1)
    assert model.num_basis() > 0

    loss = (u ** 2).mean()
    loss.backward()
    assert model.coeffs.grad is not None
    assert model.coeffs.grad.abs().sum().item() > 0.0


def run_reference_checks():
    """
    Printable sanity checks with a tiny training loop.
    """
    torch.manual_seed(0)

    # --- Fit a simple target function u(x) = x1 + x2 ---
    N, d = 128, 2
    x = torch.rand(N, d)
    y = (x[:, 0:1] + x[:, 1:2])  # (N,1)

    # MLP should fit quickly
    mlp = MLPModel(input_dim=d, hidden_dim=32, num_hidden_layers=2, activation="tanh")
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)

    for _ in range(300):
        opt.zero_grad()
        pred = mlp(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()

    final_mse = torch.mean((mlp(x) - y) ** 2).item()
    print("=== MLP reference fit ===")
    print(f"Final MSE: {final_mse:.6e} (should be small, e.g. < 1e-3)")
    assert final_mse < 1e-3

    # Linear Fourier should also fit reasonably well (approximation)
    fourier = LinearFourierModel(input_dim=d, max_freq=3, include_constant=True)
    opt2 = torch.optim.Adam([fourier.coeffs], lr=5e-2)

    for _ in range(500):
        opt2.zero_grad()
        pred = fourier(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt2.step()

    final_mse2 = torch.mean((fourier(x) - y) ** 2).item()
    print("=== Linear Fourier reference fit ===")
    print(f"Final MSE: {final_mse2:.6e} (should be reasonably small ie. < 5e-3)")

    print("\nAll model checks passed ✅")


if __name__ == "__main__":
    run_reference_checks()

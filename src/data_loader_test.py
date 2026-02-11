# src/data_loader_test.py
import os
import torch

from data_loader import load_torch_dataset


def run_reference_checks():
    torch.manual_seed(0)

    # --- Create a fake dataset with x in [-0.5, 0.5) ---
    N, d = 1000, 2
    x = torch.rand(N, d) - 0.5  # in [-0.5, 0.5)
    u = torch.sum(x ** 2, dim=1, keepdim=True)

    # Save to a temp file
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/fake_dataset.pt"
    torch.save({"x": x, "u": u}, path)

    # Load + normalize
    ds, info = load_torch_dataset(path, normalize_to_unit=True)

    # Checks
    assert ds.x.shape == (N, d)
    assert ds.u.shape == (N, 1)

    x_min = ds.x.min(dim=0).values
    x_max = ds.x.max(dim=0).values

    print("=== Loader normalization check ===")
    print("Original min:", info["original_min"].tolist())
    print("Original max:", info["original_max"].tolist())
    print("New min     :", x_min.tolist())
    print("New max     :", x_max.tolist())

    # Must be in [0,1] up to tiny numerical tolerance
    assert torch.all(ds.x >= -1e-6)
    assert torch.all(ds.x <= 1.0 + 1e-6)

    # --- Also test normalize_to_unit=False keeps range ---
    ds2, _ = load_torch_dataset(path, normalize_to_unit=False)
    x2_min = ds2.x.min(dim=0).values
    x2_max = ds2.x.max(dim=0).values

    print("\n=== Loader non-normalized check ===")
    print("Min:", x2_min.tolist())
    print("Max:", x2_max.tolist())

    # Should still look like [-0.5,0.5) roughly
    assert x2_min.min().item() < -0.45
    assert x2_max.max().item() > 0.45

    print("\nAll loader checks passed ✅")


if __name__ == "__main__":
    run_reference_checks()

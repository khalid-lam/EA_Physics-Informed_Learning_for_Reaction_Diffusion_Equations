# src/metrics_test.py
import torch
from metrics import mse, rmse, l2_error, rel_l2_error, linf_error, summarize_errors


def run_reference_checks():
    torch.manual_seed(0)

    u_true = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    u_pred = torch.tensor([[2.0], [2.0], [1.0], [8.0]])

    # diff = [1,0,-2,4]
    diff = u_pred - u_true

    mse_ref = torch.mean(diff ** 2)  # (1^2 + 0^2 + (-2)^2 + 4^2)/4 = (1+0+4+16)/4 = 21/4 = 5.25
    rmse_ref = torch.sqrt(mse_ref)
    l2_ref = torch.sqrt(torch.sum(diff ** 2))  # sqrt(21)
    linf_ref = torch.max(torch.abs(diff))      # 4
    rel_l2_ref = l2_ref / (torch.sqrt(torch.sum(u_true ** 2)) + 1e-12)

    m = mse(u_pred, u_true)
    r = rmse(u_pred, u_true)
    l2 = l2_error(u_pred, u_true)
    li = linf_error(u_pred, u_true)
    rl2 = rel_l2_error(u_pred, u_true)

    print("=== Reference metrics check ===")
    print(f"mse   = {m.item():.6e} (ref {mse_ref.item():.6e})")
    print(f"rmse  = {r.item():.6e} (ref {rmse_ref.item():.6e})")
    print(f"l2    = {l2.item():.6e} (ref {l2_ref.item():.6e})")
    print(f"linf  = {li.item():.6e} (ref {linf_ref.item():.6e})")
    print(f"rel_l2= {rl2.item():.6e} (ref {rel_l2_ref.item():.6e})")

    assert torch.allclose(m, mse_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(r, rmse_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(l2, l2_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(li, linf_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(rl2, rel_l2_ref, atol=1e-12, rtol=0.0)

    s = summarize_errors(u_pred, u_true)
    assert "mse" in s and "rmse" in s and "rel_l2" in s and "linf" in s

    print("\nAll metric checks passed ✅")


if __name__ == "__main__":
    run_reference_checks()

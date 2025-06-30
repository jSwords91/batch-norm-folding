from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

torch.manual_seed(0)

class MLPWithBN(nn.Module):
    """Basic MLP with BatchNorm"""
    def __init__(self, in_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)



def fold_batchnorm(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """
    Return a new Linear layer so that BN(Linear(x)) == new_linear(x).
    Uses running mean/var from the trained BN (fixed at inference).
    """
    W = linear.weight.data                       # shape (out, in)
    b = linear.bias.data if linear.bias is not None else torch.zeros_like(bn.running_mean)

    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps) # shape (out,)
    W_folded = scale.view(-1, 1) * W

    b_folded = scale * (b - mean) + beta

    out_features, in_features = W.shape
    folded = nn.Linear(in_features, out_features, bias=True)
    folded.weight.data.copy_(W_folded)
    folded.bias.data.copy_(b_folded)
    return folded


class MLPFolded(nn.Module):
    """MLP with folded BatchNorm"""
    def __init__(self, fc1: nn.Linear, fc2: nn.Linear, fc3: nn.Linear) -> None:
        super().__init__()
        self.fc1, self.fc2, self.fc3 = fc1, fc2, fc3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def bench(model: nn.Module, x: torch.Tensor, repeats: int = 500) -> float:
    """Return average inference time in milliseconds per forward pass."""
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / repeats * 1000.0  # ms


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"{device=}")

    model = MLPWithBN().to(device)
    opt = Adam(model.parameters(), lr=3e-2)
    for _ in range(200):
        x_batch = torch.randn(32, 10, device=device)
        y_batch = torch.randn(32, 1, device=device)
        loss = F.mse_loss(model(x_batch), y_batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()

    # fold BatchNorm layers
    fc1_f = fold_batchnorm(model.fc1, model.bn1)
    fc2_f = fold_batchnorm(model.fc2, model.bn2)
    model_f = MLPFolded(fc1_f, fc2_f, model.fc3).to(device).eval()

    # verify numerical equivalence
    x_test = torch.randn(256, 10, device=device)
    with torch.no_grad():
        y_orig = model(x_test)
        y_fold = model_f(x_test)
    max_err = (y_orig - y_fold).abs().max().item(); print(f"{max_err=}")
    assert max_err < 1e-5, "Folded model does not match original."

    params_before = param_count(model)
    params_after = param_count(model_f)
    print(f"Parameters before folding    : {params_before:,}")
    print(f"Parameters after  folding    : {params_after:,}")
    print(f"Parameters saved              : {params_before - params_after:,}")

    x_bench = torch.randn(1024, 10, device=device)
    ms_before = bench(model, x_bench)
    ms_after = bench(model_f, x_bench)
    speedup = ms_before / ms_after
    print(f"Inference time before folding: {ms_before:.3f} ms")
    print(f"Inference time after  folding: {ms_after:.3f} ms")
    print(f"Speed-up                     : {speedup:.2f}x")


if __name__ == "__main__":
    main()

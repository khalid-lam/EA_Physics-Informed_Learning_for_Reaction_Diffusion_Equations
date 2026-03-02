# src/models.py
import torch
import torch.nn as nn


class MLPModel(torch.nn.Module):
    """
    Simple fully-connected neural network for PINNs.

    Conventions
    ----------
    - Input x: (N, d)
    - Output u(x): (N, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        activation: str = "tanh",
    ):
        super().__init__()

        if activation == "tanh":
            act = torch.nn.Tanh()
        elif activation == "relu":
            act = torch.nn.ReLU()
        elif activation == "silu":
            act = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(act)

        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        layers.append(torch.nn.Linear(hidden_dim, 1))
        self.net = torch.nn.Sequential(*layers)

        # Small initialization helps stability for PINNs
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------
# Residual Block
# ---------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Fully-connected residual block for PINNs.
    """

    def __init__(self, width: int, activation=nn.Tanh):
        super().__init__()

        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)

        self.activation = activation()

        # Optional learnable residual scaling (very useful for PINNs)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return residual + self.alpha * out
        
# ---------------------------------------------------
# ResNet PINN
# ---------------------------------------------------
class ResNetPINN(nn.Module):
    """
    ResNet-style network for Physics-Informed Neural Networks.

    Input:  (N, input_dim)
    Output: (N, 1)
    """

    def __init__(
        self,
        input_dim: int,
        width: int = 128,
        num_blocks: int = 6,
        activation: str = "tanh",
    ):
        super().__init__()

        # Activation selection
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "relu":
            act = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Input layer
        self.input_layer = nn.Linear(input_dim, width)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, activation=act) for _ in range(num_blocks)]
        )

        # Output layer
        self.output_layer = nn.Linear(width, 1)

        self.activation = act()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, input_dim)
        """
        out = self.activation(self.input_layer(x))

        for block in self.blocks:
            out = self.activation(block(out))

        return self.output_layer(out)

class LinearFourierModel(torch.nn.Module):
    """
    Linear basis expansion model using Fourier features on a periodic domain [0, 1]^d.

    u(x) = sum_k c_k * phi_k(x)

    Notes
    -----
    - This is a "linear PINN" style parameterization: linear in parameters c_k.
    - We use real Fourier features: cos(2π k·x) and sin(2π k·x).
    - Output shape is (N, 1).

    Parameters
    ----------
    input_dim : int
        Dimension d.
    max_freq : int
        Maximum integer frequency in each coordinate (k_i in [-max_freq, ..., max_freq]).
        We exclude k=(0,0,...,0) from the sine/cosine set to avoid duplicates with constant.
    include_constant : bool
        Whether to include a constant basis function.
    """
    def __init__(self, input_dim, max_freq=3, include_constant=True):
        super().__init__()
        self.input_dim = input_dim
        self.max_freq = max_freq
        self.include_constant = include_constant

        # symmetric frequencies
        grids = [torch.arange(-max_freq, max_freq + 1) for _ in range(input_dim)]
        mesh = torch.meshgrid(*grids, indexing="ij")
        k = torch.stack([m.reshape(-1) for m in mesh], dim=1)

        mask_nonzero = (k != 0).any(dim=1)
        self.register_buffer("k", k[mask_nonzero].float())

        K = self.k.shape[0]
        n_basis = (1 if include_constant else 0) + 2 * K

        self.coeffs = nn.Parameter(0.01 * torch.randn(n_basis, 1))

    def features(self, x):
        k = self.k.to(x.device, x.dtype)
        phase = 2.0 * torch.pi * (x @ k.T)

        cos_part = torch.cos(phase)
        sin_part = torch.sin(phase)

        feats = []
        if self.include_constant:
            feats.append(torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
        feats.append(cos_part)
        feats.append(sin_part)

        return torch.cat(feats, dim=1)

    def forward(self, x):
        Phi = self.features(x)
        return Phi @ self.coeffs

"""Basic density network policy model."""

import torch
import torch.nn as nn
import torch.functional as F

class MLPPolicy(nn.Module):
    """Multi-layer perceptron policy model with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list,
        fixed_logstd: bool = False,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        self.mean = nn.Sequential(
            nn.Linear(dims[-1], output_dim),
        )
        if fixed_logstd:
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        else:
            self.log_std = nn.Sequential(
                nn.Linear(dims[-1], output_dim),
            )
        self.fixed_logstd = fixed_logstd

    def forward(self, state):
        """Forward pass to compute mean and standard deviation."""
        common = self.net(state)
        mean = self.mean(common)
        log_std = self.log_std(common) if not self.fixed_logstd else self.log_std
        std = torch.exp(log_std)
        return mean, std


import torch
import torch.nn as nn

import torch.nn.functional as F

class MLPPolicy_Full_Cov(nn.Module):
    "Toute matrice sym def positive s'ecrit LL.T avec L triang inf donc ici on "
    "approche Sigma par LL.T"
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        self.output_dim = output_dim
        #petit tric pour ne pas avoir a recalculer a chaque forward les indices
        self.register_buffer("tril_idx", torch.tril_indices(self.output_dim, self.output_dim, 0))
        self.register_buffer('diag_index', torch.arange(self.output_dim))
        # réseau commun
        layers = []
        dims = [input_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU()) #was nn.ReLU()
        self.net = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(dims[-1], output_dim)

        n_tril = output_dim * (output_dim + 1) // 2
        self.tril_layer = nn.Linear(dims[-1], n_tril)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_layer(h)

        # construire la matrice triangulaire
        tril_params = self.tril_layer(h)  # (batch_size, n_tril)
        batch_size = x.shape[0]

        scale_tril = torch.zeros(batch_size, self.output_dim, self.output_dim, device=x.device)

        #tril_indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=0)

        scale_tril[:, self.tril_idx[0], self.tril_idx[1]] = tril_params

        #diag_idx = torch.arange(self.output_dim)
        #modif sur la diag
        diag_coeff=scale_tril[:, self.diag_index, self.diag_index]
        diag_coeff=F.softplus(diag_coeff+1e-3)
        diag_coeff=diag_coeff.clamp(max=1e3)
        scale_tril[:, self.diag_index, self.diag_index]=diag_coeff

        return mean, scale_tril 

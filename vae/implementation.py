import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, z_dim=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar

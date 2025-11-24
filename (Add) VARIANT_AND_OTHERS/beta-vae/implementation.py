import torch
import torch.nn as nn

"""
Illustration

[Input: x (B, 784) ← flatten]
  -> Encoder: Linear + ReLU → h
  -> Heads: mu(h), logvar(h)
  -> Reparam: z = mu + eps * exp(0.5*logvar)
  -> Decoder: Linear + ReLU → recon(x)

Step-by-step
1) Encode input to hidden features
2) Predict latent mean and log-variance
3) Sample latent via reparameterization trick
4) Decode latent to reconstruction
"""
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

def run_example(batch_size=4, image_size=28, z_dim=32):
    torch.manual_seed(0)
    input_dim = image_size*image_size
    model = Model(input_dim=input_dim, z_dim=z_dim).eval()
    x = torch.randn(batch_size, image_size, image_size)
    with torch.no_grad():
        recon, mu, logvar = model(x)
    print("Beta-VAE Example")
    print(f"Input: {tuple(x.shape)} -> flattened to ({x.shape[0]}, {input_dim})")
    print(f"Recon: {tuple(recon.shape)}  Mu: {tuple(mu.shape)}  LogVar: {tuple(logvar.shape)}")
    print("Diagram: [x] -> [Enc] -> [mu, logvar] -> [Reparam z] -> [Dec] -> [recon]")

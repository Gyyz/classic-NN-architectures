import torch
import torch.nn as nn

"""
Illustration

[Noise z (B, z_dim)]
  -> Generator MLP → [Fake image (B, img_dim)]
  -> Discriminator MLP → [Score (B, 1)]

Step-by-step
1) Sample noise as latent input
2) Generator maps noise to image space (tanh outputs)
3) Discriminator judges real/fake on flattened images
"""
class Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def run_example(batch_size=4, z_dim=64, img_dim=784):
    """
    Example

    1) Build a simple GAN (MLP generator + discriminator)
    2) Sample Gaussian noise
    3) Generate fake images and score them
    4) Inspect output shapes
    """
    torch.manual_seed(0)
    G = Generator(z_dim=z_dim, img_dim=img_dim).eval()
    D = Discriminator(img_dim=img_dim).eval()
    z = torch.randn(batch_size, z_dim)
    with torch.no_grad():
        fake = G(z)
        score = D(fake)
    print("GAN Example")
    print(f"Noise z: {tuple(z.shape)}")
    print(f"Fake image: {tuple(fake.shape)}")
    print(f"Discriminator score: {tuple(score.shape)}")
    print("Diagram: [z] -> [G: MLP] -> [x_fake] -> [D: MLP] -> [score]")

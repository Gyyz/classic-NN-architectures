import torch
import torch.nn as nn

"""
Illustration

[Generator]
[Input: noise z (B, z_dim)]
  -> Linear → ReLU → Linear → Tanh → [image (B, img_dim)]

[Discriminator]
[Input: image (B, img_dim)]
  -> Flatten → Linear → LeakyReLU → Linear → [score (B, 1)]

Step-by-step
1) Generator maps noise to image
2) Discriminator scores image
"""
class Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, img_dim), nn.Tanh())
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(img_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1))
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def run_example(batch_size=4, z_dim=64, img_dim=784):
    torch.manual_seed(0)
    G = Generator(z_dim=z_dim, img_dim=img_dim).eval()
    D = Discriminator(img_dim=img_dim).eval()
    z = torch.randn(batch_size, z_dim)
    with torch.no_grad():
        fake = G(z)
        score = D(fake)
    print("StyleGAN Simplified Example")
    print(f"z: {tuple(z.shape)} -> fake image: {tuple(fake.shape)} -> score: {tuple(score.shape)}")
    print("Diagram: [z] -> [G: Linear/ReLU -> Linear/Tanh] -> [image] -> [D: Linear/LeakyReLU -> Linear] -> [score]")

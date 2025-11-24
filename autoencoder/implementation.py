import torch
import torch.nn as nn

"""
Illustration

[Input: images (B, 3, 28, 28) → flatten]
  -> Encoder: Linear + ReLU
  -> Decoder: Linear (reconstruction)

Step-by-step
1) Flatten input
2) Encode to latent features
3) Decode back to input space

Note: This is a placeholder Identity; replace with encoder/decoder blocks for
an actual autoencoder. The example demonstrates shapes and flow.
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
    def forward(self, x):
        return self.identity(x)

def run_example(batch_size=4, image_size=28):
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        out = model(x)
    print("Autoencoder Placeholder Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print("Diagram: [Images] -> [Identity] -> [Images]")

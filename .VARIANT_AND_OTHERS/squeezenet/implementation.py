import torch
import torch.nn as nn

"""
Illustration

[Input: images (B, 3, 32, 32)]
  -> Identity (placeholder) → [Output: (B, 3, 32, 32)]

Step-by-step
1) Accept image tensor
2) Pass through identity (placeholder for Fire modules)
3) Return same-shape output
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
    def forward(self, x):
        return self.identity(x)

def run_example(batch_size=4, image_size=32):
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        out = model(x)
    print("SqueezeNet Placeholder Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print("Diagram: [Images] -> [Identity] -> [Images]")

import torch
import torch.nn as nn

"""
Illustration

[Input: data x (B, D)]
  -> Identity (placeholder for coupling transforms) → [Output: (B, D)]

Step-by-step
1) Accept input vector
2) Pass through identity (placeholder for affine coupling layers)
3) Return same-shape output
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
    def forward(self, x):
        return self.identity(x)

def run_example(batch_size=4, dim=16):
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(batch_size, dim)
    with torch.no_grad():
        out = model(x)
    print("RealNVP Placeholder Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print("Diagram: [x] -> [Identity] -> [x]")

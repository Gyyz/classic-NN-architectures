import torch
import torch.nn as nn

"""
Illustration

[Input: data]
  -> Identity (placeholder for score network) → [Output]

Step-by-step
1) Accept input
2) Pass through identity (placeholder for noise-conditioned score estimation)
3) Return output
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
    print("NCSN Placeholder Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print("Diagram: [x] -> [Identity] -> [x]")

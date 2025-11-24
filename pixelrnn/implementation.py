import torch
import torch.nn as nn

"""
Illustration

[Input: images (B, C, H, W)]
  -> Identity (placeholder for RNN over pixels) → [Output: (B, C, H, W)]

Step-by-step
1) Accept image tensor
2) Pass through identity (placeholder for pixelwise recurrence)
3) Return same-shape output
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
    def forward(self, x):
        return self.identity(x)

def run_example(batch_size=4, channels=3, image_size=32):
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(batch_size, channels, image_size, image_size)
    with torch.no_grad():
        out = model(x)
    print("PixelRNN Placeholder Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print("Diagram: [Images] -> [Identity] -> [Images]")

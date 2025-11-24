import torch
import torch.nn as nn

"""
Illustration

[Input: sequence (B, C, L)]
  -> Temporal path (identity here) → [Output: (B, C, L)]

Step-by-step
1) Accept a 1D temporal signal (channels-first)
2) Pass through identity (placeholder for temporal conv/residual blocks)
3) Return same-shape output

Note: This is a minimal placeholder. A typical TCN uses causal/dilated
1D convolutions with residual connections. The example demonstrates shapes
and data flow; replace Identity with Conv1d blocks for a full TCN.
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
    def forward(self, x):
        return self.identity(x)

def run_example(batch_size=4, channels=8, length=64):
    """
    Example

    1) Build a minimal TCN placeholder
    2) Create dummy temporal data with shape (B, C, L)
    3) Forward pass through identity path
    4) Inspect output shapes
    """
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(batch_size, channels, length)
    with torch.no_grad():
        out = model(x)
    print("TCN Placeholder Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print("Diagram: [Sequence] -> [Identity] -> [Sequence]")

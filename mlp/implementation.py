import torch
import torch.nn as nn

"""
Illustration

[Input: flattened features (B, input_dim)]
  -> Linear(input_dim→hidden_dim) + ReLU
  -> Linear(hidden_dim→output_dim)

Step-by-step
1) Flatten input
2) Two-layer MLP to logits
"""
class Model(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def run_example(batch_size=4, image_size=28, hidden_dim=128, output_dim=10):
    torch.manual_seed(0)
    model = Model(input_dim=image_size*image_size, hidden_dim=hidden_dim, output_dim=output_dim).eval()
    x = torch.randn(batch_size, 1, image_size, image_size)
    with torch.no_grad():
        out = model(x)
    print("MLP Example")
    print(f"Input: {tuple(x.shape)} -> Flattened: {(batch_size, image_size*image_size)}")
    print(f"Output (logits): {tuple(out.shape)}")
    print("Diagram: [Image] -> [Flatten] -> [Linear/ReLU] -> [Linear]")

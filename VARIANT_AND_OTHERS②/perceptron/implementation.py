import torch
import torch.nn as nn

"""
Illustration

[Input: flattened features (B, input_dim)]
  -> Linear(input_dim→output_dim)

Step-by-step
1) Flatten input if needed
2) Apply a single linear layer to get logits
"""
class Model(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def run_example(batch_size=4, image_size=28, output_dim=10):
    torch.manual_seed(0)
    model = Model(input_dim=image_size*image_size, output_dim=output_dim).eval()
    x = torch.randn(batch_size, 1, image_size, image_size)
    with torch.no_grad():
        out = model(x)
    print("Perceptron Example")
    print(f"Input: {tuple(x.shape)} -> Flattened: {(batch_size, image_size*image_size)}")
    print(f"Output (logits): {tuple(out.shape)}")
    print("Diagram: [Image] -> [Flatten] -> [Linear]")

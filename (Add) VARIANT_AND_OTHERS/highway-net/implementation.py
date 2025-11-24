import torch
import torch.nn as nn

"""
Illustration

[Input: images (B, 3, 28, 28) → flatten]
  -> Linear(784→128) + ReLU
  -> Linear(128→num_classes)

Step-by-step
1) Flatten image to a vector
2) Project to hidden features
3) Classify with a final linear layer

Note: A full Highway Network uses gated skip connections between layers.
This minimal MLP demonstrates the flow; gates can be added later.
"""
class Model(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        # Step 1: flatten
        x = x.view(x.size(0), -1)
        # Step 2+3: MLP classifier
        return self.net(x)

def run_example(batch_size=4, image_size=28, num_classes=10):
        torch.manual_seed(0)
        model = Model(input_dim=image_size*image_size*3, output_dim=num_classes).eval()
        x = torch.randn(batch_size, 3, image_size, image_size)
        with torch.no_grad():
            out = model(x)
        print("HighwayNet-like MLP Example")
        print(f"Input: {tuple(x.shape)}")
        print(f"Output (logits): {tuple(out.shape)}")
        print("Diagram: [Images] -> [Flatten] -> [Linear/ReLU] -> [Linear]")

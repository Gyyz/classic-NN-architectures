import torch
import torch.nn as nn

"""
Illustration

[Input: images (B, 3, 32, 32)]
  -> Conv(3→16, 3x3) + ReLU + MaxPool(2)
  -> Conv(16→32, 3x3) + ReLU + MaxPool(2)
  -> Flatten
  -> Linear(32*8*8→64) + ReLU
  -> Linear(64→num_classes)

Step-by-step
1) Extract local features with convolution and pooling
2) Stack deeper features
3) Flatten for classification
4) Fully connected layers to logits
"""
class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def run_example(batch_size=4, image_size=32, num_classes=10):
    torch.manual_seed(0)
    model = Model(num_classes=num_classes).eval()
    x = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        out = model(x)
    print("RetinaNet-like Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output (logits): {tuple(out.shape)}")
    print("Diagram: [Images] -> [Conv/ReLU/Pool]*2 -> [Flatten] -> [FC/ReLU] -> [FC]")

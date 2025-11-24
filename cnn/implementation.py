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
1) Extract local features with convolution and reduce spatial size with pooling
2) Stack deeper features
3) Flatten to a vector for classification
4) Classify with fully connected layers
"""

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Stage 1: local feature extraction and downsampling
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # keep spatial size, expand channels
            nn.ReLU(),
            nn.MaxPool2d(2),               # halve H and W
            # Stage 2: deeper features and further downsampling
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Stage 3+4: flatten and classify
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        # Step 1+2: feature extractor
        x = self.features(x)
        # Step 3: flatten
        x = x.view(x.size(0), -1)
        # Step 4: classifier
        return self.classifier(x)

def run_example(batch_size=4, image_size=32, num_classes=10):
    """
    Example

    1) Build a CNN classifier
    2) Create dummy images
    3) Forward pass through feature extractor and classifier
    4) Inspect output shapes
    """
    torch.manual_seed(0)
    model = Model(num_classes=num_classes).eval()
    x = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        out = model(x)
    print("CNN Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output (logits): {tuple(out.shape)}")
    print("Diagram: [Images] -> [Conv/ReLU/Pool]*2 -> [Flatten] -> [FC/ReLU] -> [FC]")

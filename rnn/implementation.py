import torch
import torch.nn as nn

"""
Illustration

[Input: sequence (B, T, input_dim)]
  -> RNN (hidden_dim, num_layers)
  -> Pool last timestep (B, hidden_dim)
  -> Linear to logits (B, output_dim)

Step-by-step
1) Encode sequence with vanilla RNN
2) Select representation at last timestep
3) Classify via linear head
"""
class Model(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.rnn(x)
        pooled = out[:, -1]
        return self.fc(pooled)

def run_example(batch_size=4, seq_len=16, input_dim=128):
    torch.manual_seed(0)
    model = Model(input_dim=input_dim).eval()
    x = torch.randn(batch_size, seq_len, input_dim)
    with torch.no_grad():
        out = model(x)
    print("RNN Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output (logits): {tuple(out.shape)}")
    print("Diagram: [Seq] -> [RNN] -> [Pool last] -> [Linear]")

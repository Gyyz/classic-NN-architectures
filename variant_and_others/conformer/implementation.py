import torch
import torch.nn as nn

"""
Illustration

[Input: sequences (B, T, input_dim)]
  -> RNN (hidden_dim)
  -> Take last timestep (B, hidden_dim)
  -> Linear to class logits (B, output_dim)

Step-by-step
1) Encode sequence with recurrent cells
2) Select final timestep representation
3) Classify with a linear head

Note: Conformer combines convolution + transformer; this is a simple RNN placeholder.
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
    print("Conformer Placeholder Example")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output (logits): {tuple(out.shape)}")
    print("Diagram: [Seq] -> [RNN] -> [Last T] -> [Linear]")

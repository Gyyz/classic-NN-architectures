import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

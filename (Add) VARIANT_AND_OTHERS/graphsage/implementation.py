import torch
import torch.nn as nn

"""
Illustration

[Input: node features X (N, F), adjacency A (N, N)]
  -> GraphConv: H1 = A·X·W1 + b1
  -> ReLU
  -> GraphConv: H2 = A·H1·W2 + b2

Step-by-step
1) Aggregate neighbor features with adjacency multiplication
2) Apply linear transform per node
3) Stack layers to propagate information
"""
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        h = torch.mm(adj, x)
        return self.lin(h)

class Model(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=7):
        super().__init__()
        self.g1 = GraphConv(in_dim, hidden_dim)
        self.g2 = GraphConv(hidden_dim, out_dim)
    def forward(self, x, adj):
        h = torch.relu(self.g1(x, adj))
        return self.g2(h, adj)

def run_example(num_nodes=20, in_dim=16, num_classes=7):
    torch.manual_seed(0)
    model = Model(in_dim=in_dim, hidden_dim=32, out_dim=num_classes).eval()
    x = torch.randn(num_nodes, in_dim)
    adj = torch.eye(num_nodes)
    with torch.no_grad():
        out = model(x, adj)
    print("GraphSAGE-like Example")
    print(f"Node features X: {tuple(x.shape)}")
    print(f"Adjacency A: {tuple(adj.shape)}")
    print(f"Output (logits per node): {tuple(out.shape)}")
    print("Diagram: [X, A] -> [A·X·W1] -> [ReLU] -> [A·H1·W2]")

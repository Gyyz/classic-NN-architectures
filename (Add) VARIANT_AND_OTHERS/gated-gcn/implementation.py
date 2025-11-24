import torch
import torch.nn as nn

"""
Illustration

[Input: node features (N, in_dim), adjacency (N, N)]
  -> GraphConv(in_dim→hidden_dim) + ReLU
  -> GraphConv(hidden_dim→out_dim)

Step-by-step
1) Aggregate neighbor features via adjacency
2) Apply linear transforms per layer
3) Produce node-level outputs
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

def run_example(num_nodes=12, in_dim=16, hidden_dim=32, out_dim=7):
    torch.manual_seed(0)
    model = Model(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).eval()
    x = torch.randn(num_nodes, in_dim)
    adj = torch.randn(num_nodes, num_nodes)
    with torch.no_grad():
        out = model(x, adj)
    print("Gated-GCN Simplified Example")
    print(f"x: {tuple(x.shape)}, adj: {tuple(adj.shape)} -> out: {tuple(out.shape)}")
    print("Diagram: [x, adj] -> [GraphConv/ReLU] -> [GraphConv]")

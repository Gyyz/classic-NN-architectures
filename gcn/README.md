# Gcn

Category: Graph

Summary

Graph Convolutional Network with neighborhood aggregation.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Graph: node features X (N×F), adjacency A (N×N)] -> [Normalize: Â = D^{-1/2}(A+I)D^{-1/2}] -> [GraphConv: H^{l+1} = σ(Â H^l W_l)]*L -> [Readout: node Softmax or graph Pool -> FC]
```

Canonical Papers
- Semi-Supervised Classification with Graph Convolutional Networks (Kipf and Welling, 2016) [https://arxiv.org/abs/1609.02907]

Further Reading

- Search for more resources on Gcn.


# Attention

Category: Transformer

Summary

Mechanism to weight inputs by relevance dynamically.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Queries Q, Keys K, Values V] -> [Attention Weights α = softmax(QKᵀ/√d)] -> [Weighted Sum αV] -> [Output]
```

Canonical Papers

Further Reading

- Search for more resources on Attention.


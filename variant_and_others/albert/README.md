# Albert

Category: Transformer

Summary

Parameter-sharing and factorized embeddings for efficiency.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Tokens] -> [Embeddings (factorized) + Positional] -> [Transformer Encoder (parameters shared across layers)]*N -> [CLS Pool] -> [Task Head]
```

Canonical Papers

Further Reading

- Search for more resources on Albert.


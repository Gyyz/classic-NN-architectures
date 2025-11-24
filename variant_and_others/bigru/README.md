# Bigru

Category: Recurrent

Summary

Bidirectional GRU with forward and backward passes.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Sequence] -> [GRU Forward]*T || [GRU Backward]*T -> [Concat h_t^→, h_t^←] -> [Readout -> Head]
```

Canonical Papers

Further Reading

- Search for more resources on Bigru.


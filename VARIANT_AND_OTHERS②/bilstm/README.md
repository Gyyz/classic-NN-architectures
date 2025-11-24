# Bilstm

Category: Recurrent

Summary

Bidirectional LSTM with forward and backward passes.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Sequence] -> [LSTM Forward]*T || [LSTM Backward]*T -> [Concat h_t^→, h_t^←] -> [Readout -> Head]
```

Canonical Papers

Further Reading

- Search for more resources on Bilstm.


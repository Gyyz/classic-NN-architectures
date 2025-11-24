# Densenet

Category: Convolutional

Summary

Dense connectivity pattern to encourage feature reuse.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Image] -> [Dense Blocks: each layer receives concat of all previous] -> [Transition (BN -> Conv -> Pool)] -> [GlobalAvgPool -> FC]
```

Canonical Papers

Further Reading

- Search for more resources on Densenet.


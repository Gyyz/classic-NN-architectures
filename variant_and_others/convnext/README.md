# Convnext

Category: Convolutional

Summary

ConvNet that matches transformer performance with modern design.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Image] -> [Stem] -> [Stages: Depthwise Conv -> LayerNorm -> GELU -> 1×1 Conv]*S -> [GlobalAvgPool -> FC]
```

Canonical Papers

Further Reading

- Search for more resources on Convnext.


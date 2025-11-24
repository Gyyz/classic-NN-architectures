# Resnet

Category: Convolutional

Summary

Residual networks with identity shortcuts enabling very deep models.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Input: C×H×W] -> [Stem: Conv -> BatchNorm -> ReLU] -> [Residual Blocks: (Conv -> BN -> ReLU -> Conv -> BN) + Skip Add]*Stages (with downsampling at stage starts) -> [GlobalAvgPool] -> [Fully Connected -> Softmax/Logits]
```

Canonical Papers
- Deep Residual Learning for Image Recognition (He et al., 2015) [https://arxiv.org/abs/1512.03385]

Further Reading

- Search for more resources on Resnet.


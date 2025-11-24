# Glow

Category: Generative

Summary

Flow-based model with actnorm and invertible 1x1 convs.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Input x] ↔ [Invertible flow steps: ActNorm -> 1×1 Conv -> Affine Coupling]*L -> [Latent z]; Log-likelihood exact; Sample via inverse
```

Canonical Papers
- Glow: Generative Flow with Invertible 1x1 Convolutions (Kingma and Dhariwal, 2018) [https://arxiv.org/abs/1807.03039]

Further Reading

- Search for more resources on Glow.


# Transformer

Category: Transformer

Summary

Attention-only encoder-decoder architecture.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Input tokens] -> [Token Embedding + Positional Encoding] -> [Transformer Blocks: Multi-Head Self-Attention -> Add & LayerNorm -> Feed-Forward -> Add & LayerNorm]*N -> [Output Projection -> Softmax/Logits]
```

Canonical Papers
- Attention Is All You Need (Vaswani et al., 2017) [https://arxiv.org/abs/1706.03762]

Further Reading

- Search for more resources on Transformer.


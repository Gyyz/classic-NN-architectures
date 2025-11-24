# Gpt

Category: Transformer

Summary

Autoregressive transformer for next-token prediction.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Tokens] -> [Embedding + Positional] -> [Decoder-only blocks: Masked Self-Attn -> MLP]*N -> [LM Head -> Next-token probabilities]
```

Canonical Papers
- Improving Language Understanding by Generative Pre-Training (Radford et al., 2018) [https://openai.com/research/language-unsupervised]

Further Reading

- Search for more resources on Gpt.


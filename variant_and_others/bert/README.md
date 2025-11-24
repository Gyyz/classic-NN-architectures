# Bert

Category: Transformer

Summary

Bidirectional transformer pretraining with masked language modeling.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Tokens] -> [Token + Segment + Positional Embeddings] -> [Transformer Encoder]*N -> [CLS Pool] -> [MLM/NSP Heads or Task Head]
```

Canonical Papers
- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018) [https://arxiv.org/abs/1810.04805]

Further Reading

- Search for more resources on Bert.


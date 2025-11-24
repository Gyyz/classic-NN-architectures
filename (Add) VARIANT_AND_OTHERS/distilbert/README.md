# Distilbert

Category: Transformer

Summary

Smaller BERT via knowledge distillation.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Tokens] -> [Embeddings + Positional] -> [Transformer Encoder]*6 (distilled from BERT) -> [CLS Pool] -> [Task Head]
```

Canonical Papers

Further Reading

- Search for more resources on Distilbert.


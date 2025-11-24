# Rnn

Category: Recurrent

Summary

Recurrent network with hidden state over sequences.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Input: sequence x1..xT] -> [Recurrent cell f: h_t = f(x_t, h_{t-1})]*T -> [Readout: h_T or {h_t}] -> [Linear -> Softmax/Logits]
```

Canonical Papers
- Learning Representations by Back-Propagating Errors (Rumelhart et al., 1986) [https://www.nature.com/articles/323533a0]

Further Reading

- Search for more resources on Rnn.


# Lstm

Category: Recurrent

Summary

Gated recurrent cell with memory for long-term dependencies.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Input: sequence x1..xT] -> [LSTM cell with gates (i,f,o,g): c_t,h_t]*T -> [Optional: Bidirectional] -> [Readout] -> [Linear -> Softmax/Logits]
```

Canonical Papers
- Long Short-Term Memory (Hochreiter and Schmidhuber, 1997) [https://www.bioinf.jku.at/publications/older/2604.pdf]

Further Reading

- Search for more resources on Lstm.


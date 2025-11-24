# Vae

Category: Generative

Summary

Variational autoencoder with latent distributions.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Input x] -> [Encoder] -> [Latent µ, logσ] -> [Sample z = µ + σ ⊙ ε] -> [Decoder] -> [Reconstruction x̂]; Loss = Recon + KL(q(z|x) || p(z))
```

Canonical Papers
- Auto-Encoding Variational Bayes (Kingma and Welling, 2013) [https://arxiv.org/abs/1312.6114]

Further Reading

- Search for more resources on Vae.


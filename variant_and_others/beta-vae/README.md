# Beta Vae

Category: Generative

Summary

VAE variant encouraging disentangled representations.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Input x] -> [Encoder] -> [µ, logσ] -> [Sample z] -> [Decoder -> x̂]; Loss = Recon + β · KL(q(z|x) || p(z))
```

Canonical Papers

Further Reading

- Search for more resources on Beta Vae.


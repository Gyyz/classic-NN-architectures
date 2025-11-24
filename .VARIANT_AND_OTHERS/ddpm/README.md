# Ddpm

Category: Generative

Summary

Diffusion model with denoising score matching.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Data x_0] -> [Forward noising q(x_t|x_{t-1})]; Sampling: [x_T ~ N(0,I)] -> [Reverse denoise with UNet εθ] -> [x_0]
```

Canonical Papers
- Denoising Diffusion Probabilistic Models (Ho et al., 2020) [https://arxiv.org/abs/2006.11239]

Further Reading

- Search for more resources on Ddpm.


# Dcgan

Category: Generative

Summary

Convolutional GAN architecture for images.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Noise z] -> [Generator: ConvTranspose -> BN -> ReLU]* -> [Image] ↔ [Discriminator: Conv -> BN -> LeakyReLU]* -> [Adversarial Loss]
```

Canonical Papers
- Unsupervised Representation Learning with Deep Convolutional GANs (Radford et al., 2015) [https://arxiv.org/abs/1511.06434]

Further Reading

- Search for more resources on Dcgan.


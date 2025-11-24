# Cycle Gan

Category: Generative

Summary

Unpaired image-to-image translation with cycle consistency.

Key Ideas

- Architecture motivation
- Core building blocks
- Training considerations
- Typical applications

Detailed Flow

```
[Image A] -> [G_AB] -> [Fake B]; [Image B] -> [G_BA] -> [Fake A]; [D_A,D_B adversarial] + [Cycle-consistency: A ≈ G_BA(G_AB(A)), B ≈ G_AB(G_BA(B))]
```

Canonical Papers

Further Reading

- Search for more resources on Cycle Gan.


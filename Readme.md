# nn-architectures

This repository collects minimal, readable implementations of many neural network architectures. Each directory contains an `implementation.py` that is intentionally small and focused on the core idea. Most implementations include:

- A top-of-file illustration of the forward pass (ASCII diagram)
- A short step-by-step description
- A `run_example()` function that creates dummy inputs, runs a forward pass, and prints input/output shapes

These are meant for learning, experimentation, and quick reference—not as production-ready models.

## Quick Start

Requirements:
- Python 3.9+
- PyTorch: `python3 -m pip install torch`

Run any architecture’s example:

```bash
python3 -c "from resnet.implementation import run_example; run_example()"
```

General pattern:

```bash
python3 -c "from <architecture>.implementation import run_example; run_example()"
```

## Classic Architectures

Below are classic NN families included here, with a quick note and a sample command.

- CNN (Convolutional Neural Network)
  - Feature extraction via `Conv/ReLU/Pool`, classification via `Linear` layers
  - Try: `python3 -c "from cnn.implementation import run_example; run_example()"`

- ResNet (Residual Network)
 
- MLP (Multi-Layer Perceptron)
 
- AutoEncoder (AE)

- VAE (Variational AutoEncoder)
  - Probabilistic encoder (mean/logvar), reparameterization trick, decoder reconstructs
  - Example usage (forward pass):
    ```python
    from vae.implementation import Model
    import torch
    m = Model().eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        recon, mu, logvar = m(x)
    print(recon.shape, mu.shape, logvar.shape)
    ```

- RNN (Recurrent Neural Network)
  
- LSTM (Long Short-Term Memory)
  - RNN variant with gates to combat vanishing gradients

- GCN (Graph Convolutional Network)
  - Message passing over graphs using adjacency-based aggregation
  - Try: `python3 -c "from gcn.implementation import run_example; run_example(num_nodes=12, in_dim=16)"`

## Notes

- These implementations are intentionally minimal and sometimes placeholders to illustrate the core idea.
- Shapes and defaults in `run_example()` are chosen for clarity and speed.
- Many examples rely on random inputs; set `torch.manual_seed(0)` for deterministic prints.
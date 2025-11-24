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
  - Adds residual connections to ease optimization of deep CNNs
  - Try: `python3 -c "from resnet.implementation import run_example; run_example()"`

- MLP (Multi-Layer Perceptron)
  - Fully connected layers with nonlinearities
  - Try: `python3 -c "from mlp.implementation import run_example; run_example()"`

- AutoEncoder (AE)
  - Encoder–decoder that reconstructs inputs
  - Try: `python3 -c "from autoencoder.implementation import run_example; run_example()"`

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
  - Sequential processing, hidden state carries temporal information
  - Try: `python3 -c "from rnn.implementation import run_example; run_example(seq_len=16, input_dim=128)"`

- LSTM (Long Short-Term Memory)
  - RNN variant with gates to combat vanishing gradients
  - Try: `python3 -c "from lstm.implementation import run_example; run_example(seq_len=16, input_dim=128)"`

- GCN (Graph Convolutional Network)
  - Message passing over graphs using adjacency-based aggregation
  - Try: `python3 -c "from gcn.implementation import run_example; run_example(num_nodes=12, in_dim=16)"`

## Repository Structure

- `architecture_name/implementation.py` — minimal PyTorch module, docstring, and often `run_example()`
- `run_architecture.py` — helper script that can run selected architectures (where applicable)

Examples of directories:
- `cnn/`, `resnet/`, `vgg/`, `lenet/`, `inception/`
- `mlp/`, `perceptron/`
- `rnn/`, `lstm/`, `gru/`, `bilstm/`
- `gcn/`, `graphsage/`, `gat/`, `gin/`
- `autoencoder/`, `vae/`, `beta-vae/`
- `gan/`, `dcgan/`, `wgan/`, `stylegan/`, `stylegan2/`
- `ddpm/`, `ddim/`, `stable-diffusion/` (placeholder-style demos)
- `transformer/`, `roberta/`, `t5/`, `vit/`

## Notes

- These implementations are intentionally minimal and sometimes placeholders to illustrate the core idea.
- Shapes and defaults in `run_example()` are chosen for clarity and speed.
- Many examples rely on random inputs; set `torch.manual_seed(0)` for deterministic prints.
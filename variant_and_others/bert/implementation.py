import torch
import torch.nn as nn

"""
Illustration

[Input: token ids (B, T)]
  -> Embedding (B, T, d_model)
  -> TransformerEncoder (layers with self-attention + feedforward)
  -> Pool last token (B, d_model)
  -> Linear to class logits (B, num_classes)

Step-by-step
1) Convert integer tokens to dense vectors
2) Contextualize tokens via stacked self-attention blocks
3) Select a summary token representation
4) Classify with a linear head
"""
class Model(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=2, vocab_size=1000, num_classes=10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        # Step 1: token embedding
        x = self.emb(x)
        # Transformer expects (T, B, d_model)
        x = x.transpose(0, 1)
        # Step 2: contextual encoding
        x = self.encoder(x)
        # Back to (B, T, d_model)
        x = x.transpose(0, 1)
        # Step 3: pool last token representation
        pooled = x[:, -1]
        # Step 4: classification head
        return self.fc(pooled)

def run_example(batch_size=4, seq_len=16, vocab_size=1000):
    """
    Example

    1) Build a small Transformer encoder
    2) Create dummy token ids
    3) Forward pass through embedding and encoder
    4) Inspect output shapes
    """
    torch.manual_seed(0)
    model = Model(vocab_size=vocab_size).eval()
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        out = model(x)
    print("BERT-like Encoder Example")
    print(f"Input ids: {tuple(x.shape)}")
    print(f"Output (logits): {tuple(out.shape)}")
    print("Diagram: [Tokens] -> [Embedding] -> [Self-Attn x N] -> [Pool] -> [Linear]")

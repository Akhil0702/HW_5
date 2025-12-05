import torch
import torch.nn as nn

class SimpleTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(SimpleTransformerEncoderBlock, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # input: (batch, seq, d_model)
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # LayerNorm for both sublayers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Optional dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            attn_mask: (optional) attention mask
            key_padding_mask: (optional) padding mask
        
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # ---- Self-Attention Sub-layer ----
        # MultiHeadAttention expects (batch, seq, d_model) when batch_first=True
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection + LayerNorm
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # ---- Feed-Forward Sub-layer ----
        ffn_output = self.ffn(x)
        
        # Residual connection + LayerNorm
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x
if __name__ == "__main__":
    import torch

    d_model = 512
    num_heads = 8
    d_ff = 2048

    encoder_block = SimpleTransformerEncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )

    # Batch of 32 sequences, each with 10 tokens
    x = torch.randn(32, 10, d_model)

    out = encoder_block(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)


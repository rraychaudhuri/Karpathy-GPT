import torch
import torch.nn.functional as F



class SelfAttentionHead(torch.nn.Module):

    def __init__(self, input_dim, head_size, block_size, dropout=0.2, d="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.key = torch.nn.Linear(input_dim, head_size, bias=False).to(d)
        self.query = torch.nn.Linear(input_dim, head_size, bias=False).to(d)
        self.value = torch.nn.Linear(input_dim, head_size, bias=False).to(d)
        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size).to(d)))
        self.head_size = head_size
    
    def forward(self, x):

        # p(x, "x")

        k = self.key(x)      # (B, T, C) * (C, head_size) => (B, T, head_size)
        q = self.query(x)    # (B, T, C) * (C, head_size) => (B, T, head_size)
        v = self.value(x)    # (B, T, C) * (C, head_size) => (B, T, head_size)

        # p(k, "k")
        # p(q.transpose(-2, -1), "q.T")

        # Interaction
        wei = k @ q.transpose(-2, -1)           # (B, T, head_size) * (B, head_size, T) => (B, T, T)
        wei = wei * (self.head_size ** -0.5)    # Scaled down weights for the softmax layer

        wei = self.dropout(wei)

        # p(wei, "wei")
        # p(self.tril, "tril")

        # Only talk to previous tokens in the batch
        wei = wei.masked_fill(self.tril == 0, float("-inf"))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)                             # (B, T, T)

        out = wei @ v       # (B, T, T) * (B, T, head_size) = (B, T, head_size)
        return out


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, input_dim, head_size, block_size, number_of_heads, dropout=0.2, d="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saHeads = [SelfAttentionHead(input_dim, head_size, block_size, d=d) for i in range(number_of_heads)]

        feature_dim = head_size * number_of_heads
        
        # Projection layer : Why ????
        self.projection = torch.nn.Linear(feature_dim, feature_dim).to(d)
        self.dropout = torch.nn.Dropout(dropout)
        self.d= d
    
    def forward(self, X):
        return self.dropout(self.projection(torch.cat([sa_head(X) for sa_head in self.saHeads], dim=-1).to(self.d)))


class FeedForward(torch.nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.2, d="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 4 * output_dim).to(d),       # As mentioned in paper, project to a 4 times higher dim space
            torch.nn.ReLU().to(d),

            # Projection layer : Why ????
            torch.nn.Linear(4 * output_dim, output_dim).to(d),        
            torch.nn.Dropout(dropout)
        )
    
    def forward(self, X):
        return self.m(X)


class Block(torch.nn.Module):
    """
    Contains:
    - Layernorm #1
    - Multiple Self Attention Heads
    - Layernorm #2
    - Feed Forward
    """

    def __init__(self, input_dim, head_size, block_size, number_of_heads, dropout=0.2, d="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)

        feature_dim = head_size * number_of_heads
        self.layerNorm1 = torch.nn.LayerNorm(input_dim).to(d)
        self.saHeads = MultiHeadSelfAttention(input_dim, head_size, block_size, number_of_heads, dropout=dropout, d=d)
        self.layerNorm2 = torch.nn.LayerNorm(feature_dim).to(d)
        self.ffwd = FeedForward(feature_dim, feature_dim, d=d)

    def forward(self, X):
        X = X + self.saHeads(self.layerNorm1(X))
        out = X + self.ffwd(self.layerNorm2(X))
        return out
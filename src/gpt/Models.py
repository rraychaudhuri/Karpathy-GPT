import torch
import torch.nn.functional as F
from .Attention import MultiHeadSelfAttention, FeedForward, Block


class CustomModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def generate(self, context, max_characters=10, max_block_size=8, d="cpu"):
        self.eval()
        context = torch.tensor([context.tolist()], device=d)
        for _ in range(max_characters):
            input = context[:, -max_block_size:]
            logits, _ = self(input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1).to(d)
            context = torch.cat((context, ix), dim=1).to(d)
        self.train()
        return context.squeeze().tolist()


class BaseModel(CustomModel):
    """
    Model with an embedding, positional embedding and a linear layer
    """

    def __init__(self, vocab_length, 
                 feature_dim, 
                 output_dim, 
                 block_size, 
                 d="cpu", 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_length, embedding_dim=feature_dim, device=d)
        self.pos_embedding = torch.nn.Embedding(num_embeddings=block_size, embedding_dim=feature_dim, device=d)
        self.liner = torch.nn.Linear(feature_dim, output_dim, bias=True)
        self.d = d
    
    def forward(self, X, target=None):

        X = self.embedding(X) 
        X_pos = self.pos_embedding(torch.arange(X.shape[1], device=self.d))
        X = X + X_pos

        logits = self.liner(X)
        loss = None
        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))
        return logits, loss



class BaseModel_V2(CustomModel):
    """
    Model Contains:

    - character embedding
    - positional embedding 
    - Multiple Self Attention Heads
    - Feed Forward Network
    - Linear Layer

    """

    def __init__(self, vocab_length, 
                 feature_dim, 
                 output_dim, 
                 block_size, 
                 n=4,
                 d="cpu",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.d = d

        # character embedding
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_length, embedding_dim=feature_dim).to(self.d)

        # Positional Embedding
        self.pos_embedding = torch.nn.Embedding(num_embeddings=block_size, embedding_dim=feature_dim).to(self.d)

        # Multiple Attention heads
        self.saHeads = MultiHeadSelfAttention(input_dim=feature_dim, block_size=block_size, head_size=feature_dim//n, number_of_heads=n, d=self.d)

        # Feed Forward
        self.ffwd = FeedForward(feature_dim, feature_dim, d=self.d)

        # Linear Layer
        self.liner = torch.nn.Linear(feature_dim, output_dim, bias=True).to(self.d)

        
    
    def forward(self, X, target=None):

        X = self.embedding(X) 
        X_pos = self.pos_embedding(torch.arange(X.shape[1], device=self.d))
        X = X + X_pos

        X = self.saHeads(X)
        X = self.ffwd(X)

        logits = self.liner(X)
        loss = None
        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))
        return logits, loss
    


class BaseModel_V3(CustomModel):
    """
    Model Contains:

    - character embedding
    - positional embedding 
    - Multiple Self Attention Blocks
    - Feed Forward Network
    - Linear Layer

    """

    def __init__(self, 
                 vocab_length, 
                 feature_dim, 
                 output_dim, 
                 block_size, 
                 number_of_sa_heads=4,
                 number_of_sa_blocks=4,
                 dropout=0.2,
                 d="cpu",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.d = d

        # character embedding
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_length, embedding_dim=feature_dim).to(self.d)

        # Positional Embedding
        self.pos_embedding = torch.nn.Embedding(num_embeddings=block_size, embedding_dim=feature_dim).to(self.d)

        # Multiple  self attention blocks
        self.saBlocks = torch.nn.Sequential(
            *[Block(input_dim=feature_dim, 
                    head_size=feature_dim//number_of_sa_heads, 
                    block_size=block_size, 
                    number_of_heads=number_of_sa_heads, 
                    dropout=dropout,
                    d=self.d) 
             for _ in range(number_of_sa_blocks)],

             torch.nn.LayerNorm(feature_dim)
        )
    
        # Linear Layer
        self.liner = torch.nn.Linear(feature_dim, output_dim, bias=True).to(self.d)

    
    def forward(self, X, target=None):

        X = self.embedding(X) 
        X_pos = self.pos_embedding(torch.arange(X.shape[1], device=self.d))
        X = X + X_pos

        # p(X, "X")
        X = self.saBlocks(X)

        logits = self.liner(X)
        loss = None
        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))
        return logits, loss
import torch
import torch.nn.functional as F


class CustomModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def generate(self, context, max_characters=10, max_block_size=8):
        context = torch.tensor([context.tolist()])
        for _ in range(max_characters):
            
            # print(f"Conetxt:{context.shape}")

            input = context[:, -max_block_size:]

            logits, _ = self(input)
            # print(f"Logits:{logits.shape}")

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # print(f"Probs:{probs.shape}")

            ix = torch.multinomial(probs, num_samples=1)
            
            # print(context)
            context = torch.cat((context, ix), dim=1)
            # print(context)
        
        return context.squeeze().tolist()


class BaseModel(CustomModel):
    """
    Model with only an embedding
    """

    def __init__(self, vocab_length, d="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_length, embedding_dim=vocab_length, device=d)
        self.d = d

    def forward(self, X, target=None):

        logits = self.embedding(X)
        loss = None

        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))
        
        return logits, loss
    


class BaseModel_V2(CustomModel):
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


        
import torch


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()

    def forward(self, x: torch.Tensor):
        # x: [batch, 512]

        # Multi-head attentiion
        x = self.multi_head_attention(x)

        # Feed forward
        x = self.feed_forward(x)

        return x


class FeedForward(torch.nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048):
        super(FeedForward, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )
        self.layer_norm = torch.nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x) + x  # + residual (skip connection)
        x = self.layer_norm(x)
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        self.Wq = torch.nn.Linear(input_dim, input_dim)
        self.Wk = torch.nn.Linear(input_dim, input_dim)
        self.Wv = torch.nn.Linear(input_dim, input_dim)
        self.layer_norm = torch.nn.LayerNorm(input_dim)

        self.num_heads = num_heads

        # Check if input_dim is divisible by num_heads
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) should be divisible by num_heads ({num_heads})"
            )
        self.head_dim = input_dim // num_heads

    def forward(self, x: torch.Tensor):

        # Multi-head attention
        q: torch.Tensor = self.Wq(x)
        k: torch.Tensor = self.Wk(x)
        v: torch.Tensor = self.Wv(x)

        # Split heads
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        d_k = torch.tensor(k.shape[-1])

        # calc attention
        score = torch.matmul(q, k.transpose(-1, -2))
        score = score / torch.sqrt(d_k)
        attention = torch.softmax(score, dim=-1)
        context = torch.matmul(attention, v)

        # Concatenate heads
        context = context.view(-1, 512)

        # Normalization
        x = x + context  # residual
        x = self.layer_norm(x)

        return x


class Embedding(torch.nn.Module):
    """Convert list of indices to list of vectors using embedding layer.

    The indices are the enumeration of the vocabulary. For example,
    if the vocabulary is ["The","apple", "is", "delicious"], then the indices
    are [0, 1, 2, 3].

    If the sentence is "delicious apple", then the input is [3, 1].

    Parameters
    ----------
    num_embeddings : int
        The size of the vocabulary, by default 512.

    dim_embeddings : int
        The dimension of the embedding vectors.
        That is, each word is represented as a vector of dim_embeddings.
        By default 512.

    Returns
    -------
    x : torch.Tensor
        List of embedding vectors.

    Notes
    -----
    The embedding layer is a simple linear layer.
    In the "Attention is All You Need" paper, the embedding layer is shared by
    the encoder and decoder. That is, the same embedding layer is used.
    """

    def __init__(self, num_embeddings: int = 512, dim_embeddings: int = 512) -> None:
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, dim_embeddings)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(indices)
        return x


if __name__ == "__main__":
    layer = TransformerLayer()
    x = torch.randn(10, 512)
    out = layer(x)
    print(out)

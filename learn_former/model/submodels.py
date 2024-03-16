import torch


class FeedForward(torch.nn.Module):
    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 2048, dropout: float = 0.1
    ):
        super(FeedForward, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.Wq = torch.nn.Linear(input_dim, input_dim)
        self.Wk = torch.nn.Linear(input_dim, input_dim)
        self.Wv = torch.nn.Linear(input_dim, input_dim)

        self.dropout = torch.nn.Dropout(dropout)

        self.num_heads = num_heads

        # Check if input_dim is divisible by num_heads
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) should be divisible by num_heads ({num_heads})"
            )

        # Calculate head_dim
        self.head_dim = input_dim // num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        # Get batch size
        batch_size = k.shape[0]

        # Linear transformation of input to query, key, value
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # Split heads
        # q, k, v: [batch, tokens, 512] -> [batch, tokens, num_heads, head_dim]
        # Input dim (512) is split into num_heads (8) and head_dim (64)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to get dimensions [batch, num_heads, tokens, head_dim]
        # head dim (embedding dim) and tokens must be the last two dimensions
        # to allow for correct matrix multiplication
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Get the dimension of the key (head_dim) for scaling the attention scores
        d_k = torch.tensor(k.shape[-1])

        # calc attention scores
        # key is transposed to make matrix multiplication possible,
        # i.e. [batch, num_heads, head_dim, tokens]
        # Matrix mul is then [tokens, head_dim] x [head_dim, tokens] -> [tokens, tokens]
        # The other dimensions (batch and heads) are batchified by pytorch's broadcasting
        score = torch.matmul(q, k.transpose(-1, -2))
        score = score / torch.sqrt(d_k)

        # Mask should be [batch, num_heads, tokens, tokens]
        # Apply mask before softmax
        # -inf leads to 0 in softmax, so no attention is paid to that token
        # Actual -inf is not used because of numerical instability (leads to NaNs)
        if mask is not None:
            score = score.masked_fill(mask == 0, torch.tensor(1e-16))

        # Apply softmax
        # Softmax converts the scores to probabilities that sum to 1
        # The probabilities are used to weight the values
        attention = torch.softmax(score, dim=-1)

        # Apply attention to values
        # Because of the attention scores, some values are weighted more than others
        context = torch.matmul(attention, v)

        # Transpose context back to [batch, tokens, num_heads, head_dim]
        context = context.permute(0, 2, 1, 3).contiguous()
        # Concatenate heads again into one large embedding dim to get [batch, tokens, 512]
        context = context.view(batch_size, -1, self.num_heads * self.head_dim)

        # Apply dropout to avoid overfitting
        x = self.dropout(context)

        return x


class Embedding(torch.nn.Module):
    """Convert list of indices to list of vectors using embedding layer.

    The indices are the enumeration of the vocabulary. For example,
    if the vocabulary is ["The","apple", "is", "delicious"], then the indices
    are [0, 1, 2, 3].

    If the sentence is "delicious apple", then the input is [3, 1].

    Args:
    ----
    num_embeddings : int
        The size of the vocabulary, by default 512.

    dim_embeddings : int
        The dimension of the embedding vectors.
        That is, each word is represented as a vector of dim_embeddings.
        By default 512.

    Forward:
    -------
    indices : torch.Tensor
        List of integer indices.

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


class PositionalEncoding(torch.nn.Module):
    """Add positional encoding to the input.

    Args:
    ----
    max_len : int
        The maximum num of token of the input, by default 512.

    embedding_dims : int
        The dimension of the embedding vectors, by default 512.

    Forward:
    -------
    x : torch.Tensor
        The input tensor.

    Returns
    -------
    x : torch.Tensor
        The input tensor with positional encoding added.

    """

    def __init__(self, max_len: int = 512, embedding_dims: int = 512):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, dim)
        # each row is a position encoding
        pos_encoding = torch.zeros(max_len, embedding_dims)

        # Create a vector of shape (max_len)
        # where each element is a position
        positions = torch.arange(0, max_len)

        # Create a vector of shape (max_len, 1)
        # to allow broadcasting
        positions = positions.unsqueeze(1)

        # Create a vector of shape (dim)
        # where each element is a embedding dimension
        dims = torch.arange(0, embedding_dims, 2)

        pos_encoding[:, 0::2] = torch.sin(
            positions / (10000 ** (dims / embedding_dims))
        )
        pos_encoding[:, 1::2] = torch.cos(
            positions / (10000 ** (dims / embedding_dims))
        )

        self.pos_encoding = torch.nn.Parameter(pos_encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch, tokens, dim = x.shape

        # Add the position encoding to the input
        x = x + self.pos_encoding[:, :tokens, :]  # broadcasting
        return x


class MaskGenerator:
    """Generate padding mask and look-head mask for encoder and decoder."""

    def __init__(self, padding_id: int) -> None:
        super().__init__()
        self.padding_id = padding_id

    def padding_mask(self, x: torch.Tensor) -> torch.Tensor:

        device = x.device

        # x: [batch, tokens]
        batch_size, num_tokens = x.shape

        # Create mask of shape (batch, token)
        mask = torch.ones((batch_size, num_tokens), dtype=torch.bool)

        # mask = 0 where x is padding id
        mask[x == self.padding_id] = 0

        # Add singleton dim to allow broadcasting with multihead attention matrix
        # mask: [batch, 1, 1, tokens]
        mask = mask.unsqueeze(1).unsqueeze(2)

        return mask.to(device)

    def look_ahead_mask(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        # x: [batch, tokens]
        batch_size, num_tokens = x.shape

        # create look-ahead mask
        # mask = [batch, 1 , tokens, tokens]
        # Create a rectangular mask (tokens, tokens)
        mask = torch.ones((batch_size, num_tokens, num_tokens), dtype=torch.bool)

        # Set the upper triangular part of the mask to 0
        # Therefore, the fist token (row 0) can only attend to itself
        # The second token (row 1) can attend to itself and the first token
        mask = torch.tril(mask)
        # mask = 0 where x is padding id
        mask[x == self.padding_id] = 0

        # Add singleton dim to allow broadcasting with multihead attention matrix
        # mask: [batch, 1, tokens, tokens]
        mask = mask.unsqueeze(1)

        return mask.to(device)

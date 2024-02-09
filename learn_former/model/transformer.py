import torch

from learn_former.model.decoder import Decoder
from learn_former.model.encoder import Encoder
from learn_former.model.submodels import Embedding, Tokenizer, PositionalEncoding


class Transformer(torch.nn.Module):
    def __init__(self, num_layers: int = 6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers)
        self.decoder = Decoder(num_layers=num_layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x: [batch, tokens, 512]

        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(y, x)

        return x

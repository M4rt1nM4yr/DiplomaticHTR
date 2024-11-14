import os, sys
src_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(src_folder)

import torch
from torch import nn

from src.encoder.htr_transformer_encoder import HTRTransformerEncoder

class EncoderWrapper(nn.Module):
    def __init__(
            self,
            encoder,
            in_channels,
    ):
        super().__init__()
        if isinstance(encoder.func, type) and issubclass(encoder.func, HTRTransformerEncoder):
            self.out_dim = encoder.keywords["d_model"]
            self.encoder = encoder(
                in_channels=in_channels,
            )
        else:
            raise NotImplementedError

    def forward(
            self,
            x,
            lengths=None,
    ):
        """
        :param x: batch_size x seq_len x in_dim
        :param lengths: batch_size
        :return:
        """
        if isinstance(self.encoder, HTRTransformerEncoder):
            src_key_padding_mask = None
            if lengths is not None:
                src_key_padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
                for idx, w in enumerate(lengths):
                    src_key_padding_mask[idx, w:] = True
            return self.encoder(
                x=x,
                src_key_padding_mask=src_key_padding_mask,
            )
        else:
            raise NotImplementedError


if __name__ == "__main__":
    from functools import partial
    N = 3
    D = 64
    encoder = partial(
        HTRTransformerEncoder,
        d_model=D,
    )
    m = EncoderWrapper(encoder=encoder, in_channels=D)
    x = torch.randn(3, 15, D)
    lengths = torch.randint(1, 15, size=(3,))
    print(x.shape, lengths)
    print(m(x).shape)
import os, sys
src_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(src_folder)

import torch
from torch import nn

from src.modules.rezero_transformer import RZTXDecoderLayer, RZTXEncoderLayer
from src.utils.positional_encoding import Summer, PositionalEncoding1D
from src.modules.custom_transformer import TransformerEncoder


class HTRTransformerEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            d_model,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            pe_type='sinusoidal',
            dropout=0.1,
            activation='gelu',
            norm=True,
            rezero=True,
            channel_adapter_identity = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.channel_adapter = nn.Linear(in_channels, d_model) if not channel_adapter_identity else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        ) if not rezero else RZTXEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        encoder_layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_layer_norm if norm else None,
        )
        if pe_type=='sinusoidal':
            self.pe1d = Summer(PositionalEncoding1D(channels=d_model))
        elif pe_type=="none":
            self.pe1d = nn.Identity()
        else:
            raise NotImplementedError


    def forward(
            self,
            x,
            src_key_padding_mask=None,
    ):
        """

        :param x: batch_size x seq_len x d_model
        :param src_key_padding_mask: batch_size x seq_len
        :return:
        """
        x = self.pe1d(self.channel_adapter(x))
        x = self.encoder(
            src=x,
            src_key_padding_mask=src_key_padding_mask,
        )
        return x

if __name__ == "__main__":
    m = HTRTransformerEncoder(
        in_channels=128,
        d_model=256,
    )
    x = torch.randn(10, 64, 128)
    lengths = torch.randint(1, 64, (10,))
    src_key_padding_mask = torch.zeros(10, 64).bool()
    for idx, length in enumerate(lengths):
        src_key_padding_mask[idx, length:] = True
    y = m(x, src_key_padding_mask=src_key_padding_mask)
    print(y.shape)
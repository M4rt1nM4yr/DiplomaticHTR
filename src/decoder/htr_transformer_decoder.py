import os, sys
src_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(src_folder)

import einops
import torch
from torch import nn

from src.utils.positional_encoding import Summer, PositionalEncoding1D
from src.utils.constants import *
from src.modules.rezero_transformer import RZTXDecoderLayer
from src.modules.custom_transformer import TransformerDecoderLayer, TransformerDecoder

class HTRTransformerDecoder(nn.Module):
    def __init__(
            self,
            alphabet,
            d_model,
            memory_dim=None,
            nhead=8,
            num_layers=6,
            dim_feedforward=512,
            pe_type='sinusoidal',
            dropout=0.1,
            activation='gelu',
            norm=True,
            max_len=140,
            rezero=True,
            embed_y = True,
    ):
        super().__init__()
        memory_dim = d_model if memory_dim is None else memory_dim
        self.alphabet = alphabet
        self.max_len = max_len
        self.d_model = d_model
        self.embed_y = embed_y
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        ) if not rezero else RZTXDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        decoder_layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=decoder_layer_norm if norm else None,
        )
        if pe_type=='sinusoidal':
            self.pe1d = Summer(PositionalEncoding1D(channels=d_model))
        elif pe_type.lower()=='none'.lower():
            self.pe1d = nn.Identity()
        else:
            raise NotImplementedError
        self.embedder = nn.Embedding(len(alphabet.toPosition), d_model)
        self.predictor = nn.Linear(d_model, len(alphabet.toPosition))
        self.memory_adapter = nn.Linear(memory_dim, d_model) if memory_dim != d_model else nn.Identity()

    def forward(
            self,
            y,
            memory=None,
            memory_key_padding_mask=None,
            tgt_key_padding_mask=None,
            tgt_mask=None,
    ):
        """

        :param memory: batch_size x memory_seq_len x d_model
        :param y: batch_size x tgt_seq_len
        :param memory_key_padding_mask: batch_size x memory_seq_len
        :param tgt_key_padding_mask: batch_size x tgt_seq_len
        :param tgt_mask: tgt_seq_len x tgt_seq_len (?)
        :return: predictions: batch_size x tgt_seq_len x len(alphabet.toPosition)
        """
        y = self.pe1d(self.embedder(y)) if self.embed_y else y
        if memory is not None:
            memory = self.pe1d(self.memory_adapter(memory))
        else:
            memory = y
            memory_key_padding_mask = tgt_key_padding_mask

        decodings = self.decoder(
            tgt=y,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        predictions = self.predictor(decodings)
        return {
            "predictions": predictions,
        }

    @torch.no_grad()
    def inference(
            self,
            memory,
            memory_key_padding_mask=None,
            max_char_len=None,
    ):
        if max_char_len is None:
            max_char_len = self.max_len
        pred_logits = (torch.ones(size=(memory.shape[0], max_char_len+1)) * self.alphabet.toPosition[PAD]).long()
        pred = torch.ones(size=(memory.shape[0], max_char_len+1, len(self.alphabet.toPosition)))
        pred_logits[:, 0] = (
                torch.ones(size=pred_logits[:, 0].shape) * self.alphabet.toPosition[START_OF_SEQUENCE]).long()
        if memory.is_cuda:
            pred_logits = pred_logits.cuda()
            pred = pred.cuda()
        for i in range(1, max_char_len+1):
            dec_out = self(
                memory=memory,
                y=pred_logits[:, :i],
                memory_key_padding_mask=memory_key_padding_mask,
            )
            pred[:, i] = dec_out["predictions"][:, -1, :]
            pred_logits[:, i] = torch.argmax(dec_out["predictions"][:, -1, :], dim=-1)
        assert torch.all(pred[:,1:,:].argmax(dim=-1) == pred_logits[:,1:])
        return {
            "predictions": pred[:, 1:, :],
        }


if __name__ == "__main__":
    from src.utils.subsequent_mask import subsequent_mask
    from src.utils.alphabet import Alphabet
    alphabet = Alphabet('ab')
    m = HTRTransformerDecoder(
        alphabet=alphabet,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        pe_type='sinusoidal',
        dropout=0.1,
        activation='gelu',
        norm=True,
        max_len=100,
    ).cuda()
    N = 3
    memory = torch.randn(N, 15, 256).cuda()
    memory_widths = torch.randint(5, 15, size=(N,)).cuda()
    memory_key_padding_mask = torch.zeros(memory.shape[0], memory.shape[1]).int().cuda()
    for idx, mem_w in enumerate(memory_widths):
        memory_key_padding_mask[idx, int(mem_w):] = 1
    memory_key_padding_mask = memory_key_padding_mask.bool()
    y = torch.randint(0, len(alphabet.toPosition), size=(N, 25)).cuda()
    y[:,0] = alphabet.toPosition[START_OF_SEQUENCE]
    y_widths = torch.randint(5, 25, size=(N,)).cuda()
    tgt_key_padding_mask = torch.zeros(y.shape[0], y.shape[1]).int().cuda()
    for idx, w in enumerate(y_widths):
        tgt_key_padding_mask[idx, int(w):] = 1
        y[idx, int(w)] = alphabet.toPosition[END_OF_SEQUENCE]
        y[idx, int(w)+1:] = alphabet.toPosition[PAD]
    tgt_key_padding_mask = tgt_key_padding_mask.bool()
    if memory.is_cuda:
        tgt_key_padding_mask = tgt_key_padding_mask.cuda()
    tgt_mask = subsequent_mask(y.shape[1]).cuda()
    out = m(
        memory=memory,
        y=y,
        memory_key_padding_mask=memory_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        tgt_mask=tgt_mask,
    )
    print(out["predictions"].shape)
    out = m.inference(
        memory=memory,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    print(out["predictions"].shape)


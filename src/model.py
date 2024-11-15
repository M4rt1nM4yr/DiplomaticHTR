import os, sys
src_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(src_folder)

from functools import partial
from typing import Any, Optional

import einops
import torch
from torch import nn

from src.encoder.encoder_wrapper import EncoderWrapper
from src.utils.positional_encoding import Summer, PositionalEncoding1D

class S2SDipl(nn.Module):
    def __init__(
            self,
            alphabet,
            cnn,
            encoder,
            decoder,
            pe_connection='sinusoidal',
    ):
        super().__init__()
        self.alphabet = alphabet
        
        self.cnn = cnn
        self.encoder = EncoderWrapper(
            encoder,
            in_channels=cnn.out_dim*cnn.height_at_64px,   # because of fine-grained CNN
        )
        memory_dim = self.encoder.out_dim
        self.decoder = decoder(
            alphabet=alphabet,
            memory_dim=memory_dim,
        ) if isinstance(decoder, partial) else decoder
        if pe_connection.lower() == 'None'.lower():
            self.pe1d = nn.Identity()
        elif pe_connection.lower() == "sinusoidal".lower():
            self.pe1d = Summer(PositionalEncoding1D(channels=memory_dim))
        else:
            raise ValueError
        self.predictor_ctc = nn.Sequential(
            nn.Linear(self.encoder.out_dim, len(alphabet.toPosition)),
            nn.LogSoftmax(dim=-1)
        )

    def forward(
            self,
            x,
            y,
            img_width=None,
            tgt_key_padding_mask=None,
            tgt_mask=None,
    ):
        enc_out = self.encode(
            x,
            img_width,
        )
        memory = enc_out["memory"]
        memory = self.pe1d(memory)
        memory_key_padding_mask = torch.zeros((memory.shape[0], memory.shape[1]), dtype=torch.bool, device=memory.device)
        for idx, w in enumerate(enc_out["lengths"]):
            memory_key_padding_mask[idx, w:] = True
        dec_out = self.decoder(
            memory=memory,
            y=y,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        return {
            "log_predictions_ctc": enc_out["predictions"],
            "predictions_s2s": dec_out["predictions"],
            "lengths_ctc": enc_out["lengths"],
        }

    @torch.no_grad()
    def inference(
            self,
            x,
            img_width=None,
            max_char_len=None,
    ):
        if img_width==None:
            img_width = torch.ones((x.shape[0]))*x.shape[-1]

        enc_out = self.encode(
            x,
            img_width,
        )
        memory = enc_out["memory"]
        memory = self.pe1d(memory)
        memory_key_padding_mask = torch.zeros((memory.shape[0], memory.shape[1]), dtype=torch.bool,
                                              device=memory.device)
        for idx, w in enumerate(enc_out["lengths"]):
            memory_key_padding_mask[idx, w:] = True
        dec_out = self.decoder.inference(
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            max_char_len=max_char_len,
        )
        return {
            "log_predictions_ctc": enc_out["predictions"],
            "predictions_s2s": dec_out["predictions"],
            "lengths_ctc": enc_out["lengths"],
        }



    def encode(
            self,
            x,
            img_width=None,
    ):
        f = self.cnn(x)
        # f = torch.mean(f, dim=-2)
        f = einops.rearrange(f, 'b c h w -> b w (c h)')
        if img_width is not None:
            lengths = torch.ceil(img_width / torch.max(img_width) * f.shape[1]).cpu().int()
        else:
            lengths = None
        out = self.encoder(
            x=f,
            lengths=lengths,
        )
        predictions = self.predictor_ctc(out)
        return {
            "features": f,
            "memory": out,
            "predictions": predictions,
            "lengths": lengths,
        }

if __name__ == "__main__":
    from functools import partial
    from src.cnn.resnet import Resnet
    from src.encoder.htr_transformer_encoder import HTRTransformerEncoder
    from src.decoder.htr_transformer_decoder import HTRTransformerDecoder
    from src.utils.alphabet import Alphabet
    from src.utils.constants import *
    from src.utils.subsequent_mask import subsequent_mask
    
    rezero = True
    alphabet = Alphabet('ab')
    cnn = Resnet(in_channels=1, wider=True)
    encoder = partial(
        HTRTransformerEncoder,
        d_model=256,
        rezero=rezero,
    )
    decoder = partial(
        HTRTransformerDecoder,
        d_model=256,
        rezero=rezero,
    )
    m = S2SDipl(
        alphabet=Alphabet('ab'),
        cnn=cnn,
        encoder=encoder,
        decoder=decoder,
    ).cuda()
    
    N = 3
    x = torch.randn(N, 1, 64, 1024).cuda()
    x_widths = torch.randint(64, 1024, size=(N,)).cuda()
    y = torch.randint(0, len(alphabet.toPosition), size=(N, 25)).cuda()
    y[:,0] = alphabet.toPosition[START_OF_SEQUENCE]
    y_widths = torch.randint(5, 25, size=(N,)).cuda()
    tgt_key_padding_mask = torch.zeros(y.shape[0], y.shape[1]).int().cuda()
    for idx, w in enumerate(y_widths):
        tgt_key_padding_mask[idx, int(w):] = 1
        y[idx, int(w)] = alphabet.toPosition[END_OF_SEQUENCE]
        y[idx, int(w)+1:] = alphabet.toPosition[PAD]
    tgt_key_padding_mask = tgt_key_padding_mask.bool()
    if x.is_cuda:
        tgt_key_padding_mask = tgt_key_padding_mask.cuda()
    tgt_mask = subsequent_mask(y.shape[1]).cuda()
    
    # Check training
    out = m(
        x=x,
        y=y,
        img_width=x_widths,
        tgt_key_padding_mask=tgt_key_padding_mask,
        tgt_mask=tgt_mask,
    )
    print(out["predictions_s2s"].shape)
    
    # Check inference
    out = m.inference(
        x=x,
        img_width=x_widths,
    )
    print(out["predictions_s2s"].shape)    
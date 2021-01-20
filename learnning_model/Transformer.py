from torch import Tensor
from typing import Optional
from torch.nn import init

import torch
import torch.nn as nn
import torch.optim as optim
import math
MAX_SEQ_LEN = 75

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=75):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=TEXT.vocab.vectors, freeze=True)

    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec.to(device)


class Transformer(nn.Module):
    def __init__(self, d_model: int = 300, nhead: int = 6, num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1,
                 activation: str = "relu", target_vocab_length: int = 0, TEXT=None) -> None:
        super(Transformer, self).__init__()

        self.source_embedding = nn.Embedding(target_vocab_length, 300)
        #self.source_embedding = Embedder().cuda()
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=MAX_SEQ_LEN)
        self.pos_decoder = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=MAX_SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        self.target_embedding = nn.Embedding(target_vocab_length, 300)
        #self.target_embedding = Embedder().cuda()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm)
        self.out = nn.Linear(d_model, target_vocab_length)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

        self.src_key_padding_mask = None
        self.tgt_key_padding_mask = None

        self.TEXT = TEXT

    def _generate_square_subsequent_mask(self, src):
        input_pad = self.TEXT.stoi['<pad>']
        mask = (src == input_pad)
        return mask

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if self.src_key_padding_mask is None or self.src_key_padding_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(src).to(device)
            self.src_key_padding_mask = mask.transpose(0, 1)

        if self.tgt_key_padding_mask is None or self.tgt_key_padding_mask.size(0) != len(src):
            device = tgt.device
            mask = self._generate_square_subsequent_mask(tgt).to(device)
            self.tgt_key_padding_mask = mask.transpose(0, 1)

        src = self.source_embedding(src)
        src = self.pos_encoder(src)

        tgt = self.target_embedding(tgt)
        tgt = self.pos_decoder(tgt)

        # , src_key_padding_mask=self.src_key_padding_mask)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              memory_key_padding_mask=memory_key_padding_mask)  # ,tgt_key_padding_mask=self.tgt_key_padding_mask)
        output = self.out(output)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

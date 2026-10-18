"""Ink-to-LaTeX recognizer, to judge whether a student's ink says what it was
asked to write (and, later, to give it a loss for that).

- the ink goes in as the flow student generates it (scripts/flow_model.py):
  absolute (x, y), centred and in units of the ink's height, plus a +-1 pen
  channel. Generated ink, even a partly denoised one, can then be read with
  no conversion and no hard pen threshold, and stays differentiable;
- a strided conv stem cuts the points 4x, then a transformer encoder;
- an autoregressive transformer decoder writes the tokens. Not CTC alone:
  math is not written in reading order (fraction bars first, scripts and
  closing brackets added later), and CTC assumes it is. A CTC head on the
  encoder is kept as an auxiliary loss, which helps the alignment early on.

A flow student's ink x reaches it as
torch.cat([x[..., :2] * flow.coord_std, x[..., 2:]], dim=-1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_model import MIN_HEIGHT, sinusoid
from ink_corpus import MAX_TEXT, PAD, SEP

STRIDE = 4  # two stride-2 convolutions


def ink_positions(offsets, ink_len, mu, std):
    """Normalized corpus offsets (B, N, 3) -> (B, N, 3): (x, y) centred and in
    units of the ink's height, and a +-1 pen channel, zero on padding."""
    mask = (torch.arange(offsets.shape[1], device=offsets.device) < ink_len.unsqueeze(1)).unsqueeze(-1)
    xy = torch.cumsum((offsets[..., :2] * std[:2] + mu[:2]) * mask, dim=1)
    centre = (xy * mask).sum(dim=1, keepdim=True) / ink_len.view(-1, 1, 1)
    y = xy[..., 1:]
    height = y.masked_fill(~mask, -torch.inf).amax(dim=1) - y.masked_fill(~mask, torch.inf).amin(dim=1)
    xy = (xy - centre) / height.clamp(min=MIN_HEIGHT).unsqueeze(1)
    return torch.cat([xy, offsets[..., 2:] * 2 - 1], dim=-1) * mask


def right_padded(text, text_len):
    """Left-padded tokens (B, T), as ink_corpus.encode_texts makes them -> right-padded."""
    T = text.shape[1]
    index = (torch.arange(T, device=text.device) + (T - text_len).unsqueeze(1)) % T
    return text.gather(1, index)


class Recognizer(nn.Module):
    def __init__(self, vocab_size, coord_std, d_model=256, heads=4, encoder_layers=6, decoder_layers=3):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, coord_std=list(coord_std), d_model=d_model, heads=heads,
                           encoder_layers=encoder_layers, decoder_layers=decoder_layers)
        self.d_model = d_model
        self.register_buffer("coord_std", torch.tensor(coord_std, dtype=torch.float32))

        # x, y, pen, dx, dy
        self.stem = nn.Sequential(nn.Conv1d(5, d_model, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(d_model, d_model, 5, stride=2, padding=2), nn.GELU())

        def encoder_layer():
            return nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=0.1,
                                              activation="gelu", batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer(), encoder_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.ctc_head = nn.Linear(d_model, vocab_size)  # PAD is the CTC blank

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.token_pos = nn.Embedding(MAX_TEXT + 1, d_model)
        layer = nn.TransformerDecoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocab_size)

    def encode(self, ink, ink_len):
        """ink (B, N, 3) as ink_positions makes it. Returns the memory
        (B, M, d), its padding mask and its lengths, M = ceil(N / STRIDE)."""
        N = ink.shape[1]
        mask = (torch.arange(N, device=ink.device) < ink_len.unsqueeze(1)).unsqueeze(-1)
        xy = ink[..., :2] / self.coord_std
        delta = torch.diff(xy, dim=1, prepend=xy[:, :1]) * mask
        features = torch.cat([xy, ink[..., 2:], delta], dim=-1) * mask
        h = self.stem(features.transpose(1, 2)).transpose(1, 2)
        memory_len = ink_len
        for _ in range(2):
            memory_len = (memory_len + 1) // 2
        index = torch.arange(h.shape[1], device=ink.device)
        pad = index.unsqueeze(0) >= memory_len.unsqueeze(1)
        h = h + sinusoid(index, self.d_model).to(h.dtype)
        return self.encoder(h, src_key_padding_mask=pad), pad, memory_len

    def decode(self, tokens, memory, memory_pad):
        """tokens (B, L) right-padded, starting with SEP. Returns logits (B, L, vocab)."""
        L = tokens.shape[1]
        h = self.token_embed(tokens) + self.token_pos(torch.arange(L, device=tokens.device))
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device, dtype=torch.bool)
        h = self.decoder(h, memory, tgt_mask=causal, tgt_is_causal=True, tgt_key_padding_mask=tokens == PAD,
                         memory_key_padding_mask=memory_pad)
        return self.out(h).float()

    def loss(self, ink, ink_len, text, text_len):
        """Token cross-entropy (with SEP as end of text), the CTC loss and
        teacher-forced token accuracy. text is left-padded."""
        B = text.shape[0]
        target = right_padded(text, text_len)
        target = torch.cat([target, torch.full((B, 1), PAD, dtype=target.dtype, device=target.device)], dim=1)
        target[torch.arange(B), text_len] = SEP
        start = torch.full((B, 1), SEP, dtype=target.dtype, device=target.device)
        tokens = torch.cat([start, target[:, :-1]], dim=1)

        memory, memory_pad, memory_len = self.encode(ink, ink_len)
        logits = self.decode(tokens, memory, memory_pad)
        ce = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=PAD)
        known = target != PAD
        acc = ((logits.argmax(-1) == target) & known).sum() / known.sum()

        log_probs = F.log_softmax(self.ctc_head(memory).float(), dim=-1).transpose(0, 1)
        ctc = F.ctc_loss(log_probs, right_padded(text, text_len), memory_len, text_len, blank=PAD,
                         zero_infinity=True)
        return {"ce": ce, "ctc": ctc, "token_acc": acc}

    @torch.no_grad()
    def read(self, ink, ink_len):
        """Greedy decoding: per ink, the list of token ids it reads."""
        B = ink.shape[0]
        memory, memory_pad, _ = self.encode(ink, ink_len)
        tokens = torch.full((B, 1), SEP, dtype=torch.long, device=ink.device)
        done = torch.zeros(B, dtype=torch.bool, device=ink.device)
        for _ in range(MAX_TEXT + 1):
            nxt = self.decode(tokens, memory, memory_pad)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            done |= nxt == SEP
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            if done.all():
                break
        out = []
        for row in tokens[:, 1:].tolist():
            ids = []
            for t in row:
                if t in (SEP, PAD):
                    break
                ids.append(t)
            out.append(ids)
        return out

"""Flow-matching text-to-ink student.

Unlike the Mamba student (scripts/student_model.py), which writes one point at
a time, this model generates a whole line at once:

- the text is encoded by a bidirectional transformer over [c1 .. cU, SEP];
- a head on SEP predicts how many points the ink has (in bins of LENGTH_BIN),
  which replaces the Mamba student's "finished" class;
- a transformer over the points, with cross-attention to the text in every
  block, predicts the velocity that carries Gaussian noise to the ink along a
  straight path (x_t = (1 - t) noise + t ink, velocity = ink - noise).

The ink is generated as absolute positions, not offsets: (x, y) centred per
ink and divided by the ink's height (its highest minus its lowest y), plus a
pen channel that is +1 on the last point of a stroke and -1 elsewhere. White
noise on positions leaves the layout (where a denominator or superscript
sits) visible longest; white noise on offsets would be a random walk that
destroys it first.

Every ink is therefore one unit tall, whatever it says: the model does not
decide how large to write. Whoever places a sample chooses the height of its
box. coord_std is a corpus-wide constant that only brings the coordinates to
the unit variance of the noise.

A second head labels every point with the text token it draws. It is trained
where the corpus has an alignment (chars >= 0) and read from one extra pass
over the finished ink.

sample() returns what Student.sample() returns, normalized offsets included,
so scripts/sample_student.py and the page renderer work with either model.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ink_corpus import MAX_TEXT, PAD, SEP

LENGTH_BIN = 8      # points per length class
LENGTH_BINS = 300   # up to 2400 points; both corpora stop below that
# The Graves bias has no exact counterpart here. It lowers the temperature of
# the starting noise instead, which also gives neater, more average writing.
BIAS_TEMPERATURE = 0.25
MIN_HEIGHT = 1e-6   # denormalized offset units; keeps a perfectly flat ink finite


def sinusoid(x, dim):
    """x (...) -> (..., dim) sine and cosine features."""
    freqs = torch.exp(-math.log(10000.0) * torch.arange(dim // 2, device=x.device) / (dim // 2))
    angles = x.unsqueeze(-1).float() * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def ink_statistics(corpus, lines=4096):
    """Over the corpus's first `lines` inks: the per-axis std of centred
    positions measured in ink heights, and the median ink height in
    denormalized offset units."""
    mu, std = corpus.meta["mu"][:2], corpus.meta["std"][:2]
    scaled, heights = [], []
    for i in range(min(lines, len(corpus))):
        s, k = corpus.starts[i], corpus.lengths[i]
        xy = (corpus.offsets[s:s + k, :2] * std + mu).cumsum(axis=0)
        height = max(xy[:, 1].max() - xy[:, 1].min(), MIN_HEIGHT)
        scaled.append((xy - xy.mean(axis=0)) / height)
        heights.append(height)
    return np.concatenate(scaled).std(axis=0).tolist(), float(np.median(heights))


class TimeNorm(nn.Module):
    """LayerNorm whose scale and shift come from the flow time."""

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, time):
        scale, shift = self.modulation(time).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class Block(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.self_norm = TimeNorm(d_model)
        self.text_norm = TimeNorm(d_model)
        self.mlp_norm = TimeNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.text_attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, time, point_pad, memory, memory_pad):
        h = self.self_norm(x, time)
        x = x + self.self_attention(h, h, h, key_padding_mask=point_pad, need_weights=False)[0]
        h = self.text_norm(x, time)
        x = x + self.text_attention(h, memory, memory, key_padding_mask=memory_pad, need_weights=False)[0]
        return x + self.mlp(self.mlp_norm(x, time))


class FlowStudent(nn.Module):
    def __init__(self, vocab_size, offset_mu, offset_std, coord_std, typical_height, d_model=384, n_layers=8,
                 heads=6, text_layers=3, text_dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, offset_mu=list(offset_mu), offset_std=list(offset_std),
                           coord_std=list(coord_std), typical_height=typical_height, d_model=d_model,
                           n_layers=n_layers, heads=heads, text_layers=text_layers, text_dropout=text_dropout)
        self.typical_height = typical_height
        self.d_model = d_model
        self.text_dropout = text_dropout
        self.register_buffer("offset_mu", torch.tensor(offset_mu[:2], dtype=torch.float32))
        self.register_buffer("offset_std", torch.tensor(offset_std[:2], dtype=torch.float32))
        self.register_buffer("coord_std", torch.tensor(coord_std, dtype=torch.float32))

        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.text_pos = nn.Embedding(MAX_TEXT + 1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.text_encoder = nn.TransformerEncoder(layer, text_layers, norm=nn.LayerNorm(d_model),
                                                  enable_nested_tensor=False)
        self.null_text = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.length_head = nn.Linear(d_model, LENGTH_BINS)

        self.point_in = nn.Linear(3, d_model)
        self.position_in = nn.Linear(2 * d_model, d_model)
        self.time_in = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(Block(d_model, heads) for _ in range(n_layers))
        self.out_norm = TimeNorm(d_model)
        self.velocity = nn.Linear(d_model, 3)
        nn.init.zeros_(self.velocity.weight)
        nn.init.zeros_(self.velocity.bias)
        self.char_head = nn.Linear(d_model, MAX_TEXT)

    def to_positions(self, offsets, ink_len):
        """Normalized offsets (B, N, 3) -> the flow's ink (B, N, 3): absolute
        (x, y), centred and in units of the ink's own height, and a +-1 pen
        channel, zero on padding."""
        mask = (torch.arange(offsets.shape[1], device=offsets.device) < ink_len.unsqueeze(1)).unsqueeze(-1)
        xy = torch.cumsum((offsets[..., :2] * self.offset_std + self.offset_mu) * mask, dim=1)
        centre = (xy * mask).sum(dim=1, keepdim=True) / ink_len.view(-1, 1, 1)
        y = xy[..., 1:]
        height = y.masked_fill(~mask, -torch.inf).amax(dim=1) - y.masked_fill(~mask, torch.inf).amin(dim=1)
        xy = (xy - centre) / height.clamp(min=MIN_HEIGHT).unsqueeze(1) / self.coord_std
        return torch.cat([xy, offsets[..., 2:] * 2 - 1], dim=-1) * mask

    def to_offsets(self, x):
        """The inverse of to_positions, up to where the ink starts and how
        tall it is: every ink comes back at the corpus's typical height."""
        xy = x[..., :2] * self.coord_std * self.typical_height
        raw = torch.diff(xy, dim=1, prepend=xy[:, :1])
        return torch.cat([(raw - self.offset_mu) / self.offset_std, (x[..., 2:] > 0).float()], dim=-1)

    def encode_text(self, text, text_len, drop=None):
        """text: (B, T) left-padded. Returns the text memory (B, T + 1, d) with
        SEP last, its padding mask, and the length logits. Rows where drop is
        set get the unconditional memory used for classifier-free guidance."""
        B, T = text.shape
        sep = torch.full((B, 1), SEP, dtype=text.dtype, device=text.device)
        tokens = torch.cat([text, sep], dim=1)
        pos = torch.arange(T + 1, device=text.device) - (T - text_len).unsqueeze(1)
        pad = pos < 0
        embedded = self.char_embed(tokens) + self.text_pos(pos.clamp(0, MAX_TEXT))
        memory = self.text_encoder(embedded, src_key_padding_mask=pad)
        length_logit = self.length_head(memory[:, -1]).float()
        if drop is not None:
            memory = torch.where(drop.view(B, 1, 1), self.null_text.to(memory.dtype), memory)
            # Only the SEP position stays visible, so no row is fully masked.
            pad = pad | (drop.unsqueeze(1) & (pos < text_len.unsqueeze(1)))
        return memory, pad, length_logit

    def denoise(self, x, t, ink_len, memory, memory_pad):
        """Velocity (B, N, 3) and token-index logits (B, N, MAX_TEXT) for the
        noisy ink x at flow times t (B,)."""
        B, N, _ = x.shape
        index = torch.arange(N, device=x.device)
        point_pad = index.unsqueeze(0) >= ink_len.unsqueeze(1)
        # The fraction of the ink already drawn tells a point roughly which
        # part of the text it belongs to, whatever the ink's length.
        fraction = index.unsqueeze(0) / ink_len.unsqueeze(1)
        position = torch.cat([sinusoid(index, self.d_model).expand(B, -1, -1),
                              sinusoid(fraction * 1000, self.d_model)], dim=-1)
        h = self.point_in(x) + self.position_in(position)
        time = self.time_in(sinusoid(t * 1000, self.d_model))
        for block in self.blocks:
            h = block(h, time, point_pad, memory, memory_pad)
        h = self.out_norm(h, time)
        return self.velocity(h).float(), self.char_head(h).float()

    def loss(self, batch, generator=None):
        """A dict of losses, each a mean over real (unpadded) positions:
        flow (velocity error), smooth (error in the velocity's change between
        neighbouring points), length, char_ce and char_acc."""
        text, text_len, offsets, chars, ink_len = (batch[k] for k in ("text", "text_len", "offsets", "chars", "ink_len"))
        B, N, _ = offsets.shape
        device = offsets.device
        mask = torch.arange(N, device=device).unsqueeze(0) < ink_len.unsqueeze(1)
        x1 = self.to_positions(offsets, ink_len)
        x0 = torch.randn(x1.shape, generator=generator, device=device)
        # Logit-normal times: most steps train mid-path, where layout is decided.
        t = torch.sigmoid(torch.randn(B, generator=generator, device=device))
        xt = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
        drop = torch.rand(B, generator=generator, device=device) < self.text_dropout

        memory, memory_pad, length_logit = self.encode_text(text, text_len, drop)
        velocity, char_logit = self.denoise(xt, t, ink_len, memory, memory_pad)

        error = (velocity - (x1 - x0)) * mask.unsqueeze(-1)
        flow = error.pow(2).sum() / (3 * mask.sum())
        # Points are a tenth of a symbol apart, so small independent errors
        # show up as jitter. Penalizing the error's change between neighbours
        # weights the same regression towards getting fine detail right.
        pairs = mask[:, 1:]
        step = torch.diff(error[..., :2], dim=1) * pairs.unsqueeze(-1)
        smooth = step.pow(2).sum() / (2 * pairs.sum().clamp(min=1))

        length_bin = ((ink_len - 1) // LENGTH_BIN).clamp(0, LENGTH_BINS - 1)
        length = F.cross_entropy(length_logit, length_bin)

        # Labels cannot be read off ink that is still mostly noise, so nearly
        # finished ink (t close to 1, as at sampling) counts the most.
        known = mask & (chars >= 0)
        char_nll = F.cross_entropy(char_logit.transpose(1, 2), chars.clamp(min=0), reduction="none")
        weight = known * t.view(B, 1)
        char_ce = (char_nll * weight).sum() / weight.sum().clamp(min=1e-6)
        char_acc = ((char_logit.argmax(-1) == chars) & known).sum() / known.sum().clamp(min=1)
        return {"flow": flow, "smooth": smooth, "length": length, "char_ce": char_ce, "char_acc": char_acc}

    @torch.no_grad()
    def sample(self, text, text_len, bias=0.0, steps=32, guidance=2.0, generator=None):
        """Write a batch of lines. text: (B, T) left-padded tokens.

        Returns per line (offsets (n, 3), chars (n,), finished), as
        Student.sample() does. finished is always True: the length is drawn
        before the ink. guidance is the classifier-free guidance scale (1
        turns it off).
        """
        B, device = text.shape[0], text.device
        bias = torch.as_tensor(bias, dtype=torch.float32, device=device).reshape(-1).expand(B)
        memory, memory_pad, length_logit = self.encode_text(text, text_len)
        length_p = F.softmax(length_logit * (1 + bias).unsqueeze(1), dim=-1)
        ink_len = torch.multinomial(length_p, 1, generator=generator).squeeze(1) * LENGTH_BIN + LENGTH_BIN // 2

        x = torch.randn((B, int(ink_len.max()), 3), generator=generator, device=device)
        x = x * torch.exp(-BIAS_TEMPERATURE * bias).view(B, 1, 1)
        if guidance != 1:
            everything = torch.ones(B, dtype=torch.bool, device=device)
            null_memory, null_pad, _ = self.encode_text(text, text_len, everything)
        for n in range(steps):
            t = torch.full((B,), n / steps, device=device)
            velocity, _ = self.denoise(x, t, ink_len, memory, memory_pad)
            if guidance != 1:
                unconditional, _ = self.denoise(x, t, ink_len, null_memory, null_pad)
                velocity = unconditional + guidance * (velocity - unconditional)
            x = x + velocity / steps

        _, char_logit = self.denoise(x, torch.ones(B, device=device), ink_len, memory, memory_pad)
        valid = torch.arange(MAX_TEXT, device=device).unsqueeze(0) < text_len.unsqueeze(1)
        labels = char_logit.masked_fill(~valid.unsqueeze(1), -torch.inf).argmax(-1).cpu()
        offsets = self.to_offsets(x).cpu()
        out = []
        for b in range(B):
            n = int(ink_len[b])
            off = offsets[b, :n].clone()
            off[-1, 2] = 1.0
            out.append((off, labels[b, :n], True))
        return out

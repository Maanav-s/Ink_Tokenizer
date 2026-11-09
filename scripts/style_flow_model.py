"""Spline flow student conditioned on a writer's style, a VAE-style latent
inferred from other ink by the same writer (an experiment on top of
scripts/spline_flow_model.py).

A style encoder reads a set of reference inks and gives the mean and log
variance of a latent z; z is added, through a linear map, to the flow time
embedding that modulates every block, so it can change how every point is
written but not what the text is. The references are never the line being
written: for a line whose writer is known (corpus `writers`, "source/id")
they are other lines by that writer, so z can only carry what those lines
share with this one, which is style rather than content. A line with no
known writer gets a short crop of itself, which leaks some content, so the
KL term and z's small size must keep that crop from being copied.

Losses on top of SplineFlowStudent's:
- kl: KL(q(z | references) || N(0, I)), summed over z's dimensions: the
  information z carries, in nats. The trainer (scripts/train_style_flow.py)
  holds it near a target by adapting its weight, rather than fixing one.
- contrastive: supervised contrastive loss for lines with a known writer, on
  the normalized output of a small projection head over z's mean when
  projection_dim > 0 (as in SimCLR/SupCon: the head, not z, collapses each
  writer's lines together, so z keeps how one writer's lines differ, which
  the flow can use), else on the means themselves. Every such line is also encoded on its own, so each has
  a positive (its references' mean); negatives are only lines from the same
  source, as telling corpora apart (different digitizers, sampling rates)
  would be learned instead of telling writers apart.

style_in and null_style start at zero, so a checkpoint without a style
encoder (init_from_plain) starts out writing exactly as it did. A fraction
style_dropout of lines train with null_style instead of z, which is what the
model writes with when no style is given (sample() without cond).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_model import sinusoid
from ink_corpus import MAX_TEXT
from spline_flow_model import SplineFlowStudent


class StyleEncoder(nn.Module):
    def __init__(self, z_dim, d_model=256, layers=4, heads=4):
        super().__init__()
        self.d_model = d_model
        self.point_in = nn.Linear(5, d_model)  # x, y, pen, dx, dy
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        self.pool = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.out = nn.Linear(d_model, 2 * z_dim)

    def forward(self, ink, ink_len, owner, n_sets):
        """ink (R, N, 3) in the flow's form (FlowStudent.to_positions), one row
        per reference; owner (R,) the set each reference belongs to. Returns
        the mean and log variance of z, each (n_sets, z_dim), from the average
        of each set's pooled references."""
        R, N, _ = ink.shape
        index = torch.arange(N, device=ink.device)
        pad = index.unsqueeze(0) >= ink_len.unsqueeze(1)
        delta = torch.diff(ink[..., :2], dim=1, prepend=ink[:, :1, :2]) * ~pad.unsqueeze(-1)
        h = self.point_in(torch.cat([ink, delta], dim=-1)) + sinusoid(index, self.d_model).to(ink.dtype)
        h = self.encoder(h, src_key_padding_mask=pad)
        pooled = self.pool(self.query.expand(R, -1, -1).to(h.dtype), h, h, key_padding_mask=pad,
                           need_weights=False)[0][:, 0].float()
        sums = torch.zeros(n_sets, self.d_model, device=ink.device).index_add_(0, owner, pooled)
        counts = torch.bincount(owner, minlength=n_sets).clamp(min=1).unsqueeze(1)
        mu, logvar = self.out(sums / counts).float().chunk(2, dim=-1)
        return mu, logvar.clamp(-10, 10)


def supervised_contrastive(embedding, writer, source, temperature):
    """Supervised contrastive loss (Khosla et al. 2020) over embeddings (M, D):
    rows with the same writer id are positives, rows of another source are
    left out of the denominator. Rows without a positive do not count."""
    e = F.normalize(embedding, dim=-1)
    sim = e @ e.T / temperature
    eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
    positive = (writer.unsqueeze(0) == writer.unsqueeze(1)) & ~eye
    allowed = (source.unsqueeze(0) == source.unsqueeze(1)) & ~eye
    log_p = sim - torch.logsumexp(sim.masked_fill(~allowed, -torch.inf), dim=1, keepdim=True)
    has_positive = positive.any(dim=1)
    if not has_positive.any():
        return embedding.sum() * 0
    per_row = (log_p * positive).sum(dim=1) / positive.sum(dim=1).clamp(min=1)
    return -per_row[has_positive].mean()


class StyleSplineFlowStudent(SplineFlowStudent):
    def __init__(self, z_dim=32, style_d_model=256, style_layers=4, style_dropout=0.1, temperature=0.1,
                 projection_dim=0, **config):
        super().__init__(**config)
        self.config.update(z_dim=z_dim, style_d_model=style_d_model, style_layers=style_layers,
                           style_dropout=style_dropout, temperature=temperature, projection_dim=projection_dim)
        self.style_dropout = style_dropout
        self.temperature = temperature
        self.projection = (nn.Sequential(nn.Linear(z_dim, 4 * projection_dim), nn.GELU(),
                                         nn.Linear(4 * projection_dim, projection_dim))
                           if projection_dim else nn.Identity())
        self.style_encoder = StyleEncoder(z_dim, style_d_model, style_layers)
        self.style_in = nn.Linear(z_dim, self.d_model)
        nn.init.zeros_(self.style_in.weight)
        nn.init.zeros_(self.style_in.bias)
        self.null_style = nn.Parameter(torch.zeros(self.d_model))

    def init_from_plain(self, state):
        """Load a SplineFlowStudent state dict: every weight it has, the
        first rows of the character embedding if the vocabulary has grown
        (pool_corpus.py appends tokens). Returns the names left as they were."""
        own = self.state_dict()
        loaded = []
        for name, value in state.items():
            if name not in own:
                continue
            if own[name].shape == value.shape:
                own[name] = value
            elif name == "char_embed.weight" and value.shape[1] == own[name].shape[1]:
                own[name][:len(value)] = value
            else:
                continue
            loaded.append(name)
        self.load_state_dict(own)
        return sorted(set(own) - set(loaded))

    def encode_style(self, offsets, ink_len, owner, n_sets):
        """Corpus offsets of reference inks -> mean and log variance of z."""
        return self.style_encoder(self.to_positions(offsets, ink_len), ink_len, owner, n_sets)

    def embed_lines(self, offsets, ink_len):
        """The mean of z for each ink on its own (B, z_dim)."""
        return self.encode_style(offsets, ink_len, torch.arange(len(ink_len), device=offsets.device),
                                 len(ink_len))[0]

    def identity(self, offsets, ink_len):
        """Each ink's writer embedding (B, D), on its own: the space the
        contrastive loss shapes, for telling writers apart."""
        return self.projection(self.embed_lines(offsets, ink_len))

    def style_cond(self, z):
        """z (B, z_dim) -> what denoise() adds to the time embedding."""
        return self.style_in(z)

    @torch.no_grad()
    def transport(self, x, text, text_len, ink_len, steps=None, guidance=None, cond=None, style_guidance=1.0):
        """FlowStudent.transport(), plus guidance on the style: with cond given
        and style_guidance g != 1, the velocity is pushed g - 1 times further
        from the no-style velocity (null_style), as text guidance is pushed
        from the no-text one; both guidances add up."""
        if cond is None or style_guidance == 1:
            return super().transport(x, text, text_len, ink_len, steps, guidance, cond)
        steps = self.sample_steps if steps is None else steps
        guidance = self.sample_guidance if guidance is None else guidance
        B, device = x.shape[0], x.device
        memory, memory_pad, _ = self.encode_text(text, text_len)
        if guidance != 1:
            everything = torch.ones(B, dtype=torch.bool, device=device)
            null_memory, null_pad, _ = self.encode_text(text, text_len, everything)
        for n in range(steps):
            t = torch.full((B,), n / steps, device=device)
            velocity, _ = self.denoise(x, t, ink_len, memory, memory_pad, cond)
            no_style, _ = self.denoise(x, t, ink_len, memory, memory_pad)
            update = velocity + (style_guidance - 1) * (velocity - no_style)
            if guidance != 1:
                no_text, _ = self.denoise(x, t, ink_len, null_memory, null_pad, cond)
                update = update + (guidance - 1) * (velocity - no_text)
            x = x + update / steps

        _, char_logit = self.denoise(x, torch.ones(B, device=device), ink_len, memory, memory_pad, cond)
        valid = torch.arange(MAX_TEXT, device=device).unsqueeze(0) < text_len.unsqueeze(1)
        return x, char_logit.masked_fill(~valid.unsqueeze(1), -torch.inf).argmax(-1)

    def denoise(self, x, t, ink_len, memory, memory_pad, cond=None):
        if cond is None:
            cond = self.null_style.expand(x.shape[0], -1)
        return super().denoise(x, t, ink_len, memory, memory_pad, cond)

    def loss(self, batch, generator=None, t_mean=0.0):
        """SplineFlowStudent's losses, plus kl (summed over z), contrastive and, when
        evaluating (generator given), flow_null: the flow loss with the same
        noise and times but null_style, so flow_null - flow is what the style
        is worth. Evaluation uses z's mean and never drops the style. t_mean
        is as in FlowStudent.loss()."""
        B, device = batch["text"].shape[0], batch["text"].device
        mu, logvar = self.encode_style(batch["ref_offsets"], batch["ref_len"], batch["ref_owner"], B)
        if self.training:
            cond = self.style_cond(mu + torch.randn_like(mu) * torch.exp(0.5 * logvar))
            drop = torch.rand(B, device=device) < self.style_dropout
            cond = torch.where(drop.unsqueeze(1), self.null_style.to(cond.dtype), cond)
        else:
            cond = self.style_cond(mu)

        state = generator.get_state() if generator is not None else None
        losses = super().loss(batch, generator, cond=cond, t_mean=t_mean)
        if generator is not None:
            generator.set_state(state)
            losses["flow_null"] = super().loss(batch, generator, cond=self.null_style.expand(B, -1),
                                               t_mean=t_mean)["flow"]

        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        losses["kl"] = kl.sum(dim=1).mean()

        labeled = batch["writer"] >= 0
        if labeled.sum() > 0:
            anchor = self.identity(batch["anchor_offsets"], batch["anchor_len"])
            losses["contrastive"] = supervised_contrastive(
                torch.cat([self.projection(mu[labeled]), anchor]), batch["writer"][labeled].repeat(2),
                batch["source"][labeled].repeat(2), self.temperature)
        else:
            losses["contrastive"] = mu.sum() * 0
        return losses

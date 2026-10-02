"""Mamba text-to-ink student.

One causal sequence per line: the text, left-padded, then a separator, then
the ink:

    [PAD .. PAD, c1 .. cU, SEP, p1 .. pN]

There is no attention window over the text: the Mamba state has to remember
the text and keep its own place in it. From the SEP position onwards, each
position predicts:

- the next point, as the teacher does: a mixture of bivariate Gaussians for
  (dx, dy) and a Bernoulli for end of stroke, in the teacher's normalized
  offset space;
- which character the next point draws, as an index 0..U-1 into the text,
  or U for "the line is finished". This head gives every sampled point a
  character label, which the page renderer needs, and decides when to stop.
  It is trained on the teacher's attention alignment.

Text characters get a position embedding so the index head can count.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

from ink_corpus import MAX_TEXT, PAD, SEP


class Block(nn.Module):
    def __init__(self, d_model, layer_idx, d_state, headdim):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mixer = Mamba2(d_model, d_state=d_state, headdim=headdim, layer_idx=layer_idx)

    def forward(self, x, inference_params=None):
        return x + self.mixer(self.norm(x), inference_params=inference_params)


class Student(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, d_state=64, headdim=32, mixtures=20):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_state=d_state,
                           headdim=headdim, mixtures=mixtures)
        self.mixtures = mixtures
        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.text_pos = nn.Embedding(MAX_TEXT + 1, d_model)
        self.point_in = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList(Block(d_model, i, d_state, headdim) for i in range(n_layers))
        self.norm = nn.RMSNorm(d_model)
        # pi, mu (2), sd (2), rho, eos
        self.mdn = nn.Linear(d_model, 6 * mixtures + 1)
        self.char_head = nn.Linear(d_model, MAX_TEXT + 1)

    def embed_text(self, text, text_len):
        """text: (B, T) left-padded, ending with SEP appended here."""
        B, T = text.shape
        sep = torch.full((B, 1), SEP, dtype=text.dtype, device=text.device)
        tokens = torch.cat([text, sep], dim=1)
        # Position of each character in its own text (0 for the first char);
        # padding gets 0 too and SEP gets text_len.
        pos = torch.arange(T + 1, device=text.device) - (T - text_len).unsqueeze(1)
        return self.char_embed(tokens) + self.text_pos(pos.clamp(0, MAX_TEXT))

    def run(self, x, inference_params=None):
        for block in self.blocks:
            x = block(x, inference_params)
        return self.norm(x)

    def forward(self, text, text_len, offsets):
        """Hidden states at SEP and every ink point: (B, N + 1, d). Position
        j predicts point j + 1 (or "finished" at j = N)."""
        prefix = self.embed_text(text, text_len)
        h = self.run(torch.cat([prefix, self.point_in(offsets)], dim=1))
        return h[:, prefix.shape[1] - 1:]

    def heads(self, h, bias=0.0):
        m = self.mixtures
        out = self.mdn(h).float()
        pi_logit = out[..., :m] * (1 + bias)
        mu = out[..., m:3 * m]
        log_sd = out[..., 3 * m:5 * m] - bias
        rho = torch.tanh(out[..., 5 * m:6 * m]).clamp(-0.999, 0.999)
        eos_logit = out[..., 6 * m]
        return pi_logit, mu, log_sd, rho, eos_logit, self.char_head(h).float()

    def loss(self, batch):
        """Mean per-point NLL of the ink (mixture + end of stroke), and the
        character-index cross-entropy, over real (unpadded) positions."""
        text, text_len, offsets, chars, ink_len = (batch[k] for k in ("text", "text_len", "offsets", "chars", "ink_len"))
        h = self.forward(text, text_len, offsets)
        pi_logit, mu, log_sd, rho, eos_logit, char_logit = self.heads(h)
        B, N, _ = offsets.shape
        steps = torch.arange(N + 1, device=offsets.device)
        point_mask = steps[:N].unsqueeze(0) < ink_len.unsqueeze(1)    # positions 0..N-1 predict points 1..N
        char_mask = steps.unsqueeze(0) <= ink_len.unsqueeze(1)        # plus the "finished" target

        m = self.mixtures
        target = offsets
        x1, x2 = target[..., 0:1], target[..., 1:2]
        mu1, mu2 = mu[:, :N, :m], mu[:, :N, m:]
        ls1, ls2 = log_sd[:, :N, :m], log_sd[:, :N, m:]
        r = rho[:, :N]
        z1, z2 = (x1 - mu1) * torch.exp(-ls1), (x2 - mu2) * torch.exp(-ls2)
        one_minus = 1 - r ** 2
        log_n = (-math.log(2 * math.pi) - ls1 - ls2 - 0.5 * torch.log(one_minus)
                 - (z1 ** 2 + z2 ** 2 - 2 * r * z1 * z2) / (2 * one_minus))
        log_mix = torch.logsumexp(F.log_softmax(pi_logit[:, :N], dim=-1) + log_n, dim=-1)
        eos_nll = F.binary_cross_entropy_with_logits(eos_logit[:, :N], target[..., 2], reduction="none")
        ink_nll = ((-log_mix + eos_nll) * point_mask).sum() / point_mask.sum()

        char_target = torch.cat([chars, torch.zeros(B, 1, dtype=chars.dtype, device=chars.device)], dim=1)
        char_target[torch.arange(B), ink_len] = text_len
        char_ce = F.cross_entropy(char_logit.transpose(1, 2), char_target, reduction="none")
        char_ce = (char_ce * char_mask).sum() / char_mask.sum()
        char_acc = ((char_logit.argmax(-1) == char_target) & char_mask).sum() / char_mask.sum()
        return ink_nll, char_ce, char_acc

    @torch.no_grad()
    def sample(self, text, text_len, bias=0.0, max_steps_per_char=40, generator=None):
        """Write a batch of lines. text: (B, T) left-padded tokens.

        Returns per line (offsets (n, 3), chars (n,), finished). A line stops
        when the index head's most likely class is "finished"; points are
        labelled with the most likely real character.
        """
        B = text.shape[0]
        device = text.device
        steps = max_steps_per_char * int(text_len.max()) + 100
        ip = InferenceParams(max_seqlen=text.shape[1] + 1 + steps, max_batch_size=B)
        prefix = self.embed_text(text, text_len)
        h = self.run(prefix, ip)[:, -1:]
        ip.seqlen_offset += prefix.shape[1]

        done = torch.zeros(B, dtype=torch.bool, device=device)
        ended = torch.full((B,), -1, device=device)
        points, labels = [], []
        valid = torch.arange(MAX_TEXT + 1, device=device).unsqueeze(0) < text_len.unsqueeze(1)
        for n in range(steps):
            pi_logit, mu, log_sd, rho, eos_logit, char_logit = (t[:, 0] for t in self.heads(h, bias))
            finish = ~done & (char_logit.argmax(-1) == text_len)
            ended[finish] = n
            done |= finish
            if done.all():
                break
            x = sample_point(pi_logit, mu, log_sd, rho, eos_logit, generator)
            points.append(x)
            labels.append(char_logit.masked_fill(~valid, -torch.inf).argmax(-1))
            h = self.run(self.point_in(x).unsqueeze(1).to(h.dtype), ip)
            ip.seqlen_offset += 1

        points = torch.stack(points, dim=1).cpu() if points else torch.zeros(B, 0, 3)
        labels = torch.stack(labels, dim=1).cpu() if labels else torch.zeros(B, 0, dtype=torch.long)
        out = []
        for b in range(B):
            n = int(ended[b]) if done[b] else points.shape[1]
            off = points[b, :n].clone()
            if n:
                off[-1, 2] = 1.0
            out.append((off, labels[b, :n], bool(done[b])))
        return out


def sample_point(pi_logit, mu, log_sd, rho, eos_logit, generator=None):
    """One (dx, dy, eos) per row. End of stroke is thresholded, as the teacher
    did when it generated the corpus."""
    m = pi_logit.shape[-1]
    comp = torch.multinomial(F.softmax(pi_logit, dim=-1), 1, generator=generator)
    mu1, mu2 = mu[:, :m].gather(1, comp), mu[:, m:].gather(1, comp)
    sd1, sd2 = log_sd[:, :m].gather(1, comp).exp(), log_sd[:, m:].gather(1, comp).exp()
    r = rho.gather(1, comp)
    z1 = torch.randn(mu1.shape, generator=generator, device=mu1.device)
    z2 = torch.randn(mu1.shape, generator=generator, device=mu1.device)
    dx = mu1 + sd1 * z1
    dy = mu2 + sd2 * (r * z1 + torch.sqrt(1 - r ** 2) * z2)
    return torch.cat([dx, dy, (eos_logit > 0).float().unsqueeze(1)], dim=1)


def load_student(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Student(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt

"""Mamba + attention text-to-ink student, sized like the flow student.

It writes one point at a time with the Mamba student's heads and loss
(scripts/student_model.py), so their ink_nll values are comparable. Its width,
depth and text encoder are the flow student's (scripts/flow_model.py), which
puts the two at about the same parameter count: what is left to compare is
writing point by point against flow matching.

The sequence is [START, p1 .. pN]; position j predicts point j + 1 (or
"finished" at j = N). There is no text prefix, so lines of any text length
share a batch. Each block is Mamba2, then attention in two places, then an
MLP:

- stroke attention reads the block's own hidden state at START and at the
  last point of every stroke finished so far. Mamba carries the pen's local
  motion; these summaries give exact recall of what was drawn earlier (where
  a fraction bar began, where the baseline is) at tens of memory tokens
  instead of thousands of points;
- text attention reads the text, encoded by a bidirectional transformer as
  in the flow student. Reading it in every block lets what was read enter
  the state of the Mamba layers above.

Each point also carries where the pen is after it: the sum of the offsets so
far, in typical ink heights. Offsets alone would leave the model to integrate
them, and the stroke summaries could not be looked up by place.

sample() returns what Student.sample() returns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

from flow_model import sinusoid
from ink_corpus import MAX_TEXT, PAD, SEP
from student_model import Student, sample_point

# Sinusoid input units per ink height: the shortest wavelength is then about
# 0.06 heights and the longest far wider than any line.
POSITION_SCALE = 100.0


class Attention(nn.Module):
    """Multi-head attention of normalized queries x (B, L, d) over a memory
    (B, S, d). visible, broadcastable to (B, 1, L, S), is set where a query
    may read a memory token; every query must see at least one."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

    def forward(self, x, memory, visible):
        context = F.scaled_dot_product_attention(self.split_heads(self.query(x)), self.split_heads(self.key(memory)),
                                                 self.split_heads(self.value(memory)), attn_mask=visible)
        return self.out(context.transpose(1, 2).reshape(x.shape))


class StrokeIndex:
    """The stroke summaries of a whole teacher-forced batch. ended (B, L) is
    set at the positions that are summaries."""

    def __init__(self, ended):
        count = ended.sum(dim=1)
        slots = torch.arange(int(count.max()), device=ended.device)
        # A stable sort lists each row's summary positions first, in order.
        self.positions = ended.int().argsort(dim=1, descending=True, stable=True)[:, :len(slots)]
        stored = slots.unsqueeze(0) < count.unsqueeze(1)
        query = torch.arange(ended.shape[1], device=ended.device)
        causal = self.positions.unsqueeze(1) <= query.view(1, -1, 1)
        self.visible = (stored.unsqueeze(1) & causal).unsqueeze(1)

    def read(self, h, layer):
        return h.gather(1, self.positions.unsqueeze(-1).expand(-1, -1, h.shape[-1])), self.visible


class StrokeCache:
    """The stroke summaries collected so far while sampling, per block."""

    def __init__(self, n_layers):
        self.memory = [None] * n_layers
        self.count = None

    def begin(self, ended):
        """Call before each position. ended (B,) is set for the rows where
        this position is a summary."""
        self.rows = ended.nonzero(as_tuple=True)[0]
        self.slots = (torch.zeros_like(ended, dtype=torch.long) if self.count is None else self.count)[self.rows]
        self.count = ended.long() if self.count is None else self.count + ended

    def read(self, h, layer):
        memory = self.memory[layer]
        if memory is None:
            memory = h.new_zeros(h.shape[0], 0, h.shape[2])
        needed = int(self.count.max())
        if memory.shape[1] < needed:
            memory = F.pad(memory, (0, 0, 0, needed - memory.shape[1]))
        memory[self.rows, self.slots] = h[self.rows, 0]
        self.memory[layer] = memory
        visible = torch.arange(memory.shape[1], device=h.device).unsqueeze(0) < self.count.unsqueeze(1)
        return memory, visible[:, None, None, :]


class Block(nn.Module):
    def __init__(self, d_model, heads, layer_idx, d_state, headdim):
        super().__init__()
        self.mamba_norm = nn.RMSNorm(d_model)
        self.stroke_norm = nn.RMSNorm(d_model)
        self.text_norm = nn.RMSNorm(d_model)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mamba = Mamba2(d_model, d_state=d_state, headdim=headdim, layer_idx=layer_idx)
        self.stroke_attention = Attention(d_model, heads)
        self.text_attention = Attention(d_model, heads)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, text, text_visible, strokes, layer, inference_params=None):
        x = x + self.mamba(self.mamba_norm(x), inference_params=inference_params)
        h = self.stroke_norm(x)
        memory, visible = strokes.read(h, layer)
        x = x + self.stroke_attention(h, memory, visible)
        x = x + self.text_attention(self.text_norm(x), text, text_visible)
        return x + self.mlp(self.mlp_norm(x))


class HybridStudent(nn.Module):
    def __init__(self, vocab_size, offset_mu, offset_std, typical_height, d_model=384, n_layers=8, heads=6,
                 text_layers=3, d_state=64, headdim=32, mixtures=20):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, offset_mu=list(offset_mu), offset_std=list(offset_std),
                           typical_height=typical_height, d_model=d_model, n_layers=n_layers, heads=heads,
                           text_layers=text_layers, d_state=d_state, headdim=headdim, mixtures=mixtures)
        self.mixtures = mixtures
        self.d_model = d_model
        self.typical_height = typical_height
        self.register_buffer("offset_mu", torch.tensor(offset_mu[:2], dtype=torch.float32))
        self.register_buffer("offset_std", torch.tensor(offset_std[:2], dtype=torch.float32))

        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.text_pos = nn.Embedding(MAX_TEXT + 1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.text_encoder = nn.TransformerEncoder(layer, text_layers, norm=nn.LayerNorm(d_model),
                                                  enable_nested_tensor=False)

        self.start = nn.Parameter(torch.randn(d_model) / d_model ** 0.5)
        self.point_in = nn.Linear(3, d_model)
        self.position_in = nn.Linear(2 * d_model, d_model)
        self.blocks = nn.ModuleList(Block(d_model, heads, i, d_state, headdim) for i in range(n_layers))
        self.norm = nn.RMSNorm(d_model)
        # pi, mu (2), sd (2), rho, eos
        self.mdn = nn.Linear(d_model, 6 * mixtures + 1)
        self.char_head = nn.Linear(d_model, MAX_TEXT + 1)

    # The same outputs and targets as the Mamba student, so the two are
    # trained and measured alike.
    heads = Student.heads
    loss = Student.loss

    def encode_text(self, text, text_len):
        """text: (B, T) left-padded. Returns the text memory (B, T + 1, d) with
        SEP last, and where it is not padding."""
        B, T = text.shape
        sep = torch.full((B, 1), SEP, dtype=text.dtype, device=text.device)
        tokens = torch.cat([text, sep], dim=1)
        pos = torch.arange(T + 1, device=text.device) - (T - text_len).unsqueeze(1)
        pad = pos < 0
        embedded = self.char_embed(tokens) + self.text_pos(pos.clamp(0, MAX_TEXT))
        memory = self.text_encoder(embedded, src_key_padding_mask=pad)
        return memory, ~pad[:, None, None, :]

    def embed_points(self, points, pen):
        """points (B, N, 3) normalized offsets, pen (B, N, 2) the pen's
        position after each of them in denormalized offset units."""
        place = sinusoid(pen / self.typical_height * POSITION_SCALE, self.d_model).flatten(-2)
        return self.point_in(points) + self.position_in(place)

    def forward(self, text, text_len, offsets):
        """Hidden states at START and every ink point: (B, N + 1, d)."""
        B = offsets.shape[0]
        memory, text_visible = self.encode_text(text, text_len)
        pen = torch.cumsum(offsets[..., :2] * self.offset_std + self.offset_mu, dim=1)
        x = torch.cat([self.start.expand(B, 1, -1), self.embed_points(offsets, pen)], dim=1)
        # START is a summary too, so a query in the first stroke has something to read.
        start = torch.ones(B, 1, dtype=torch.bool, device=offsets.device)
        strokes = StrokeIndex(torch.cat([start, offsets[..., 2] > 0.5], dim=1))
        for layer, block in enumerate(self.blocks):
            x = block(x, memory, text_visible, strokes, layer)
        return self.norm(x)

    @torch.no_grad()
    def sample(self, text, text_len, bias=0.0, max_steps_per_char=40, generator=None):
        """Write a batch of lines. text: (B, T) left-padded tokens.

        Returns per line (offsets (n, 3), chars (n,), finished), as
        Student.sample() does, and stops a line the same way."""
        B, device = text.shape[0], text.device
        steps = max_steps_per_char * int(text_len.max()) + 100
        memory, text_visible = self.encode_text(text, text_len)
        ip = InferenceParams(max_seqlen=steps + 1, max_batch_size=B)
        strokes = StrokeCache(len(self.blocks))

        x = self.start.expand(B, 1, -1)
        stroke_ended = torch.ones(B, dtype=torch.bool, device=device)
        pen = torch.zeros(B, 2, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        ended = torch.full((B,), -1, device=device)
        points, labels = [], []
        valid = torch.arange(MAX_TEXT + 1, device=device).unsqueeze(0) < text_len.unsqueeze(1)
        for n in range(steps):
            strokes.begin(stroke_ended)
            for layer, block in enumerate(self.blocks):
                x = block(x, memory, text_visible, strokes, layer, ip)
            ip.seqlen_offset += 1
            pi_logit, mu, log_sd, rho, eos_logit, char_logit = (t[:, 0] for t in self.heads(self.norm(x), bias))
            finish = ~done & (char_logit.argmax(-1) == text_len)
            ended[finish] = n
            done |= finish
            if done.all():
                break
            point = sample_point(pi_logit, mu, log_sd, rho, eos_logit, generator)
            points.append(point)
            labels.append(char_logit.masked_fill(~valid, -torch.inf).argmax(-1))
            pen = pen + point[:, :2] * self.offset_std + self.offset_mu
            stroke_ended = point[:, 2] > 0.5
            x = self.embed_points(point.unsqueeze(1), pen.unsqueeze(1))

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

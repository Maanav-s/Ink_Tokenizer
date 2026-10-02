"""Graves (2013) handwriting synthesis network, ported from
pytorch-handwriting-synthesis-toolkit (MIT, X-rayLaser) to load its pretrained
weights with current torch.

This is the teacher for the Mamba text-to-ink student. Parameter names match
the original so its state_dict loads unchanged. Training code is not ported:
the teacher is only sampled.

Sampling differs from the original in two ways:
- it is batched. A batch shares one priming sequence (one writer) and holds
  many lines of text, which is how corpus generation uses it;
- it returns the attention window's argmax at every step, which aligns each
  pen point with the character it draws.

Ink is in the network's normalized offset space: (dx, dy, end_of_stroke) per
step. Undo it with the checkpoint's mu/std (see Teacher.denormalize).
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT = "Epoch_52"  # the one the toolkit's own demo and txt2script use
SENTINEL = "  "          # the toolkit's examples end every line with two spaces


class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ci = nn.Parameter(torch.empty(hidden_size))
        self.b_i = nn.Parameter(torch.empty(hidden_size))
        self.W_cf = nn.Parameter(torch.empty(hidden_size))
        self.b_f = nn.Parameter(torch.empty(hidden_size))
        self.b_c = nn.Parameter(torch.empty(hidden_size))
        self.W_co = nn.Parameter(torch.empty(hidden_size))
        self.b_o = nn.Parameter(torch.empty(hidden_size))
        self.W_xh = nn.Parameter(torch.empty(input_size + hidden_size, hidden_size * 4))

    def step(self, x, state):
        h, c = state
        z_i, z_f, z_c, z_o = (torch.cat([x, h], dim=1) @ self.W_xh).chunk(4, 1)
        i = torch.sigmoid(z_i + c * self.W_ci + self.b_i)
        f = torch.sigmoid(z_f + c * self.W_cf + self.b_f)
        c = f * c + i * torch.tanh(z_c + self.b_c)
        o = torch.sigmoid(z_o + c * self.W_co + self.b_o)
        h = o * torch.tanh(c)
        return h, (h, c)


class SoftWindow(nn.Module):
    def __init__(self, input_size, num_components):
        super().__init__()
        self.alpha = nn.Linear(input_size, num_components)
        self.beta = nn.Linear(input_size, num_components)
        self.k = nn.Linear(input_size, num_components)

    def step(self, h, chars, prev_k):
        """h: (B, H), chars: (B, U, A) one-hot, prev_k: (B, K).
        Returns the window vector (B, A), phi (B, U) and k (B, K)."""
        alpha = torch.exp(self.alpha(h)).unsqueeze(2)
        beta = torch.exp(self.beta(h)).unsqueeze(2)
        k = prev_k + torch.exp(self.k(h))
        u = torch.arange(chars.shape[1], device=h.device)
        phi = (alpha * torch.exp(-beta * (k.unsqueeze(2) - u) ** 2)).sum(dim=1)
        return torch.bmm(phi.unsqueeze(1), chars).squeeze(1), phi, k


class MixtureDensityLayer(nn.Module):
    def __init__(self, input_size, num_components):
        super().__init__()
        self.num_components = num_components
        self.pi = nn.Linear(input_size, num_components)
        self.mu = nn.Linear(input_size, num_components * 2)
        self.sd = nn.Linear(input_size, num_components * 2)
        self.ro = nn.Linear(input_size, num_components)
        self.eos = nn.Linear(input_size, 1)

    def forward(self, x, bias=0.0):
        """bias > 0 sharpens the mixture: neater, less varied handwriting."""
        pi = F.softmax(self.pi(x) * (1 + bias), dim=-1)
        sd = torch.exp(self.sd(x) - bias)
        mu = self.mu(x)
        ro = torch.tanh(self.ro(x))
        eos = torch.sigmoid(self.eos(x)).squeeze(-1)
        return pi, mu, sd, ro, eos


def sample_mixture(pi, mu, sd, ro, eos, generator):
    """One (dx, dy, eos) per batch row. End of stroke is thresholded rather than
    sampled, as in the original toolkit."""
    m = pi.shape[1]
    comp = torch.multinomial(pi, 1, generator=generator)
    mu1, mu2 = mu[:, :m].gather(1, comp), mu[:, m:].gather(1, comp)
    sd1, sd2 = sd[:, :m].gather(1, comp), sd[:, m:].gather(1, comp)
    r = ro.gather(1, comp)
    z1 = torch.randn(mu1.shape, generator=generator, device=mu1.device)
    z2 = torch.randn(mu1.shape, generator=generator, device=mu1.device)
    dx = mu1 + sd1 * z1
    dy = mu2 + sd2 * (r * z1 + torch.sqrt(1 - r ** 2) * z2)
    return torch.cat([dx, dy, (eos > 0.5).float().unsqueeze(1)], dim=1)


class SynthesisNetwork(nn.Module):
    def __init__(self, alphabet_size, input_size=3, hidden_size=400, gaussian_components=10, output_mixtures=20):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.gaussian_components = gaussian_components
        self.lstm1 = PeepholeLSTM(input_size + alphabet_size, hidden_size)
        self.window = SoftWindow(hidden_size, gaussian_components)
        self.lstm2 = PeepholeLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.lstm3 = PeepholeLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.mixture = MixtureDensityLayer(hidden_size * 3, output_mixtures)

    def initial_state(self, batch_size, device):
        def zeros():
            h = torch.zeros(batch_size, self.lstm1.hidden_size, device=device)
            return h, torch.zeros_like(h)
        return {"s1": zeros(), "s2": zeros(), "s3": zeros(),
                "w": torch.zeros(batch_size, self.alphabet_size, device=device),
                "k": torch.zeros(batch_size, self.gaussian_components, device=device)}

    def step(self, x, chars, state, bias=0.0):
        """Advance one pen point. Returns the mixture parameters for the next
        point, phi over the characters, and the new state."""
        h1, s1 = self.lstm1.step(torch.cat([x, state["w"]], dim=1), state["s1"])
        w, phi, k = self.window.step(h1, chars, state["k"])
        h2, s2 = self.lstm2.step(torch.cat([x, h1, w], dim=1), state["s2"])
        h3, s3 = self.lstm3.step(torch.cat([x, h2, w], dim=1), state["s3"])
        mixture = self.mixture(torch.cat([h1, h2, h3], dim=1), bias)
        return mixture, phi, {"s1": s1, "s2": s2, "s3": s3, "w": w, "k": k}


class Teacher:
    """The pretrained synthesis network plus its tokenizer and normalization."""

    def __init__(self, checkpoint_dir, device="cpu"):
        meta = json.load(open(os.path.join(checkpoint_dir, "meta.json")))
        self.charset = meta["charset"]
        self.mu = torch.tensor(meta["mu"])
        self.std = torch.tensor(meta["std"])
        self.device = torch.device(device)
        # Token 0 is the toolkit's padding/unknown character.
        self.index = {ch: i + 1 for i, ch in enumerate(self.charset)}
        self.net = SynthesisNetwork(len(self.charset) + 1).to(self.device)
        state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()

    def unsupported(self, text):
        return sorted({ch for ch in text if ch not in self.index})

    def encode(self, texts):
        """One-hot (B, U, A), zero-padded to the longest text."""
        u = max(len(t) for t in texts)
        out = torch.zeros(len(texts), u, len(self.charset) + 1, device=self.device)
        for b, text in enumerate(texts):
            for i, ch in enumerate(text):
                out[b, i, self.index[ch]] = 1.0
        return out

    def denormalize(self, offsets):
        return offsets * self.std.to(offsets.device) + self.mu.to(offsets.device)

    @torch.no_grad()
    def sample(self, texts, bias=0.0, seed=0, prime=None, max_steps_per_char=40):
        """Write each text as one line, all in the same style.

        prime: (prime_text, prime_offsets) from an earlier sample. The network
        first reads that ink as if it had written it, which carries its style
        over (Graves 2013, section 5.5).

        Returns one dict per text:
          offsets: (N, 3) normalized (dx, dy, eos), starting after the prime;
          char: (N,) index into the text of the character each point draws,
                from the window's argmax. It can fall outside the text:
                below 0 while the window is still on the prime, len(text)
                or more once it reaches the end sentinel;
          finished: whether the window reached the end of the text before
                the step limit. Unfinished lines are truncated and should be
                rejected.
        """
        for t in texts:
            bad = self.unsupported(t)
            if bad:
                raise ValueError(f"teacher cannot write {bad!r} in {t!r}")
        gen = torch.Generator(device=self.device).manual_seed(seed)
        prime_text = prime[0] if prime else ""
        full = [prime_text + t + SENTINEL for t in texts]
        chars = self.encode(full)
        B = len(texts)
        start = len(prime_text)
        last = torch.tensor([len(f) - 1 for f in full], device=self.device)

        state = self.net.initial_state(B, self.device)
        x = torch.zeros(B, 3, device=self.device)
        if prime:
            for p in prime[1].to(self.device):
                _, _, state = self.net.step(x, chars, state, bias)
                x = p.expand(B, 3)

        steps = max_steps_per_char * max(len(t) for t in texts) + 100
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        points, aligned, ended = [], [], torch.full((B,), -1, device=self.device)
        for n in range(steps):
            mixture, phi, state = self.net.step(x, chars, state, bias)
            x = sample_mixture(*mixture, gen)
            at = phi.argmax(dim=1)
            # The original toolkit's end test: the window has settled on the
            # final sentinel character.
            last_phi = phi.gather(1, last.unsqueeze(1)).squeeze(1)
            end = ~done & ((last_phi > 0.8) | ((at == last) & (x[:, 2] > 0.5)))
            x[end, 2] = 1.0
            ended[end] = n
            points.append(x.clone())
            aligned.append(at)
            done |= end
            if done.all():
                break

        points = torch.stack(points, dim=1).cpu()
        aligned = torch.stack(aligned, dim=1).cpu()
        out = []
        for b, text in enumerate(texts):
            n = int(ended[b]) + 1 if done[b] else points.shape[1]
            out.append({"offsets": points[b, :n], "char": aligned[b, :n] - start, "finished": bool(done[b])})
        return out


def default_checkpoint_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "teacher", CHECKPOINT)

"""Flow-matching student that writes B-spline control points instead of
points (an experiment; the point flow student is scripts/flow_model.py).

The model is FlowStudent unchanged, trained on a corpus converted by
scripts/spline_corpus.py, so its "points" are each stroke's control points
and the pen channel marks a stroke's last one. A control point stands for
about points_per_control points, so lines are that many times shorter, and
the curve between control points is smooth by construction rather than by
the smooth loss.

The length head counts control points, in bins of SPLINE_LENGTH_BIN: bins of
8 would be about 30 points, a large error on a short ink.

sample() decodes the control points into dense ink (spline_corpus.decode_ink),
so it returns what FlowStudent.sample() does and scripts/sample_student.py,
scripts/visualize_flow.py, scripts/measure_legibility.py and the page renderer
work with it as they are. dense_ink() does the same decoding in torch,
differentiably, for the recognizer's loss (scripts/train_flow_legibility.py).
"""
import numpy as np
import torch

from flow_model import MIN_HEIGHT, FlowStudent
from spline_corpus import decode_ink, decode_weights

SPLINE_LENGTH_BIN = 2  # control points per length class; 300 bins reach 600


class SplineFlowStudent(FlowStudent):
    length_bin = SPLINE_LENGTH_BIN

    def __init__(self, points_per_control, **config):
        super().__init__(**config)
        self.points_per_control = points_per_control
        self.config["points_per_control"] = points_per_control

    @torch.no_grad()
    def sample(self, text, text_len, bias=0.0, steps=None, guidance=None, generator=None):
        mu, std = self.offset_mu.cpu().numpy(), self.offset_std.cpu().numpy()
        out = []
        for control, chars, finished in super().sample(text, text_len, bias, steps, guidance, generator):
            offsets, labels = decode_ink(control.numpy(), chars.numpy(), self.points_per_control, mu, std)
            out.append((torch.from_numpy(offsets), torch.from_numpy(labels), finished))
        return out

    def dense_ink(self, x, pen, ink_len):
        """Control points in the flow's form (B, N, 3), with the strokes given
        by pen (B, N), the 0/1 flags of the corpus offsets, -> the decoded
        ink in the recognizer's input form (recognizer_model.ink_positions)
        (B, M, 3) and its lengths (B,). Differentiable in x's positions; the
        pen comes from pen, as x's pen channel is not exact mid-flow."""
        B, device = x.shape[0], x.device
        decoded = [decode_weights(pen[b, :n].float().cpu().numpy(), self.points_per_control)
                   for b, n in enumerate(ink_len.tolist())]
        lengths = torch.tensor([len(ends) for _, _, ends in decoded], device=device)
        M, K = int(lengths.max()), decoded[0][0].shape[1]
        index = torch.zeros(B, M, K, dtype=torch.long)
        weight = torch.zeros(B, M, K)
        ends = torch.zeros(B, M)
        for b, (i, w, e) in enumerate(decoded):
            index[b, :len(e)], weight[b, :len(e)], ends[b, :len(e)] = (torch.from_numpy(i), torch.from_numpy(w),
                                                                       torch.from_numpy(e.astype(np.float32)))
        index, weight, ends = index.to(device), weight.to(device), ends.to(device)

        xy = x[..., :2].float() * self.coord_std
        mixed = torch.gather(xy, 1, index.view(B, M * K, 1).expand(-1, -1, 2)).view(B, M, K, 2)
        xy = (mixed * weight.unsqueeze(-1)).sum(dim=2)
        # The decoded ink's centre and height differ a little from the
        # control points', so normalize it again as ink_positions does.
        mask = (torch.arange(M, device=device) < lengths.unsqueeze(1)).unsqueeze(-1)
        centre = (xy * mask).sum(dim=1, keepdim=True) / lengths.view(-1, 1, 1)
        y = xy[..., 1:]
        height = y.masked_fill(~mask, -torch.inf).amax(dim=1) - y.masked_fill(~mask, torch.inf).amin(dim=1)
        xy = (xy - centre) / height.clamp(min=MIN_HEIGHT).unsqueeze(1)
        return torch.cat([xy, ends.unsqueeze(-1) * 2 - 1], dim=-1) * mask, lengths

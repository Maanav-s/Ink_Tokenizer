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
scripts/visualize_flow.py and the page renderer work with it as they are.
"""
import numpy as np
import torch

from flow_model import FlowStudent
from spline_corpus import decode_ink

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

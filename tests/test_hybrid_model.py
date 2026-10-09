"""CPU checks for the hybrid student; a CPU recurrence stands in for Mamba's CUDA blocks."""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hybrid_model import HybridStudent, StrokeIndex


class Recurrence(nn.Module):
    """A causal running sum that keeps its state between sampling steps."""
    def forward(self, x, inference_params=None):
        h = x.cumsum(dim=1)
        if inference_params is not None:
            state = inference_params.key_value_memory_dict.get(id(self))
            if state is not None:
                h = h + state
            inference_params.key_value_memory_dict[id(self)] = h[:, -1:]
        return 0.1 * h


def hybrid():
    torch.manual_seed(3)
    model = HybridStudent(vocab_size=9, offset_mu=[0.1, 0.0, 0.0], offset_std=[1.0, 2.0, 1.0], typical_height=5.0,
                          d_model=32, n_layers=2, heads=4, text_layers=1, mixtures=2)
    for block in model.blocks:
        block.mamba = Recurrence()
    return model.eval()


def ink(batch, points):
    offsets = torch.randn(batch, points, 3)
    offsets[..., 2] = (torch.rand(batch, points) < 0.3).float()
    return offsets


class HybridTests(unittest.TestCase):
    def test_stroke_index_lists_finished_strokes_only(self):
        ended = torch.tensor([[1, 0, 1, 0, 0], [1, 0, 0, 0, 1]], dtype=torch.bool)
        strokes = StrokeIndex(ended)
        self.assertEqual(strokes.positions.tolist(), [[0, 2], [0, 4]])
        visible = strokes.visible[:, 0]
        self.assertEqual(visible[0].tolist(), [[True, False], [True, False], [True, True], [True, True], [True, True]])
        self.assertEqual(visible[1, 3].tolist(), [True, False])
        self.assertEqual(visible[1, 4].tolist(), [True, True])

    def test_a_position_does_not_depend_on_later_points(self):
        model = hybrid()
        text, text_len = torch.tensor([[2, 3, 4]]), torch.tensor([3])
        offsets = ink(1, 12)
        changed = offsets.clone()
        changed[:, 7:] = ink(1, 5)
        with torch.no_grad():
            a, b = model(text, text_len, offsets), model(text, text_len, changed)
        torch.testing.assert_close(a[:, :8], b[:, :8])
        self.assertFalse(torch.allclose(a[:, 8:], b[:, 8:]))

    def test_padding_and_batch_companions_do_not_change_a_line(self):
        model = hybrid()
        offsets = ink(2, 9)
        offsets[0, 6:] = 0
        with torch.no_grad():
            single = model(torch.tensor([[2, 3]]), torch.tensor([2]), offsets[:1, :6])
            mixed = model(torch.tensor([[0, 0, 2, 3], [4, 5, 6, 7]]), torch.tensor([2, 4]), offsets)
        torch.testing.assert_close(single, mixed[:1, :7], atol=1e-5, rtol=1e-4)

    def test_sampling_matches_the_teacher_forced_pass(self):
        model = hybrid()
        text, text_len = torch.tensor([[0, 2, 3], [4, 5, 6]]), torch.tensor([2, 3])
        seen = []
        heads = model.heads

        def recording_heads(h, bias=0.0):
            seen.append(h)
            return heads(h, bias)

        model.heads = recording_heads
        lines = model.sample(text, text_len, max_steps_per_char=1, generator=torch.Generator().manual_seed(0))
        model.heads = heads
        stepped = torch.cat(seen, dim=1)
        # Strokes must end mid-line, at different points per row, for the cache to be exercised.
        ends = [offsets[:-1, 2].nonzero().flatten().tolist() for offsets, _, _ in lines]
        self.assertTrue(ends[0] and ends[1] and ends[0] != ends[1])
        for b, (offsets, _, _) in enumerate(lines):
            n = len(offsets)
            self.assertGreater(n, 1)
            with torch.no_grad():
                full = model(text[b:b + 1], text_len[b:b + 1], offsets.unsqueeze(0))
            # sample() forces the last point to end a stroke, so only the
            # positions before it saw the same input.
            torch.testing.assert_close(full[0, :n], stepped[b, :n], atol=1e-4, rtol=1e-3)

    def test_loss_backpropagates_into_every_part(self):
        model = hybrid().train()
        offsets = ink(2, 6)
        batch = dict(text=torch.tensor([[0, 2, 3], [4, 5, 6]]), text_len=torch.tensor([2, 3]), offsets=offsets,
                     chars=torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 2, 2, 2]]), ink_len=torch.tensor([5, 6]))
        ink_nll, char_ce, _ = model.loss(batch)
        self.assertTrue(torch.isfinite(ink_nll + char_ce))
        (ink_nll + char_ce).backward()
        for name, parameter in model.named_parameters():
            if name.startswith("blocks.0.") or name in ("start", "position_in.weight", "char_embed.weight"):
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)


if __name__ == "__main__":
    unittest.main()

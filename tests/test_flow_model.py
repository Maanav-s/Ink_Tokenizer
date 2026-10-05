"""CPU checks for the flow-matching student's ink representation, loss and sampling."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flow_model import LENGTH_BIN, FlowStudent


class FlowStudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = FlowStudent(vocab_size=8, offset_mu=[0.1, -0.2, 0.0], offset_std=[0.5, 0.3, 1.0],
                                 coord_std=[2.0, 1.0], d_model=16, n_layers=2, heads=2, text_layers=1)
        offsets = torch.randn(2, 5, 3)
        offsets[..., 2] = torch.tensor([[0, 1, 0, 1, 0], [0, 0, 1, 0, 1]])
        self.batch = {"text": torch.tensor([[0, 2, 3], [4, 5, 6]]), "text_len": torch.tensor([2, 3]),
                      "offsets": offsets, "chars": torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 2, 2]]),
                      "ink_len": torch.tensor([4, 5])}

    def test_positions_round_trip_to_offsets(self):
        offsets, ink_len = self.batch["offsets"], self.batch["ink_len"]
        x = self.model.to_positions(offsets, ink_len)
        self.assertTrue((x[0, 4] == 0).all())
        torch.testing.assert_close(x[1, :, :2].mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
        back = self.model.to_offsets(x)
        # The first offset only says where the ink starts, which centring drops.
        torch.testing.assert_close(back[1, 1:], offsets[1, 1:], atol=1e-5, rtol=0)
        torch.testing.assert_close(back[0, 1:4], offsets[0, 1:4], atol=1e-5, rtol=0)

    def test_padding_does_not_change_the_velocity(self):
        longer = dict(self.batch)
        longer["offsets"] = torch.cat([self.batch["offsets"], torch.randn(2, 3, 3)], dim=1)
        longer["chars"] = torch.cat([self.batch["chars"], torch.zeros(2, 3, dtype=torch.long)], dim=1)
        # The velocity head starts at zero, which would make this check trivial.
        torch.nn.init.normal_(self.model.velocity.weight)
        velocity = []
        for batch in (self.batch, longer):
            x = self.model.to_positions(batch["offsets"], batch["ink_len"])
            memory, pad, _ = self.model.encode_text(batch["text"], batch["text_len"])
            velocity.append(self.model.denoise(x, torch.tensor([0.3, 0.7]), batch["ink_len"], memory, pad)[0])
        torch.testing.assert_close(velocity[1][0, :4], velocity[0][0, :4], atol=1e-5, rtol=0)
        torch.testing.assert_close(velocity[1][1, :5], velocity[0][1, :5], atol=1e-5, rtol=0)

    def test_loss_without_alignment_is_finite(self):
        losses = self.model.loss({**self.batch, "chars": torch.full((2, 5), -1)})
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        self.assertEqual(losses["char_ce"].item(), 0.0)

    def test_sample_matches_the_student_interface(self):
        self.model.eval()
        out = self.model.sample(self.batch["text"], self.batch["text_len"], bias=torch.zeros(2, 1, 1), steps=2)
        for offsets, chars, finished in out:
            self.assertEqual(offsets.shape[0] % LENGTH_BIN, LENGTH_BIN // 2)
            self.assertEqual(offsets.shape, (chars.shape[0], 3))
            self.assertEqual(offsets[-1, 2].item(), 1.0)
            self.assertTrue(finished)
        self.assertTrue((out[0][1] < 2).all())


if __name__ == "__main__":
    unittest.main()

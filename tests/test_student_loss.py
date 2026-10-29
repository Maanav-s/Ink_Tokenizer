"""CPU checks for the character-index loss with known and unknown alignment."""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from student_model import Student


class CharLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = Student(vocab_size=8, d_model=16, n_layers=0, mixtures=2)
        self.batch = {"text": torch.tensor([[0, 2, 3], [4, 5, 6]]), "text_len": torch.tensor([2, 3]),
                      "offsets": torch.randn(2, 4, 3), "ink_len": torch.tensor([3, 4])}

    def char_logits(self):
        b = self.batch
        return self.model.heads(self.model(b["text"], b["text_len"], b["offsets"]))[-1]

    def test_known_alignment_is_plain_cross_entropy(self):
        chars = torch.tensor([[0, 1, 1, 0], [0, 1, 2, 2]])
        _, char_ce, _ = self.model.loss({**self.batch, "chars": chars})
        logits = self.char_logits()
        targets = [[0, 1, 1, 2], [0, 1, 2, 2, 3]]
        expected = torch.cat([F.cross_entropy(logits[b, :len(t)], torch.tensor(t), reduction="none")
                              for b, t in enumerate(targets)]).mean()
        torch.testing.assert_close(char_ce, expected)

    def test_unknown_alignment_only_penalizes_finishing(self):
        chars = torch.full((2, 4), -1)
        _, char_ce, _ = self.model.loss({**self.batch, "chars": chars})
        p = F.softmax(self.char_logits(), dim=-1)
        losses = []
        for b, (n, u) in enumerate([(3, 2), (4, 3)]):
            losses.append(-torch.log(1 - p[b, :n, u]))
            losses.append(-torch.log(p[b, n:n + 1, u]))
        torch.testing.assert_close(char_ce, torch.cat(losses).mean())


if __name__ == "__main__":
    unittest.main()

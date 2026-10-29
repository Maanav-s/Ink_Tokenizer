"""CPU checks for text conditioning; Mamba's CUDA blocks are omitted."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from student_model import Student, TextAttention, load_student


class TextAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Student(vocab_size=8, d_model=16, n_layers=0, mixtures=2)
        self.text = torch.tensor([[0, 2, 3], [4, 5, 6]])
        self.lengths = torch.tensor([2, 3])

    def test_padding_is_invisible_and_text_affects_output(self):
        attention = self.model.text_attention
        prefix = self.model.embed_text(self.text, self.lengths)
        h = torch.randn(2, 4, 16)
        expected = attention(h, attention.prepare(prefix, self.lengths))
        changed = prefix.clone()
        changed[0, 0] = torch.randn(16) * 100
        torch.testing.assert_close(expected, attention(h, attention.prepare(changed, self.lengths)))
        changed[0, 1] = torch.randn(16) * 100
        self.assertFalse(torch.allclose(expected, attention(h, attention.prepare(changed, self.lengths))))

    def test_cached_attention_matches_full_sequence(self):
        attention = self.model.text_attention
        memory = attention.prepare(self.model.embed_text(self.text, self.lengths), self.lengths)
        h = torch.randn(2, 5, 16)
        full = attention(h, memory)
        stepped = torch.cat([attention(h[:, i:i + 1], memory) for i in range(5)], dim=1)
        torch.testing.assert_close(full, stepped)

    def test_loss_backpropagates_into_attention(self):
        offsets = torch.randn(2, 4, 3) * 0.1
        offsets[..., 2] = 0
        batch = dict(text=self.text, text_len=self.lengths, offsets=offsets,
                     chars=torch.tensor([[0, 0, 1, 1], [0, 1, 2, 0]]), ink_len=torch.tensor([4, 3]))
        ink, char, _ = self.model.loss(batch)
        self.assertTrue(torch.isfinite(ink + char))
        (ink + char).backward()
        for parameter in self.model.text_attention.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.text_attention.value.weight.grad.abs().sum().item(), 0)

    def test_sampling_uses_same_conditioning_and_prepares_once(self):
        text, lengths = self.text[1:], self.lengths[1:]
        seen = []
        original_heads = self.model.heads

        def heads(h, bias=0):
            seen.append(h.clone())
            outputs = original_heads(h, bias)
            outputs[-1].fill_(-100)
            outputs[-1][..., 3 if len(seen) == 4 else 0] = 100
            return outputs

        attention = self.model.text_attention
        with patch.object(self.model, "heads", side_effect=heads), \
                patch.object(attention, "prepare", wraps=attention.prepare) as prepare:
            offsets, _, finished = self.model.sample(text, lengths)[0]
            self.assertEqual(prepare.call_count, 1)
        self.assertTrue(finished)
        self.assertEqual(len(offsets), 3)
        # Sampling forces the last point's pen-up flag, so compare preceding predictions.
        full = self.model(text, lengths, offsets[None, :-1])
        torch.testing.assert_close(torch.cat(seen[:3], dim=1), full)

    def test_new_and_legacy_checkpoints_load_strictly(self):
        for heads in (0, 4):
            model = Student(vocab_size=8, d_model=16, n_layers=0, cross_attention_heads=heads)
            config = dict(model.config)
            if heads == 0:
                del config["cross_attention_heads"]
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
                path = os.path.join(directory, "checkpoint.pt")
                torch.save(dict(config=config, model=model.state_dict()), path)
                loaded, _ = load_student(path, "cpu")
            self.assertEqual(loaded.config["cross_attention_heads"], heads)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name])

    def test_empty_text_and_invalid_head_count(self):
        result = self.model(torch.empty(1, 0, dtype=torch.long), torch.tensor([0]), torch.zeros(1, 1, 3))
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaises(ValueError):
            TextAttention(16, 3)


if __name__ == "__main__":
    unittest.main()

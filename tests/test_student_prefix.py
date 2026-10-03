"""Padding invariance, encoder gradients, and checkpoint architecture checks."""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ink_corpus import Corpus
from student_model import Student, load_student


class RecurrentBlock(nn.Module):
    """CPU recurrence that makes unwanted prefix steps observable."""
    def forward(self, x, inference_params=None):
        h = x.cumsum(dim=1)
        if inference_params is not None:
            state = inference_params.key_value_memory_dict.get(0)
            if state is not None:
                h = h + state
            inference_params.key_value_memory_dict[0] = h[:, -1:]
        return h


class PrefixTests(unittest.TestCase):
    def model(self, conditioning):
        torch.manual_seed(12)
        model = Student(vocab_size=9, d_model=16, n_layers=0, mixtures=2,
                        conditioning=conditioning, prefix_slots=4)
        model.blocks.append(RecurrentBlock())
        return model.eval()

    def test_predictions_are_invariant_to_padding_and_batch_companions(self):
        offsets = torch.randn(2, 5, 3) * 0.1
        for conditioning in ("unpadded", "transformer"):
            model = self.model(conditioning)
            single = model(torch.tensor([[2, 3]]), torch.tensor([2]), offsets[:1])
            padded = model(torch.tensor([[0, 0, 2, 3]]), torch.tensor([2]), offsets[:1])
            mixed = model(torch.tensor([[0, 0, 2, 3], [4, 5, 6, 7]]), torch.tensor([2, 4]), offsets)
            torch.testing.assert_close(single, padded)
            torch.testing.assert_close(single, mixed[:1])
        legacy = self.model("legacy")
        self.assertFalse(torch.allclose(
            legacy(torch.tensor([[2, 3]]), torch.tensor([2]), offsets[:1]),
            legacy(torch.tensor([[0, 0, 2, 3]]), torch.tensor([2]), offsets[:1])))

    def test_cached_sampling_matches_training_and_preserves_order_and_bias(self):
        for conditioning in ("unpadded", "transformer"):
            model = self.model(conditioning)
            text = torch.tensor([[0, 0, 2, 3], [4, 5, 6, 7]])
            lengths = torch.tensor([2, 4])
            original_heads = model.heads
            seen = {}

            def heads(h, bias=0):
                length = int(bias[0, 0, 0])
                seen.setdefault(length, []).append(h.detach().clone())
                outputs = original_heads(h, 0)
                outputs[-1].fill_(-100)
                outputs[-1][..., length if len(seen[length]) == 4 else 0] = 100
                return outputs

            # Test each row with extra PAD; transformer also exercises its one-time encoder cache.
            for row, length in enumerate(lengths.tolist()):
                with patch.object(model, "heads", side_effect=heads), \
                        patch.object(model, "conditioning_prefix", wraps=model.conditioning_prefix) as prepare:
                    offsets, _, finished = model.sample(text[row:row + 1], lengths[row:row + 1],
                                                        bias=torch.tensor([[[float(length)]]]))[0]
                    self.assertEqual(prepare.call_count, 1)
                self.assertTrue(finished)
                full = model(text[row:row + 1], lengths[row:row + 1], offsets[None, :-1])
                torch.testing.assert_close(torch.cat(seen[length][:3], dim=1), full)
            if conditioning == "unpadded":
                def sample_group(tokens, group_lengths, group_bias, *args):
                    return [(int(length), float(bias.item())) for length, bias in zip(group_lengths, group_bias)]

                with patch.object(model, "sample_sequence", side_effect=sample_group) as sample:
                    result = model.sample(text.flip(0), lengths.flip(0), torch.tensor([[[4.]], [[2.]]]))
                self.assertEqual(result, [(4, 4.), (2, 2.)])
                self.assertEqual([int(call.args[2].item()) for call in sample.call_args_list], [2, 4])

    def test_transformer_slots_and_gradients(self):
        model = self.model("transformer")
        text, lengths = torch.tensor([[0, 2, 3], [4, 5, 6]]), torch.tensor([2, 3])
        prefix, memory, memory_len = model.conditioning_prefix(text, lengths)
        self.assertEqual(prefix.shape, (2, 5, 16))
        self.assertEqual(memory.shape, (2, 4, 16))
        self.assertIsNone(memory_len)
        offsets = torch.randn(2, 4, 3) * 0.1
        offsets[..., 2] = 0
        losses = model.loss(dict(text=text, text_len=lengths, offsets=offsets,
                                 chars=torch.full((2, 4), -1), ink_len=torch.tensor([4, 4])))
        (losses[0] + losses[1]).backward()
        for parameter in model.text_encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.text_encoder.queries.grad.abs().sum().item(), 0)
        empty = model(torch.empty(1, 0, dtype=torch.long), torch.tensor([0]), torch.zeros(1, 1, 3))
        self.assertTrue(torch.isfinite(empty).all())

    def test_checkpoint_roundtrip_and_legacy_default(self):
        for conditioning in ("legacy", "unpadded", "transformer"):
            model = Student(vocab_size=9, d_model=16, n_layers=0, conditioning=conditioning, prefix_slots=4)
            config = dict(model.config)
            if conditioning == "legacy":
                for key in ("conditioning", "text_encoder_layers", "text_encoder_heads", "prefix_slots"):
                    del config[key]
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
                path = Path(directory) / "checkpoint.pt"
                torch.save(dict(config=config, model=model.state_dict()), path)
                loaded, _ = load_student(path, "cpu")
            self.assertEqual(loaded.config["conditioning"], conditioning)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name])

    def test_equal_length_batches_cover_corpus_and_are_reproducible(self):
        corpus = Corpus.__new__(Corpus)
        corpus.texts = ["a", "bc", "d", "ef", "gh", "i"]
        corpus.text_lengths = [len(text) for text in corpus.texts]
        corpus.lengths = np.array([2, 3, 4, 5, 6, 7])
        batches = corpus.batches(12, random.Random(3), equal_text_lengths=True)
        self.assertEqual(batches, corpus.batches(12, random.Random(3), equal_text_lengths=True))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(6)))
        for batch in batches:
            self.assertEqual(len({corpus.text_lengths[i] for i in batch}), 1)
            self.assertLessEqual(len(batch) * max(corpus.lengths[batch]), 12)


if __name__ == "__main__":
    unittest.main()

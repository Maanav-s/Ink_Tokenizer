"""Text tokenization for corpus lines."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ink_corpus import PAD, char_tokens, encode_texts, split_text


class TokenizerTests(unittest.TestCase):
    def test_latex_commands_are_single_tokens(self):
        self.assertEqual(split_text(r"\frac{\alpha}{x_1}", "latex"),
                         ["\\frac", "{", "\\alpha", "}", "{", "x", "_", "1", "}"])
        self.assertEqual(split_text(r"\begin{matrix}a\\ b\end{matrix}", "latex"),
                         ["\\begin{matrix}", "a", "\\\\", " ", "b", "\\end{matrix}"])
        self.assertEqual(split_text(r"\{x\}", "latex"), ["\\{", "x", "\\}"])

    def test_char_tokenizer_splits_every_character(self):
        self.assertEqual(split_text(r"\frac", "char"), ["\\", "f", "r", "a", "c"])

    def test_encode_left_pads_by_token_count(self):
        token = char_tokens(["\\frac", "{", "}", "x"])
        text, text_len = encode_texts([r"\frac{x}{x}", "x"], token, "latex")
        torch.testing.assert_close(text_len, torch.tensor([7, 1]))
        self.assertEqual(text[1].tolist(), [PAD] * 6 + [token["x"]])
        self.assertEqual(text[0, 0].item(), token["\\frac"])


if __name__ == "__main__":
    unittest.main()

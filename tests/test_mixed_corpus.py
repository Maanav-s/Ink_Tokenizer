"""CPU checks for converting teacher ink into the merged corpus's conventions."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mixed_corpus import convert_ink, resample


class MixedCorpusTests(unittest.TestCase):
    def test_resampled_points_are_evenly_spaced_and_keep_their_labels(self):
        stroke = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 3.0]])
        xy, labels = resample(stroke, np.array([4, 4, 7]), 0.5)
        np.testing.assert_allclose(np.linalg.norm(np.diff(xy, axis=0), axis=1), 0.5)
        np.testing.assert_allclose(xy[[0, -1]], stroke[[0, -1]])
        self.assertEqual(labels.tolist(), [4, 4, 4, 7, 7, 7, 7, 7, 7])

    def test_converted_ink_keeps_its_strokes_and_scaled_shape(self):
        ink = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 1], [5, 5, 0], [10, 0, 1]], dtype=np.float32)
        offsets, labels = convert_ink(ink, np.array([0, 0, 0, 1, 1]), scale=0.1, spacing=0.25)
        self.assertEqual(len(offsets), len(labels))
        self.assertEqual(int(offsets[:, 2].sum()), 2)
        self.assertEqual(offsets[-1, 2], 1.0)
        np.testing.assert_allclose(offsets[:, :2].sum(axis=0), ink[:, :2].sum(axis=0) * 0.1, atol=1e-6)
        self.assertEqual(sorted(set(labels.tolist())), [0, 1])


if __name__ == "__main__":
    unittest.main()

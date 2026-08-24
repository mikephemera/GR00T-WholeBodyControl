from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.play_clips_g1 import load_qpos_recording


class QposPlaybackLoadingTest(unittest.TestCase):
    def test_loads_recording_without_pickle(self):
        qpos = np.zeros((300, 36), dtype=np.float32)
        qpos[:, 3] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "walk.npz"
            np.savez(
                path,
                qpos=qpos,
                fps=np.asarray(30.0),
                mode_name=np.asarray("walk", dtype=np.str_),
            )
            library = load_qpos_recording(path, expected_nq=36)

        self.assertEqual(library.names, ("walk",))
        self.assertEqual(library.qpos.shape, (1, 300, 36))
        self.assertEqual(library.lengths.tolist(), [300])
        self.assertEqual(library.fps, 30.0)


if __name__ == "__main__":
    unittest.main()

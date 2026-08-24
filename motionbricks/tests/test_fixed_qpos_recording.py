from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from motionbricks.motion_backbone.demo.controllers import fixed_controller
from scripts.interactive_demo_g1 import QPOS_FPS, save_qpos_recording


MOTIONBRICKS_ROOT = Path(__file__).resolve().parents[1]


class FixedQposRecordingTest(unittest.TestCase):
    @staticmethod
    def _load_model():
        model = mujoco.MjModel.from_xml_path(
            str(MOTIONBRICKS_ROOT / "assets/skeletons/g1/scene_29dof.xml")
        )
        model.opt.timestep = 1.0 / QPOS_FPS
        return model

    def test_fixed_controller_emits_constant_walk_command(self):
        controller = fixed_controller(
            mode="walk",
            target_speed_mps=1.5,
            movement_heading=0.0,
            facing_heading=0.0,
            random_seed=0,
            min_token=6,
            max_token=16,
        )
        model = SimpleNamespace(nq=36)
        data = SimpleNamespace(qpos=np.zeros(36, dtype=np.float64))

        first = controller.generate_control_signals(None, model, data, visualize=False)
        second = controller.generate_control_signals(None, model, data, visualize=False)

        for control in (first, second):
            self.assertEqual(control["mode"].item(), 2)
            np.testing.assert_array_equal(control["movement_direction"].numpy(), [[1.0, 0.0, 0.0]])
            np.testing.assert_array_equal(control["facing_direction"].numpy(), [[1.0, 0.0, 0.0]])
            self.assertEqual(control["movement_angle"].item(), 0.0)
            self.assertEqual(control["facing_angle"].item(), 0.0)
            self.assertEqual(control["target_vel"].item(), 0.75)
            self.assertEqual(control["random_seed"].item(), 0)

    def test_qpos_npz_contract_and_no_pickle_reload(self):
        model = self._load_model()
        controller = fixed_controller(mode="walk", target_speed_mps=1.5, random_seed=0)
        qpos = np.zeros((300, 36), dtype=np.float32)
        qpos[:, 2] = 0.78
        qpos[:, 3] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "walk_python_10s.npz"
            save_qpos_recording(output_path, qpos, model, controller)

            with np.load(output_path, allow_pickle=False) as recording:
                self.assertEqual(recording["qpos"].shape, (300, 36))
                self.assertTrue(np.isfinite(recording["qpos"]).all())
                np.testing.assert_allclose(
                    np.linalg.norm(recording["qpos"][:, 3:7], axis=1), 1.0, atol=1e-6
                )
                self.assertEqual(recording["time_s"].shape, (300,))
                np.testing.assert_allclose(np.diff(recording["time_s"]), 1.0 / QPOS_FPS)
                self.assertEqual(recording["fps"].item(), QPOS_FPS)
                self.assertEqual(recording["mode"].item(), 2)
                self.assertEqual(recording["mode_name"].item(), "walk")
                self.assertEqual(recording["target_speed_mps"].item(), 1.5)
                self.assertEqual(recording["movement_heading_rad"].item(), 0.0)
                self.assertEqual(recording["facing_heading_rad"].item(), 0.0)
                self.assertEqual(recording["random_seed"].item(), 0)
                self.assertEqual(recording["joint_names"].shape, (29,))
                self.assertEqual(
                    recording["qpos_layout"].item(),
                    "root_xyz,root_quaternion_wxyz,29_sdk_order_joints",
                )

    def test_qpos_recording_refuses_to_overwrite(self):
        model = self._load_model()
        controller = fixed_controller()
        qpos = np.zeros((1, 36), dtype=np.float32)
        qpos[:, 3] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "existing.npz"
            output_path.touch()
            with self.assertRaises(FileExistsError):
                save_qpos_recording(output_path, qpos, model, controller)


if __name__ == "__main__":
    unittest.main()

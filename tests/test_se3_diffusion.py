import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from se3_diffusion import (
        EsdfDistanceField,
        DiffusionSchedule,
        GuidanceConfig,
        build_model,
        ddim_sample,
        guidance_cost,
        pose7_to_pose9,
        pose9_to_pose7_numpy,
        resample_se3_path,
    )


@unittest.skipIf(torch is None, "PyTorch is installed in the training environment")
class SE3DiffusionTest(unittest.TestCase):

    def test_esdf_linear_field_interpolation_and_gradient(self) -> None:
        grid = np.broadcast_to(
            np.arange(10, dtype=np.float32)[:, None, None],
            (10, 12, 14),
        ).copy()
        field = EsdfDistanceField(
            grid, np.zeros(3, dtype=np.float32), 0.5
        )
        position = torch.tensor(
            [2.3, 1.0, 2.0], dtype=torch.float32, requires_grad=True
        )
        value = field.sample(position)
        value.backward()
        self.assertAlmostEqual(float(value), 4.6, places=5)
        # The field value is the grid index = x / cell, so the gradient is
        # 1 / cell along x.
        np.testing.assert_allclose(position.grad.numpy(), [2.0, 0.0, 0.0])

    def test_esdf_cache_roundtrip(self) -> None:
        import tempfile
        from pathlib import Path

        grid = np.random.default_rng(3).normal(size=(4, 5, 6)).astype(
            np.float32
        )
        field = EsdfDistanceField(
            grid, np.zeros(3, dtype=np.float32), 0.05
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "esdf.npz"
            field.save_cache(path)
            restored = EsdfDistanceField.load_cache(path)
        np.testing.assert_allclose(restored.grid, grid, atol=1e-6)
        self.assertEqual(restored.cell_size, 0.05)
    def test_pose_representation_round_trip(self) -> None:
        pose = np.asarray([
            [0.1, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.3, 1.2, 0.9238795, 0.0, 0.0, 0.3826834],
        ])
        restored = pose9_to_pose7_numpy(pose7_to_pose9(pose))
        np.testing.assert_allclose(restored[:, :3], pose[:, :3], atol=1e-7)
        np.testing.assert_allclose(
            np.abs(np.sum(restored[:, 3:7] * pose[:, 3:7], axis=1)),
            1.0,
            atol=1e-6,
        )

    def test_resampling_preserves_endpoints(self) -> None:
        path = np.asarray([
            [0, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0.9238795, 0, 0, 0.3826834],
            [1, 1, 1, 0.7071068, 0, 0, 0.7071068],
        ], dtype=np.float64)
        result = resample_se3_path(path, 64)
        np.testing.assert_allclose(result[0], path[0], atol=1e-6)
        np.testing.assert_allclose(result[-1], path[-1], atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(result[:, 3:7], axis=1), 1.0, atol=1e-7
        )

    def test_model_shapes_and_gradients(self) -> None:
        for model_type in ("unet", "dit", "dit_cross"):
            with self.subTest(model_type=model_type):
                model = build_model(
                    model_type, sequence_length=64, unet_channels=16,
                    dit_dimension=32, dit_depth=2, dit_heads=4,
                )
                path = torch.randn(2, 64, 9)
                timestep = torch.randint(0, 100, (2,))
                condition = torch.randn(2, 18)
                obstacles = torch.randn(2, 32, 10)
                mask = torch.zeros(2, 32, dtype=torch.bool)
                mask[:, :12] = True
                output = model(path, timestep, condition, obstacles, mask)
                self.assertEqual(output.shape, path.shape)
                output.square().mean().backward()
                self.assertTrue(all(
                    parameter.grad is not None for parameter in model.parameters()
                ))

    def test_guidance_cost_is_differentiable(self) -> None:
        path = torch.randn(2, 64, 9, requires_grad=True)
        obstacles = torch.zeros(32, 10)
        obstacles[0, :6] = torch.tensor(
            [0, 0, 1, 1, 1, 1], dtype=torch.float32
        )
        obstacles[0, 6] = 1.0
        mask = torch.zeros(32, dtype=torch.bool)
        mask[0] = True
        cost = guidance_cost(
            path, torch.zeros(1, 1, 9), torch.ones(1, 1, 9),
            obstacles, mask, torch.tensor([-3.0, -3.5, 0.4]),
            torch.tensor([3.0, 3.5, 3.6]), GuidanceConfig(enabled=True),
        ).sum()
        cost.backward()
        self.assertTrue(bool(torch.isfinite(path.grad).all()))

    def test_velocity_ddim_keeps_hard_endpoints(self) -> None:
        model = build_model("unet", sequence_length=16, unet_channels=16)
        conditions = torch.randn(2, 18)
        obstacles = torch.zeros(32, 10)
        obstacle_mask = torch.zeros(32, dtype=torch.bool)
        obstacles[0, 6] = 1.0
        obstacle_mask[0] = True
        sample = ddim_sample(
            model, conditions,
            DiffusionSchedule.cosine(10, torch.device("cpu")),
            16, 5, obstacles, obstacle_mask,
            torch.zeros(1, 1, 9), torch.ones(1, 1, 9),
            torch.tensor([-3.0, -3.5, 0.4]),
            torch.tensor([3.0, 3.5, 3.6]),
            GuidanceConfig(enabled=False),
            torch.Generator().manual_seed(7),
        )
        self.assertTrue(bool(torch.isfinite(sample).all()))
        torch.testing.assert_close(sample[:, 0], conditions[:, :9])
        torch.testing.assert_close(sample[:, -1], conditions[:, 9:])


if __name__ == "__main__":
    unittest.main()

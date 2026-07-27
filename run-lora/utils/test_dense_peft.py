"""Unit tests for Dense PEFT infrastructure and CurvatureDense variant.

Run with:  cd run-lora/utils && python -m unittest test_dense_peft.py -v
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_peft import (
    DENSE_PEFT_TYPE,
    DenseConfig,
    DenseLayer,
    DenseLinear,
    DenseModel,
    DensePeftModelForCausalLM,
    DenseVariant,
    get_dense_model,
    register_dense_peft_method,
)
from curvature_dense import (
    CurvatureDense,
    apply_curvature_dense,
    get_curvature_dense_model,
)


# ── trivial test model ──────────────────────────────────────────────────────

class TrivialModel(nn.Module):
    def __init__(self, in_features=4, out_features=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


# ── Dense PEFT tests ─────────────────────────────────────────────────────────

class TestDensePEFT(unittest.TestCase):
    def setUp(self):
        register_dense_peft_method()

    def test_dense_linear_creation(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_dense_model(model, config)
        dense_module = peft_model.model.linear
        self.assertIsInstance(dense_module, DenseLinear)
        self.assertIn("default", dense_module.dense_delta)
        self.assertEqual(dense_module.dense_delta["default"].shape, (8, 4))

    def test_dense_zero_init_forward(self):
        """With Delta W = 0, output matches base model."""
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_dense_model(model, config)
        x = torch.randn(2, 4)
        out_base = model(x)
        out_dense = peft_model(x)
        self.assertTrue(torch.allclose(out_base, out_dense, atol=1e-6))

    def test_dense_forward_with_nonzero_delta(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_dense_model(model, config)
        dense_module = peft_model.model.linear

        delta = torch.randn(8, 4)
        dense_module.dense_delta["default"].data.copy_(delta)

        x = torch.randn(2, 4)
        out_dense = peft_model(x)
        # DenseLayer.forward calls base_layer(x) which includes bias.
        # F.linear(x, weight) omits bias, so use the base layer directly.
        base_layer = peft_model.model.linear.get_base_layer()
        expected = base_layer(x) + F.linear(x, delta)
        self.assertTrue(torch.allclose(out_dense, expected, atol=1e-5))

    def test_dense_merge_unmerge_roundtrip(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_dense_model(model, config)
        dense_module = peft_model.model.linear

        delta = torch.randn(8, 4)
        dense_module.dense_delta["default"].data.copy_(delta)

        x = torch.randn(2, 4)
        out_with_delta = peft_model(x)

        # Merge
        dense_module.merge()
        self.assertTrue(dense_module.merged)
        out_merged = peft_model(x)
        self.assertTrue(torch.allclose(out_with_delta, out_merged, atol=1e-5))

        # Unmerge
        dense_module.unmerge()
        self.assertFalse(dense_module.merged)
        out_unmerged = peft_model(x)
        out_base = model(x)
        self.assertTrue(torch.allclose(out_unmerged, out_base, atol=1e-6))

    def test_dense_disable_adapters(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_dense_model(model, config)
        dense_module = peft_model.model.linear

        delta = torch.randn(8, 4)
        dense_module.dense_delta["default"].data.copy_(delta)

        x = torch.randn(2, 4)
        # model.linear was replaced in-place by DenseLinear; use the
        # unwrapped base layer to get the "no adapter" reference output.
        out_base = dense_module.get_base_layer()(x)

        dense_module.enable_adapters(False)
        self.assertTrue(dense_module.disable_adapters)
        out_disabled = peft_model(x)
        self.assertTrue(torch.allclose(out_disabled, out_base, atol=1e-6))

        dense_module.enable_adapters(True)
        self.assertFalse(dense_module.disable_adapters)
        out_enabled = peft_model(x)
        base_layer = peft_model.model.linear.get_base_layer()
        expected = base_layer(x) + F.linear(x, delta)
        self.assertTrue(torch.allclose(out_enabled, expected, atol=1e-5))

    def test_dense_model_state_dict_keys(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_dense_model(model, config)
        state_keys = set(peft_model.state_dict().keys())
        self.assertTrue(any("dense_delta" in k for k in state_keys))

    def test_dense_save_merged_model(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_dense_model(model, config)
        dense_module = peft_model.model.linear

        delta = torch.randn(8, 4)
        dense_module.dense_delta["default"].data.copy_(delta)

        merged = DensePeftModelForCausalLM._merged_base_state_dict(peft_model)
        self.assertIn("linear.weight", merged)
        self.assertEqual(merged["linear.weight"].shape, (8, 4))
        # Merged weight = base + delta
        expected = model.linear.weight + delta
        self.assertTrue(torch.allclose(merged["linear.weight"], expected, atol=1e-5))


# ── CurvatureDense tests ─────────────────────────────────────────────────────

class TestCurvatureDense(unittest.TestCase):
    def setUp(self):
        register_dense_peft_method()

    def test_curvature_dense_buffers_created(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_curvature_dense_model(model, config)
        dense_module = peft_model.model.linear

        self.assertIn("default", dense_module.dense_variant)
        self.assertIsInstance(
            dense_module.dense_variant["default"], CurvatureDense
        )
        self.assertTrue(hasattr(dense_module, "U_in_bar_default"))
        self.assertTrue(hasattr(dense_module, "U_out_bar_default"))
        # Initial buffers are empty (0 columns)
        self.assertEqual(dense_module.U_in_bar_default.shape, (4, 0))
        self.assertEqual(dense_module.U_out_bar_default.shape, (8, 0))

    def test_curvature_dense_empty_projection_is_identity(self):
        """With empty projection buffers, output matches plain Dense."""
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_curvature_dense_model(model, config)
        dense_module = peft_model.model.linear

        delta = torch.randn(8, 4)
        dense_module.dense_delta["default"].data.copy_(delta)

        x = torch.randn(2, 4)
        out_cd = peft_model(x)
        base_layer = peft_model.model.linear.get_base_layer()
        expected = base_layer(x) + F.linear(x, delta)
        self.assertTrue(torch.allclose(out_cd, expected, atol=1e-5))

    def test_curvature_dense_forward_matches_manual_projection(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_curvature_dense_model(model, config)
        dense_module = peft_model.model.linear

        delta = torch.randn(8, 4)
        dense_module.dense_delta["default"].data.copy_(delta)

        # Orthonormal projection bases (project out first 2 dims each side)
        U_in = torch.eye(4)[:, :2]
        U_out = torch.eye(8)[:, :2]
        dense_module._buffers["U_in_bar_default"] = U_in.to(delta)
        dense_module._buffers["U_out_bar_default"] = U_out.to(delta)

        x = torch.randn(2, 4)
        out_model = peft_model(x)

        # Manual: base_layer(x) + (I - U_out U_out^T) @ delta @ (I - U_in U_in^T) @ x
        x_proj = x - (x @ U_in) @ U_in.T
        manual_update = F.linear(x_proj, delta)
        manual_update = manual_update - (manual_update @ U_out) @ U_out.T
        base_layer = peft_model.model.linear.get_base_layer()
        expected = base_layer(x) + manual_update
        self.assertTrue(torch.allclose(out_model, expected, atol=1e-5))

    def test_curvature_dense_get_delta_weight(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_curvature_dense_model(model, config)
        dense_module = peft_model.model.linear

        delta = torch.randn(8, 4)
        U_in = torch.eye(4)[:, :2]
        U_out = torch.eye(8)[:, :2]
        dense_module._buffers["U_in_bar_default"] = U_in.to(delta)
        dense_module._buffers["U_out_bar_default"] = U_out.to(delta)

        projected = CurvatureDense.get_delta_weight(
            dense_module, "default", delta
        )
        # Manual
        expected = delta - U_out @ (U_out.T @ delta)
        expected = expected - (expected @ U_in) @ U_in.T
        self.assertTrue(torch.allclose(projected, expected, atol=1e-5))

    def test_apply_curvature_dense_idempotent_guard(self):
        model = TrivialModel()
        config = DenseConfig(target_modules=["linear"])
        peft_model = get_curvature_dense_model(model, config)
        # Second call should raise
        with self.assertRaises(ValueError):
            apply_curvature_dense(peft_model)


# ── DenseConfig tests ────────────────────────────────────────────────────────

class TestDenseConfig(unittest.TestCase):
    def test_peft_type_is_dense(self):
        config = DenseConfig(target_modules=["q_proj"])
        self.assertEqual(config.peft_type, DENSE_PEFT_TYPE)

    def test_bias_must_be_none(self):
        with self.assertRaises(ValueError):
            DenseConfig(target_modules=["q_proj"], bias="all")

    def test_modules_to_save_not_supported(self):
        with self.assertRaises(ValueError):
            DenseConfig(target_modules=["q_proj"], modules_to_save=["lm_head"])


if __name__ == "__main__":
    unittest.main()

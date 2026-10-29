"""Unit tests for OPLoRA SVD bases, bilateral projection, and fold.

Run with:  cd run-lora/utils && python -m unittest test_op_lora.py -v
"""

import unittest

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from op_lora import (
    OPLora,
    apply_op_lora,
    compute_top_k_bases,
    fold_oplora_projections,
    project_delta,
    subspace_alignment,
)


def _make_full_rank_weight(out_features: int, in_features: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator, dtype=torch.float32)
    # Stretch the spectrum so the top-k directions are well separated.
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    s = torch.linspace(s.numel(), 1.0, s.numel())
    return u @ torch.diag(s) @ vh


class TestOPLoRAMath(unittest.TestCase):
    def setUp(self):
        self.out_features = 8
        self.in_features = 6
        self.k = 2
        self.rank = 3
        self.weight = _make_full_rank_weight(self.out_features, self.in_features)
        self.U_k, self.S_k, self.V_k = compute_top_k_bases(self.weight, self.k)
        generator = torch.Generator().manual_seed(1)
        self.A = torch.randn(self.rank, self.in_features, generator=generator)
        self.B = torch.randn(self.out_features, self.rank, generator=generator)
        self.scaling = 1.0

    def test_top_k_bases_shapes_and_orthonormality(self):
        self.assertEqual(tuple(self.U_k.shape), (self.out_features, self.k))
        self.assertEqual(tuple(self.V_k.shape), (self.in_features, self.k))
        self.assertEqual(tuple(self.S_k.shape), (self.k,))
        torch.testing.assert_close(self.U_k.T @ self.U_k, torch.eye(self.k), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(self.V_k.T @ self.V_k, torch.eye(self.k), atol=1e-5, rtol=1e-5)
        reconstructed = self.U_k @ torch.diag(self.S_k) @ self.V_k.T
        u, s, vh = torch.linalg.svd(self.weight, full_matrices=False)
        torch.testing.assert_close(reconstructed, u[:, : self.k] @ torch.diag(s[: self.k]) @ vh[: self.k, :])

    def test_k_clamped_when_larger_than_min_dim(self):
        U_k, S_k, V_k = compute_top_k_bases(self.weight, k=64)
        min_dim = min(self.out_features, self.in_features)
        self.assertEqual(U_k.shape[1], min_dim)
        self.assertEqual(V_k.shape[1], min_dim)
        self.assertEqual(S_k.numel(), min_dim)

    def test_rho_k_is_zero(self):
        delta = project_delta(self.B, self.A, self.U_k, self.V_k, self.scaling)
        rho = subspace_alignment(delta, self.U_k)
        self.assertLess(rho.item(), 1e-6)

    def test_proposition2_preserves_singular_triplets(self):
        delta = project_delta(self.B, self.A, self.U_k, self.V_k, self.scaling)
        w_prime = self.weight + delta
        torch.testing.assert_close(w_prime @ self.V_k, self.U_k @ torch.diag(self.S_k), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(w_prime.T @ self.U_k, self.V_k @ torch.diag(self.S_k), atol=1e-4, rtol=1e-4)

    def test_zero_b_leaves_base_unchanged(self):
        delta = project_delta(torch.zeros_like(self.B), self.A, self.U_k, self.V_k, self.scaling)
        torch.testing.assert_close(delta, torch.zeros_like(self.weight))

    def test_fold_matches_projected_delta(self):
        a_eff = self.A - (self.A @ self.V_k) @ self.V_k.T
        b_eff = self.B - self.U_k @ (self.U_k.T @ self.B)
        folded = self.scaling * (b_eff @ a_eff)
        projected = project_delta(self.B, self.A, self.U_k, self.V_k, self.scaling)
        torch.testing.assert_close(folded, projected, atol=1e-5, rtol=1e-5)


class TrivialModel(nn.Module):
    def __init__(self, in_features=6, out_features=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.data.copy_(_make_full_rank_weight(out_features, in_features, seed=2))

    def forward(self, x):
        return self.linear(x)


class TestOPLoRAPeft(unittest.TestCase):
    def setUp(self):
        self.k = 2
        self.rank = 3
        self.model = TrivialModel()
        self.base_weight = self.model.linear.weight.detach().clone()
        config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank,
            target_modules=["linear"],
            lora_dropout=0.0,
            init_lora_weights=True,
            bias="none",
        )
        self.peft_model = apply_op_lora(get_peft_model(self.model, config), k=self.k)
        self.module = self.peft_model.model.linear

    def test_apply_attaches_variant_and_buffers(self):
        self.assertIsInstance(self.module.lora_variant["default"], OPLora)
        self.assertTrue(hasattr(self.module, "U_k_default"))
        self.assertTrue(hasattr(self.module, "V_k_default"))
        self.assertEqual(self.module.U_k_default.shape[1], self.k)
        self.assertEqual(self.module.V_k_default.shape[1], self.k)

    def test_forward_matches_projected_delta(self):
        with torch.no_grad():
            self.module.lora_A["default"].weight.normal_()
            self.module.lora_B["default"].weight.normal_()
        x = torch.randn(4, 6)
        y = self.peft_model(x)
        delta = OPLora._compute_delta_weight(self.module, "default")
        expected = x @ (self.base_weight + delta).T
        torch.testing.assert_close(y, expected, atol=1e-4, rtol=1e-4)

    def test_fold_removes_variant_and_preserves_delta(self):
        with torch.no_grad():
            self.module.lora_A["default"].weight.normal_()
            self.module.lora_B["default"].weight.normal_()
        delta_before = OPLora._compute_delta_weight(self.module, "default").detach().clone()
        fold_oplora_projections(self.peft_model)
        self.assertNotIn("default", getattr(self.module, "lora_variant", {}))
        self.assertFalse(hasattr(self.module, "U_k_default"))
        self.assertFalse(hasattr(self.module, "V_k_default"))
        scaling = self.module.scaling["default"]
        delta_after = scaling * (
            self.module.lora_B["default"].weight @ self.module.lora_A["default"].weight
        )
        torch.testing.assert_close(delta_after, delta_before, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

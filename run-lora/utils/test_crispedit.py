"""Focused tests for CrispEdit projection math and the standalone optimizer.

Run from the ``run-lora`` directory::

    python -m pytest run-lora/utils/test_crispedit.py -q
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.crispedit import build_projection_cache, install_gradient_projection  # noqa: E402
from utils.curv_lora import get_energy_threshold  # noqa: E402
from utils.projected_adamw import ProjectedAdamW, project_gradient  # noqa: E402


def _top_heavy_factors(size, decay):
    """Positive-definite factors with a very top-heavy Kronecker spectrum."""
    values = torch.tensor([decay ** i for i in range(size)])
    return torch.diag(values), torch.diag(values)


def _reference_projection(grad, factor_a, factor_b, energy_ratio):
    """The literal original CrispEdit projection, using the full bases."""
    Sa, Ua = torch.linalg.eigh(factor_a.float())
    Sb, Ub = torch.linalg.eigh(factor_b.float())
    kron = torch.outer(Sa.clamp(min=0), Sb.clamp(min=0))
    M_keep = kron < get_energy_threshold(kron.reshape(-1), energy_ratio)
    return Ub @ ((Ub.T @ grad @ Ua) * M_keep.T) @ Ua.T


def test_projection_cache_matches_original_full_basis():
    factor_a, factor_b = _top_heavy_factors(8, 0.5)
    torch.manual_seed(0)
    grad = torch.randn(8, 8)
    cache = build_projection_cache(factor_a, factor_b, energy_ratio=0.9)

    assert torch.allclose(
        project_gradient(grad, cache),
        _reference_projection(grad, factor_a, factor_b, 0.9),
        atol=1e-4,
    )


def test_projection_cache_stores_only_the_small_corner():
    size = 16
    factor_a, factor_b = _top_heavy_factors(size, 0.25)
    cache = build_projection_cache(factor_a, factor_b, energy_ratio=0.9)

    assert cache["Ua"].shape[0] == size and cache["Ub"].shape[0] == size
    assert 0 < cache["Ua"].shape[1] <= size // 4, "the stored basis must be small"
    assert 0 < cache["Ub"].shape[1] <= size // 4
    assert cache["M"].dtype == torch.bool
    assert cache["M"].shape == (cache["Ua"].shape[1], cache["Ub"].shape[1])
    assert torch.allclose(
        cache["Ua"].T @ cache["Ua"], torch.eye(cache["Ua"].shape[1]), atol=1e-5
    )


def test_projection_cache_dtype():
    factor_a, factor_b = _top_heavy_factors(8, 0.5)
    cache = build_projection_cache(factor_a, factor_b, 0.9, dtype=torch.bfloat16)
    assert cache["Ua"].dtype == torch.bfloat16 and cache["Ub"].dtype == torch.bfloat16
    assert cache["M"].dtype == torch.bool


def test_project_gradient_ignores_non_2d():
    cache = build_projection_cache(*_top_heavy_factors(4, 0.5), energy_ratio=0.9)
    grad = torch.tensor([1.0, 2.0, 3.0])
    assert project_gradient(grad, cache) is grad


def test_gradient_hook_projects_before_accumulation():
    """The hook must project the gradient DeepSpeed later reduces.

    ``Tensor.register_hook`` is a pre-accumulate hook, while DeepSpeed reads the
    accumulated ``.grad`` from a ``register_post_accumulate_grad_hook`` (see
    ``deepspeed/utils/torch.py::register_grad_hook``), so this test reproduces
    that ordering: the DeepSpeed-like post-accumulate hook must observe the
    already-projected gradient.
    """
    factor_a, factor_b = _top_heavy_factors(4, 0.5)
    cache = build_projection_cache(factor_a, factor_b, energy_ratio=0.9)

    torch.manual_seed(0)
    plain = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        plain.weight.copy_(torch.randn(4, 4))
    projected = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        projected.weight.copy_(plain.weight)
    install_gradient_projection({"lin": projected}, {"lin": cache})
    seen_by_deepspeed_like_hook = {}
    projected.weight.register_post_accumulate_grad_hook(
        lambda p: seen_by_deepspeed_like_hook.update(seen=p.grad.clone())
    )

    x = torch.ones(3, 4)
    plain(x).sum().backward()
    projected(x).sum().backward()
    expected = project_gradient(plain.weight.grad.clone(), cache)

    assert torch.allclose(projected.weight.grad, expected, atol=1e-5)
    assert torch.allclose(seen_by_deepspeed_like_hook["seen"], expected, atol=1e-5)


def test_projected_adamw():
    cache = build_projection_cache(*_top_heavy_factors(4, 0.5), energy_ratio=0.9)
    p1 = nn.Parameter(torch.zeros(4, 4))
    p2 = nn.Parameter(torch.zeros(4, 4))
    p3 = nn.Parameter(torch.zeros(3))
    p1.grad = torch.arange(16.0).reshape(4, 4)
    p2.grad = torch.arange(16.0).reshape(4, 4) + 1
    p3.grad = torch.tensor([1.0, 2.0, 3.0])
    g1, g2, g3 = p1.grad.clone(), p2.grad.clone(), p3.grad.clone()

    ProjectedAdamW([p1, p2, p3], projection_cache_map={p1: cache}, lr=0.1).step()

    assert torch.allclose(p1.grad, project_gradient(g1, cache))
    assert torch.equal(p2.grad, g2)  # unmapped
    assert torch.equal(p3.grad, g3)  # 1-D

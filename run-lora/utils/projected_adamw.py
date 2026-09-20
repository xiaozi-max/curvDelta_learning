"""ProjectedAdamW — AdamW that projects mapped 2-D gradients before stepping.

This is the original CrispEdit ``ProjectedAdam``: ``step()`` rewrites ``p.grad``
in place with the Kronecker-eigenbasis map before AdamW sees it. The original
writes the map with the *kept* (low-curvature) mask::

    grad_proj = U_B @ ((U_B.T @ grad @ U_A) * M_keep.T) @ U_A.T

Because ``Ua``/``Ub`` are complete orthonormal bases and the two masks are
complementary (``M_keep + M_drop = 1``), the same operator is also::

    grad_proj = grad - U_B @ ((U_B.T @ grad @ U_A) * M_drop.T) @ U_A.T

and that second form is what this helper evaluates. It is not a different
algorithm: keeping the low-curvature directions is exactly removing the
high-curvature ones. The reason for writing it this way is that at
``energy_ratio=0.9`` the kept mask is almost entirely True, so the dropped
corner is the small one; ``utils/crispedit.py`` stores only that corner's
suffix bases, and neither the full ``Ua``/``Ub`` nor ``M_keep`` is ever
materialised.

``cache`` is a dict of ``Ua``, ``Ub`` (orthonormal eigenvectors of the K-FAC A/B
factors) and a boolean ``M`` (True = high-curvature Kronecker pair to remove).
"""

import torch


def project_gradient(grad, cache):
    """Remove the mapped component of a 2-D gradient; non-2-D is unchanged."""
    if grad.ndim != 2:
        return grad
    Ua = cache["Ua"].to(device=grad.device, dtype=grad.dtype)
    Ub = cache["Ub"].to(device=grad.device, dtype=grad.dtype)
    M = cache["M"].to(device=grad.device, dtype=grad.dtype)
    return grad - Ub @ ((Ub.T @ grad @ Ua) * M.T) @ Ua.T


class ProjectedAdamW(torch.optim.AdamW):
    """AdamW that projects each mapped 2-D gradient inside ``step()``."""

    def __init__(self, params, projection_cache_map=None, **kwargs):
        super().__init__(params, **kwargs)
        self.set_projection_cache(projection_cache_map)

    def set_projection_cache(self, projection_cache_map):
        self._projection_cache_map = projection_cache_map or {}
        for group in self.param_groups:
            group["projection_cache_map"] = self._projection_cache_map

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                cache = self._projection_cache_map.get(p)
                if grad is None or grad.ndim != 2 or cache is None:
                    continue
                p.grad.copy_(project_gradient(grad, cache))
        return super().step(closure)

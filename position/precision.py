"""Precision invariants for fixed positional buffers."""

from __future__ import annotations

from collections.abc import Callable

import torch


class PreserveFP32BuffersMixin:
    """Keep named fixed buffers in fp32 across module-wide dtype casts.

    ``Module.half()`` and ``Module.bfloat16()`` normally convert floating-point
    buffers together with parameters.  That is unsafe for cached positional
    tables: converting an fp32 table to a narrow dtype and back cannot recover
    the position/frequency precision that was lost.  Subclasses name the fixed
    buffers that must follow device moves while remaining fp32.
    """

    _fp32_buffer_names: tuple[str, ...] = ()

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ):
        originals = {
            name: self._buffers.get(name)
            for name in self._fp32_buffer_names
            if name in self._buffers
        }
        module = super()._apply(fn, recurse=recurse)
        for name, original in originals.items():
            transformed = self._buffers.get(name)
            if original is None or transformed is None:
                continue
            # Use the original fp32 values, not the transformed narrow tensor,
            # so an explicit .half()/.bfloat16() cannot quantize the cache.
            self._buffers[name] = original.to(
                device=transformed.device,
                dtype=torch.float32,
            )
        return module
